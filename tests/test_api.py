"""Integration checks against the real supplied models, not mocked scores."""

import json
from html.parser import HTMLParser

from fastapi.testclient import TestClient
import pytest

from backend.app import app
from backend.service import PROJECT_ROOT
from backend.schemas import Transaction


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as session:
        yield session


@pytest.fixture
def transaction():
    return json.loads((PROJECT_ROOT / "example_frontend_transaction.json").read_text())


def test_model_is_ready_and_metadata_uses_trained_categories(client):
    assert client.get("/api/health").json()["status"] == "ready"
    response = client.get("/api/metadata")
    assert response.status_code == 200
    metadata = response.json()
    assert "travel" in metadata["options"]["merchant_category"]
    assert "pos_terminal" in metadata["options"]["device_type"]
    assert len(metadata["input_schema"]["fields"]) == 14
    assert 0 <= metadata["thresholds"]["review"] < metadata["thresholds"]["high"] <= 1


def test_prediction_matches_original_export_including_explanation(client, transaction):
    expected = json.loads((PROJECT_ROOT / "example_frontend_prediction.json").read_text())
    response = client.post("/api/predict", json=transaction)
    assert response.status_code == 200
    actual = response.json()
    assert actual["fraud_probability"] == pytest.approx(expected["fraud_probability"], rel=1e-6)
    assert actual["risk_level"] == expected["risk_level"]
    assert actual["fraud_probability"] + actual["legitimate_probability"] == pytest.approx(1)
    assert actual["fraud_type_probabilities_given_fraud"] == pytest.approx(expected["fraud_type_probabilities_given_fraud"])
    assert sum(actual["overall_probabilities_by_fraud_type"].values()) == pytest.approx(actual["fraud_probability"])
    assert actual["explanation"]["reconstructed_fraud_probability"] == pytest.approx(actual["fraud_probability"], rel=2e-5)
    assert actual["safe_for_automatic_decision"] is False
    assert actual["explanation"]["risk_increasing_characteristics"][0]["characteristic_group"] == "Amount and deviation"


def test_samples_return_different_real_scores(client):
    metadata = client.get("/api/metadata").json()
    predictions = []
    for sample in metadata["samples"]:
        response = client.post("/api/predict", json=sample["transaction"])
        assert response.status_code == 200
        predictions.append(response.json())
    assert predictions[-1]["fraud_probability"] > predictions[0]["fraud_probability"]
    assert predictions[1]["risk_level"] == "Review"
    assert predictions[-1]["risk_level"] == "High"


def test_binary_model_matches_native_export_on_all_samples(client):
    import numpy as np
    import pandas as pd
    from xgboost import Booster, DMatrix
    from fraud_banking_explainable_v3 import calibrate, make_features

    native = Booster()
    native.load_model(PROJECT_ROOT / "binary_xgboost_booster.json")
    bundle = client.app.state.bundle
    samples = client.get("/api/metadata").json()["samples"]
    frame = pd.DataFrame([sample["transaction"] for sample in samples])
    matrix = bundle["preprocessor"].transform(make_features(frame))
    expected = calibrate(native.predict(DMatrix(matrix)), bundle["platt"])
    actual = [client.post("/api/predict", json=sample["transaction"]).json()["fraud_probability"] for sample in samples]
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


@pytest.mark.parametrize("field,value", [
    ("amount", -1),
    ("ip_risk_score", 101),
    ("ip_risk_score", "NaN"),
    ("amount", "Infinity"),
    ("velocity_1h", 1.5),
    ("device_known", 2),
    ("merchant_country", "USA"),
    ("merchant_category", "   "),
    ("timestamp", "2024-09-12T12:30:00"),
    ("timestamp", "not-a-date"),
])
def test_invalid_inputs_are_rejected(client, transaction, field, value):
    transaction[field] = value
    assert client.post("/api/predict", json=transaction).status_code == 422


def test_missing_context_and_unexpected_fields_are_rejected(client, transaction):
    del transaction["device_known"]
    assert client.post("/api/predict", json=transaction).status_code == 422
    transaction["device_known"] = 1
    transaction["is_fraud"] = 1
    assert client.post("/api/predict", json=transaction).status_code == 422


def test_timezone_offsets_are_equivalent(client, transaction):
    utc = client.post("/api/predict", json=transaction).json()
    transaction["timestamp"] = "2024-09-12T14:18:38+05:30"
    offset = client.post("/api/predict", json=transaction).json()
    assert offset["fraud_probability"] == utc["fraud_probability"]


def test_model_files_are_not_served(client):
    assert client.get("/fraud_banking_explainable_v3.pkl").status_code == 404
    assert client.get("/fraud_banking_explainable_v3.joblib").status_code == 404


class FormParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.fields = {}
        self.labels = set()
        self.scripts = []
        self.stylesheets = []
        self.context_open = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in {"input", "select"} and attrs.get("name"):
            self.fields[attrs["name"]] = attrs
        if tag == "label":
            self.labels.add(attrs.get("for"))
        if tag == "script":
            self.scripts.append(attrs.get("src"))
        if tag == "link" and attrs.get("rel") == "stylesheet":
            self.stylesheets.append(attrs.get("href"))
        if tag == "details" and attrs.get("id") == "account-details":
            self.context_open = "open" in attrs


def test_page_and_assets_are_served_without_a_build(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    page = FormParser()
    page.feed(response.text)
    assert page.scripts == ["/static/app.js"]
    assert page.stylesheets == ["/static/styles.css"]
    for path in page.scripts + page.stylesheets:
        asset = client.get(path)
        assert asset.status_code == 200
        assert len(asset.content) > 100


def test_html_form_matches_the_required_api_inputs(client):
    page = FormParser()
    page.feed(client.get("/").text)
    assert set(page.fields) == set(Transaction.model_fields)
    assert page.context_open is False
    for name, attrs in page.fields.items():
        assert "required" in attrs
        assert attrs["id"] in page.labels
        if attrs.get("type") == "number":
            assert attrs["min"] == "0"
    assert page.fields["ip_risk_score"]["max"] == "100"
    assert page.fields["velocity_1h"]["step"] == "1"


def test_static_routes_do_not_expose_project_files(client):
    assert client.get("/static/requirements.txt").status_code == 404
    assert client.get("/static/fraud_banking_explainable_v3.pkl").status_code == 404
    assert client.get("/static/package.json").status_code == 404
    assert client.get("/assets/missing.js").status_code == 404
