"""Load the local trusted bundle and connect it to the web application."""

import json
from pathlib import Path

from backend.model_loader import load_portable_bundle
from fraud_banking_explainable_v3 import predict_api

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "fraud_banking_explainable_v3.pkl"


def load_bundle() -> dict:
    """Load once at startup. No model paths or uploads are accepted from users."""
    bundle = load_portable_bundle(MODEL_PATH)
    if bundle.get("version") != "banking_explainable_v3":
        raise ValueError("Expected the banking_explainable_v3 inference bundle.")
    # A small local app does not need every CPU thread for a single transaction.
    def limit_threads(item):
        if isinstance(item, dict):
            for child in item.values():
                limit_threads(child)
        elif hasattr(item, "get_booster"):
            item.set_params(n_jobs=1)

    limit_threads(bundle)
    return bundle


def get_samples() -> list[dict]:
    """Return editable synthetic examples; labels describe inputs, not outcomes."""
    original = json.loads((PROJECT_ROOT / "example_frontend_transaction.json").read_text())
    elevated = {
        **original,
        "timestamp": "2024-09-12T23:30:00Z",
        "amount": 1200,
        "card_present": 0,
        "device_type": "web_browser",
        "device_known": 0,
        "ip_risk_score": 35,
        "is_foreign_txn": 1,
        "time_since_last_s": 60,
        "velocity_1h": 2,
        "amount_vs_avg_ratio": 3,
        "account_age_days": 180,
    }
    unusual = {
        **elevated,
        "timestamp": "2024-09-12T02:15:00Z",
        "amount": 4800,
        "ip_risk_score": 97,
        "time_since_last_s": 10,
        "velocity_1h": 12,
        "amount_vs_avg_ratio": 12,
        "account_age_days": 7,
        "has_2fa": 0,
    }
    return [
        {"id": "original", "name": "Card payment", "description": "A card payment with example account history.", "transaction": original},
        {"id": "unfamiliar", "name": "Online purchase", "description": "An international online purchase using a new device.", "transaction": elevated},
        {"id": "unusual", "name": "Rapid payments", "description": "Several payments close together from a recently opened account.", "transaction": unusual},
    ]


def get_metadata(bundle: dict) -> dict:
    """Expose safe model information and the actual fitted category choices."""
    categorical = bundle["preprocessor"].named_transformers_["cat"]
    encoder = categorical.named_steps["encode"]
    options = {
        name: [str(value) for value in values]
        for name, values in zip(bundle["model_categorical"], encoder.categories_)
        if name in {"merchant_category", "merchant_country", "device_type"}
    }
    return {
        "version": bundle["version"],
        "model_name": bundle["model_name"],
        "currency": "USD",
        "synthetic_only": bundle["synthetic_only"],
        "thresholds": {"review": bundle["review_threshold"], "high": bundle["high_threshold"]},
        "options": options,
        "samples": get_samples(),
        "input_schema": json.loads((PROJECT_ROOT / "frontend_backend_input_schema.json").read_text()),
    }


def assess_transaction(bundle: dict, record: dict) -> dict:
    """Run the real saved pipeline, calibration, subtype ranking, and TreeSHAP."""
    return predict_api(bundle, record, strict=True, explain=True)
