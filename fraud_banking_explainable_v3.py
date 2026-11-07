"""Inference helpers for the saved banking-fraud pipeline."""
import numpy as np
import pandas as pd
from scipy.special import expit, logit, softmax
from scipy.sparse import hstack, csr_matrix
from xgboost import DMatrix

SUBTYPE_MIN_CONFIDENCE = 0.55
SUBTYPE_MIN_MARGIN = 0.10


RAW_ALLOWED = [
    "amount", "merchant_category", "merchant_country", "card_present",
    "device_type", "device_known", "ip_risk_score", "is_foreign_txn",
    "time_since_last_s", "velocity_1h", "amount_vs_avg_ratio",
    "account_age_days", "has_2fa",
]

RAW_CATEGORIES = ["merchant_category", "merchant_country", "device_type"]

NUMERIC = [
    "amount", "log_amount", "ip_risk_score", "time_since_last_s",
    "log_time_since_last_s", "velocity_1h", "amount_vs_avg_ratio",
    "log_amount_ratio", "account_age_days", "card_present", "device_known",
    "is_foreign_txn", "has_2fa", "hour", "weekday", "month", "hour_sin",
    "hour_cos", "weekday_sin", "weekday_cos", "is_night", "is_weekend",
    "unknown_device", "no_2fa", "high_velocity", "high_ip_risk",
    "large_amount_ratio", "new_account", "round_amount",
]

CATEGORICAL = RAW_CATEGORIES + [
    "amount_band", "ratio_band", "ip_band", "velocity_band", "hour_period",
    "account_age_band", "device_known_level", "is_foreign_level",
    "has_2fa_level", "card_present_level",
]

FEATURE_GROUPS = {
    "Amount and deviation": ["amount", "log_amount", "amount_vs_avg_ratio", "log_amount_ratio",
                             "large_amount_ratio", "round_amount", "amount_band", "ratio_band"],
    "Velocity": ["time_since_last_s", "log_time_since_last_s", "velocity_1h", "high_velocity", "velocity_band"],
    "IP risk": ["ip_risk_score", "high_ip_risk", "ip_band"],
    "Device": ["device_type", "device_known", "unknown_device", "device_known_level"],
    "Authentication": ["has_2fa", "no_2fa", "has_2fa_level"],
    "Merchant geography": ["merchant_country", "is_foreign_txn", "is_foreign_level"],
    "Merchant category": ["merchant_category"],
    "Card channel": ["card_present", "card_present_level"],
    "Account age": ["account_age_days", "new_account", "account_age_band"],
    "Time": ["hour", "weekday", "month", "hour_sin", "hour_cos", "weekday_sin", "weekday_cos",
             "is_night", "is_weekend", "hour_period"],
}

REQUIRED_API_FIELDS = ["timestamp"] + RAW_ALLOWED

NUMERIC_API_FIELDS = ["amount", "ip_risk_score", "time_since_last_s", "velocity_1h",
                      "amount_vs_avg_ratio", "account_age_days"]

BINARY_API_FIELDS = ["card_present", "device_known", "is_foreign_txn", "has_2fa"]

CATEGORICAL_API_FIELDS = ["merchant_category", "merchant_country", "device_type"]


def standardize_columns(frame):
    copy = frame.copy()
    copy.columns = [str(c).strip().lower().replace(" ", "_") for c in copy.columns]
    return copy


def make_features(frame):
    raw = standardize_columns(frame)
    out = pd.DataFrame(index=raw.index)

    for name in NUMERIC:
        out[name] = np.nan

    def numeric(name):
        if name not in raw:
            return pd.Series(np.nan, index=raw.index, dtype=float)
        s = raw[name]
        if s.dtype == object or str(s.dtype).startswith("string"):
            s = s.astype("string").str.lower().str.strip().replace({
                "true": "1", "false": "0", "yes": "1", "no": "0",
            })
        return pd.to_numeric(s, errors="coerce").astype(float)

    amount = numeric("amount").clip(lower=0)
    out["amount"] = amount
    out["log_amount"] = np.log1p(amount)
    out["ip_risk_score"] = numeric("ip_risk_score")
    out["time_since_last_s"] = numeric("time_since_last_s").clip(lower=0)
    out["log_time_since_last_s"] = np.log1p(out["time_since_last_s"])
    out["velocity_1h"] = numeric("velocity_1h").clip(lower=0)
    out["amount_vs_avg_ratio"] = numeric("amount_vs_avg_ratio").clip(lower=0)
    out["log_amount_ratio"] = np.log1p(out["amount_vs_avg_ratio"].clip(upper=1e5))
    out["account_age_days"] = numeric("account_age_days").clip(lower=0)
    for col in ["card_present", "device_known", "is_foreign_txn", "has_2fa"]:
        out[col] = numeric(col)

    date = pd.to_datetime(raw["timestamp"], errors="coerce", utc=True)
    out["hour"] = date.dt.hour.astype(float)
    out["weekday"] = date.dt.dayofweek.astype(float)
    out["month"] = date.dt.month.astype(float)
    for col, period in [("hour", 24), ("weekday", 7)]:
        out[col + "_sin"] = np.sin(2 * np.pi * out[col] / period)
        out[col + "_cos"] = np.cos(2 * np.pi * out[col] / period)

    # Keep missing values missing, instead of silently treating missing as safe.
    def boolean_or_missing(x, condition):
        return pd.Series(np.where(x.isna(), np.nan, condition.astype(float)), index=raw.index)
    out["is_night"] = boolean_or_missing(out["hour"], out["hour"] < 6)
    out["is_weekend"] = boolean_or_missing(out["weekday"], out["weekday"] >= 5)
    out["unknown_device"] = boolean_or_missing(out["device_known"], out["device_known"] == 0)
    out["no_2fa"] = boolean_or_missing(out["has_2fa"], out["has_2fa"] == 0)
    out["high_velocity"] = boolean_or_missing(out["velocity_1h"], out["velocity_1h"] > 3)
    out["high_ip_risk"] = boolean_or_missing(out["ip_risk_score"], out["ip_risk_score"] >= 70)
    out["large_amount_ratio"] = boolean_or_missing(out["amount_vs_avg_ratio"], out["amount_vs_avg_ratio"] > 5)
    out["new_account"] = boolean_or_missing(out["account_age_days"], out["account_age_days"] <= 30)
    out["round_amount"] = np.where(amount.notna(), np.isclose(amount % 100, 0), np.nan)
    for col in RAW_CATEGORIES:
        if col in raw:
            values = raw[col].astype("string").str.strip().fillna("Unknown")
            out[col] = values.replace("", "Unknown").astype(str)
        else:
            out[col] = "Unknown"

    def levels(x, bins, labels):
        return pd.cut(x, bins=bins, labels=labels, include_lowest=True).astype("string").fillna("Missing").astype(str)

    out["amount_band"] = levels(amount, [-np.inf, 10, 50, 100, 500, 1000, 5000, np.inf],
                                 ["<=10", "10-50", "50-100", "100-500", "500-1k", "1k-5k", ">5k"])
    out["ratio_band"] = levels(out["amount_vs_avg_ratio"], [-np.inf, 0.5, 1, 2, 5, np.inf],
                                ["<=0.5", "0.5-1", "1-2", "2-5", ">5"])
    out["ip_band"] = levels(out["ip_risk_score"], [-np.inf, 25, 50, 70, 85, np.inf],
                             ["<=25", "25-50", "50-70", "70-85", ">85"])
    out["velocity_band"] = levels(out["velocity_1h"], [-np.inf, 0, 1, 3, 6, np.inf],
                                   ["0", "1", "2-3", "4-6", ">6"])
    out["account_age_band"] = levels(out["account_age_days"], [-np.inf, 7, 30, 180, 365, np.inf],
                                      ["<=7d", "8-30d", "31-180d", "181-365d", ">365d"])
    out["hour_period"] = levels(out["hour"], [-np.inf, 5, 11, 17, 21, np.inf],
                                 ["night", "morning", "afternoon", "evening", "late_evening"])
    for raw_col, level_col in [("device_known", "device_known_level"),
                                ("is_foreign_txn", "is_foreign_level"),
                                ("has_2fa", "has_2fa_level"),
                                ("card_present", "card_present_level")]:
        out[level_col] = out[raw_col].map({0.0: "No", 1.0: "Yes"}).fillna("Missing")

    return out[NUMERIC + CATEGORICAL].replace([np.inf, -np.inf], np.nan).copy()


def predict_positive(model, X):
    return np.asarray(model.predict_proba(X)[:, 1], dtype=float)


def pair_strings(frame, pairs):
    return pd.DataFrame({f"{a}__{b}": (frame[a].astype(str) + "|" + frame[b].astype(str))
                         for a, b in pairs}, index=frame.index)


def build_sparse_matrix(base_matrix, frame, pairs, enc):
    if not pairs:
        return base_matrix
    extra = enc.transform(pair_strings(frame, pairs))
    return hstack([base_matrix, extra], format="csr")


def calibrate(p, slope_intercept):
    a, b = slope_intercept
    return expit(a * logit(np.clip(np.asarray(p, dtype=float), 1e-6, 1-1e-6)) + b)


def model_input(bundle, frame):
    f = make_features(frame)
    main = bundle["preprocessor"].transform(f)
    if bundle["model_name"] == "Sparse interactions":
        mat = build_sparse_matrix(main, f, bundle["pairs"], bundle["pair_encoder"])
    elif bundle["model_name"] == "XGBoost + bank anomaly":
        numeric = bundle["anomaly_preprocessor"].transform(f[bundle["anomaly_columns"]])
        score = -bundle["anomaly_model"].score_samples(numeric)
        mat = hstack([main, csr_matrix(score.reshape(-1, 1))], format="csr")
    else:
        mat = main
    return mat, f, main


def predict_fraud_transactions(bundle, raw_records):
    """Low-level batch inference from a saved model bundle (calibrated scores)."""
    matrix, f, main = model_input(bundle, raw_records)
    fraud = calibrate(predict_positive(bundle["model"], matrix), bundle["platt"])
    risk = np.where(fraud >= bundle["high_threshold"], "High",
                    np.where(fraud >= bundle["review_threshold"], "Review", "Low"))
    result = pd.DataFrame({"fraud_probability": fraud,
                           "legitimate_probability": 1-fraud,
                           "risk_level": risk,
                           "review_recommended": fraud >= bundle["review_threshold"]}, index=raw_records.index)
    if bundle.get("type_model") is not None:
        conditional = calibrate_type_probabilities(
            type_probabilities(bundle["type_model"], main), bundle.get("type_calibration"))
        high_idx = np.argmax(conditional, axis=1)
        result["most_likely_fraud_type_if_fraud"] = bundle["type_encoder"].inverse_transform(high_idx)
        result["type_probability_given_fraud"] = conditional[np.arange(len(high_idx)), high_idx]
        result["joint_probability_of_top_fraud_type"] = fraud * result["type_probability_given_fraud"].values
    return result


def calibrate_type_probabilities(prob, params):
    prob = np.clip(np.asarray(prob, dtype=float), 1e-8, 1.0)
    if params is None:
        return prob / np.maximum(prob.sum(axis=1, keepdims=True), 1e-12)
    z = params["temperature_slope"] * np.log(prob) + np.asarray(params["class_bias"])
    return softmax(z, axis=1)


def family_for_raw_feature(col):
    for group, columns in FEATURE_GROUPS.items():
        if col in columns:
            return group
    return "Other"


def raw_feature_for_encoded(name):
    if name.startswith("num__"):
        return name[len("num__"):]
    if name.startswith("cat__"):
        suffix = name[len("cat__"):]
        for feat in sorted(CATEGORICAL, key=len, reverse=True):
            if suffix == feat or suffix.startswith(feat + "_"):
                return feat
        return suffix
    if name == "bank_anomaly_score":
        return "Anomaly"
    return "Interaction"


def family_for_encoded(name):
    source = raw_feature_for_encoded(name)
    if source == "Anomaly":
        return "Anomaly"
    if source == "Interaction":
        return "Interaction"
    return family_for_raw_feature(source)


def get_feature_importance_contributions(bundle, matrix):
    model = bundle["model"]
    if bundle["model_name"] == "Sparse interactions":
        return matrix.multiply(model.coef_[0]).toarray(), float(model.intercept_[0])
    raw = model.get_booster().predict(DMatrix(matrix), pred_contribs=True)
    return raw[:, :-1], raw[:, -1]


def validate_api_inputs(raw, strict=True):
    if isinstance(raw, dict):
        raw = pd.DataFrame([raw])
    if not isinstance(raw, pd.DataFrame):
        raise TypeError("Supply a dictionary or a pandas DataFrame of transactions.")
    data = standardize_columns(raw)
    if len(data) == 0:
        raise ValueError("At least one transaction is required.")
    if "timestamp" not in data:
        raise ValueError("timestamp is required, e.g. 2024-07-01T12:30:00Z")
    ts = pd.to_datetime(data["timestamp"], errors="coerce", utc=True)
    if ts.isna().any():
        raise ValueError("Every timestamp must be a parseable ISO 8601 date/time.")
    if "amount" not in data:
        raise ValueError("amount is required")
    for col in NUMERIC_API_FIELDS + BINARY_API_FIELDS:
        if col in data:
            converted = pd.to_numeric(data[col].replace({"True": 1, "False": 0,
                                                          "true": 1, "false": 0}), errors="coerce")
            invalid = data[col].notna() & converted.isna()
            if invalid.any():
                raise ValueError(f"Invalid nonnumeric input in {col}")
            data[col] = converted
    if data.amount.isna().any() or (data.amount < 0).any():
        raise ValueError("amount must be a nonnegative number")
    for col in BINARY_API_FIELDS:
        if col in data and (~data[col].isin([0, 1]) & data[col].notna()).any():
            raise ValueError(f"{col} must be 0 or 1")
    if "ip_risk_score" in data and ((data.ip_risk_score < 0) | (data.ip_risk_score > 100)).any():
        raise ValueError("ip_risk_score must be in [0, 100]")
    for col in ["time_since_last_s", "velocity_1h", "amount_vs_avg_ratio", "account_age_days"]:
        if col in data and (data[col] < 0).any():
            raise ValueError(f"{col} cannot be negative")
    if "velocity_1h" in data and (data.velocity_1h.dropna() % 1 != 0).any():
        raise ValueError("velocity_1h must be an integer count")
    missing_by_row = []
    for i in range(len(data)):
        missing_by_row.append([field for field in REQUIRED_API_FIELDS
                               if field not in data or pd.isna(data.iloc[i][field]) or
                               (field in CATEGORICAL_API_FIELDS and str(data.iloc[i][field]).strip() == "")])
    if strict and any(missing_by_row):
        raise ValueError("Missing required bank context: " + ", ".join(sorted(set(sum(missing_by_row, []))))
                         + ". For a provisional estimate only, set strict=False.")
    # Add absent fields explicitly; make_features preserves their NaNs via preprocessing.
    for name in REQUIRED_API_FIELDS:
        if name not in data:
            data[name] = np.nan
    data["timestamp"] = ts
    return data, missing_by_row


def predict_api(bundle, record, *, strict=True, explain=True, top=5):
    """Return JSON-ready prediction(s) for FastAPI or a manual banking form.

    In strict mode, all real-time and historical indicators must be present.
    In provisional mode, missing inputs are disclosed and no automated action
    is recommended, because model predictions can be severely degraded.
    """
    data, missing_rows = validate_api_inputs(record, strict=strict)
    output = predict_fraud_transactions(bundle, data)
    result = []
    raw_features = make_features(data)
    main = bundle["preprocessor"].transform(raw_features)
    conditional = None
    if bundle.get("type_model") is not None:
        conditional = calibrate_type_probabilities(
            type_probabilities(bundle["type_model"], main), bundle.get("type_calibration"))
    for i in range(len(data)):
        row = output.iloc[i]
        p = float(row["fraud_probability"])
        complete = not missing_rows[i]
        risk = str(row["risk_level"]) if complete else "Incomplete input"
        action = ({"High": "Request manual verification before completing transaction",
                   "Review": "Send for manual risk review",
                   "Low": "No model-triggered intervention"}.get(risk, "Request missing bank context"))
        item = {
            "fraud_probability": p, "legitimate_probability": float(1-p),
            "fraud_probability_percent": round(100*p, 4),
            "risk_level": risk,
            "model_alert": bool(p >= bundle["high_threshold"]) if complete else None,
            "recommended_action": action,
            "input_complete": complete,
            "missing_inputs": missing_rows[i],
            "safe_for_automatic_decision": False,  # Synthetic benchmark is NOT bank-certified.
            "synthetic_data_notice": "Demonstration model trained on generated fraud, not approved for autonomous bank decisions",
            "fraud_type_probabilities_given_fraud": {},
            "overall_probabilities_by_fraud_type": {},
        }
        if conditional is not None:
            probs = conditional[i]
            names = bundle["type_encoder"].classes_.tolist()
            item["fraud_type_probabilities_given_fraud"] = {name: float(v) for name, v in zip(names, probs)}
            item["overall_probabilities_by_fraud_type"] = {name: float(p*v) for name, v in zip(names, probs)}
            idx = int(np.argmax(probs))
            item["most_likely_type_if_fraud"] = names[idx]
            item["type_probability_given_fraud"] = float(probs[idx])
            item["joint_probability_of_most_likely_type"] = float(p * probs[idx])
            ordered = np.argsort(probs)[::-1][:min(3, len(probs))]
            item["top_three_fraud_types"] = [
                {"fraud_type": names[j],
                 "probability_given_fraud": float(probs[j]),
                 "overall_transaction_probability": float(p * probs[j])}
                for j in ordered
            ]
            margin = float(probs[ordered[0]] - probs[ordered[1]]) if len(ordered) > 1 else 1.0
            confident = bool(probs[ordered[0]] >= bundle.get("subtype_min_confidence", SUBTYPE_MIN_CONFIDENCE)
                             and margin >= bundle.get("subtype_min_margin", SUBTYPE_MIN_MARGIN))
            item["fraud_subtype_status"] = ("Tentative leading subtype" if confident
                                              else "Uncertain subtype: multiple categories plausible")
            item["subtype_confidence_heuristic_passed"] = confident
            item["subtype_probability_margin"] = margin
            item["subtype_uncertainty_note"] = ("This is a heuristic ranking, not a validated medical/security certainty. "
                                                "Do not treat the subtype as a confirmed fraud classification.")
        if explain:
            details = explanation_details(bundle, data.iloc[[i]], top=top)
            item["explanation"] = details
        result.append(item)
    return result[0] if len(result) == 1 else result


def explanation_details(bundle, raw, top=6):
    matrix, feats, _ = model_input(bundle, raw)
    values, bias = get_feature_importance_contributions(bundle, matrix)
    if len(values) != 1:
        raise ValueError("Supply one row for an explanation")
    values = values[0] * bundle["platt"][0]
    bias = float(np.asarray(bias).reshape(-1)[0])
    bias_cal = bias*bundle["platt"][0] + bundle["platt"][1]
    mapping = bundle["model_feature_names"]
    families = [family_for_encoded(name) for name in mapping]
    group_values = {}
    elements = {}
    for enc, fam, val in zip(mapping, families, values):
        group_values[fam] = group_values.get(fam, 0.0) + float(val)
        elements.setdefault(fam, []).append((enc, float(val)))
    ranked_pos = sorted(((g, v) for g, v in group_values.items() if v > 0), key=lambda z: -z[1])[:top]
    ranked_neg = sorted(((g, v) for g, v in group_values.items() if v < 0), key=lambda z: z[1])[:top]
    def present(group, contribution):
        important_members = sorted(elements[group], key=lambda r: abs(r[1]), reverse=True)[:3]
        sources = [c for c in FEATURE_GROUPS.get(group, []) if c in feats]
        observed = {c: (None if pd.isna(feats.iloc[0][c]) else
                        str(feats.iloc[0][c]) if c in CATEGORICAL else float(feats.iloc[0][c]))
                    for c in sources[:4]}
        return {"characteristic_group": group, "calibrated_logodds_effect": float(contribution),
                "encoded_evidence": [{"feature": k, "logodds_effect": v} for k, v in important_members],
                "observed_values": observed}
    prediction = expit(bias_cal + values.sum())
    evidence = []
    supported = bundle.get("supported_patterns")
    if isinstance(supported, pd.DataFrame) and not supported.empty:
        for row in supported.itertuples(index=False):
            current = str(feats.iloc[0][row.Feature_A]) + "|" + str(feats.iloc[0][row.Feature_B])
            if current == row.Observed_pattern:
                evidence.append({"features": str(row.Features), "observed_pattern": current,
                                 "train_support": int(row.Transactions),
                                 "training_frauds": int(row.Frauds),
                                 "conditional_lift": float(row.Conditional_lift)})
    # Warning: a negative SHAP group effect is relative to the model baseline, not
    # proof that the real transaction is safe. Interactions can distribute credit.
    return {
        "explanation_method": "XGBoost TreeSHAP log-odds" if bundle["model_name"] != "Sparse interactions"
                               else "Sparse linear log-odds coefficients",
        "calibrated_baseline_logodds": float(bias_cal),
        "reconstructed_fraud_probability": float(prediction),
        "risk_increasing_characteristics": [present(*z) for z in ranked_pos],
        "risk_decreasing_characteristics": [present(*z) for z in ranked_neg],
        "supported_train_only_interaction_evidence": evidence[:top],
        "limitations": "Model contributions, not causal proof. Correlated groups share attribution. Synthetic training labels.",
    }


def type_probabilities(model, X):
    """Return probabilities in LabelEncoder class order for flat or hierarchical models."""
    if model is None:
        return None
    if not isinstance(model, dict) or model.get("kind") != "hierarchical_subtype":
        probabilities = np.asarray(model.predict_proba(X), dtype=float)
        return probabilities / np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-12)
    family_p = np.asarray(model["family_model"].predict_proba(X), dtype=float)
    n = X.shape[0]
    out = np.zeros((n, len(model["class_names"])), dtype=float)
    for fid, family in enumerate(model["family_names"]):
        classes = model["family_class_indices"][family]
        if len(classes) == 1:
            out[:, classes[0]] = family_p[:, fid]
        else:
            sub_p = np.asarray(model["sub_models"][family].predict_proba(X), dtype=float)
            out[:, classes] = family_p[:, [fid]] * sub_p
    out = np.maximum(out, 1e-10)
    return out / out.sum(axis=1, keepdims=True)
