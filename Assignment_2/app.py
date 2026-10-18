import json
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model" / "churn_pipeline.joblib"
METRICS_PATH = ROOT / "model" / "metrics.json"
app = Flask(__name__)

FEATURES = ["gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService", "MultipleLines",
            "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
            "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
            "MonthlyCharges", "TotalCharges"]
INTERNET_ADDONS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
NUMBERS = {"tenure": ("Tenure", 0, 72), "MonthlyCharges": ("Monthly charges", 0, 10000),
           "TotalCharges": ("Total charges", 0, 1000000)}

YES_NO = ["Yes", "No"]
ADDON = ["No internet service", "No", "Yes"]
OPTIONS = {
    "gender": ["Female", "Male"],
    "SeniorCitizen": [("0", "No"), ("1", "Yes")],
    "Partner": YES_NO, "Dependents": YES_NO, "PhoneService": YES_NO,
    "MultipleLines": ["No phone service", "No", "Yes"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    **{k: ADDON for k in INTERNET_ADDONS},
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": YES_NO,
    "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
}

# Defaults are the sample customer from Assignment 1, section 16.
DEFAULTS = {
    "gender": "Male", "SeniorCitizen": "0", "Partner": "Yes", "Dependents": "No", "tenure": "12",
    "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "Yes", "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "Yes", "StreamingMovies": "Yes", "Contract": "Month-to-month",
    "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
    "MonthlyCharges": "85.50", "TotalCharges": "1026.00",
}


def load_model_info():
    try:
        return json.loads(METRICS_PATH.read_text())
    except (OSError, ValueError):
        return {}


def validate(values):
    """Return (errors, numeric values)."""
    errors, nums = [], {}
    for key, opts in OPTIONS.items():
        allowed = [o[0] if isinstance(o, tuple) else o for o in opts]
        if values[key] not in allowed:
            errors.append(f"Choose a valid value for {key}.")
    for key, (label, lo, hi) in NUMBERS.items():
        try:
            n = float(values[key])
        except ValueError:
            errors.append(f"Enter a valid number for {label.lower()}.")
            continue
        if not lo <= n <= hi:
            errors.append(f"{label} must be between {lo} and {hi}.")
        else:
            nums[key] = n
    if errors:
        return errors, nums

    if values["PhoneService"] == "No" and values["MultipleLines"] != "No phone service":
        errors.append("Multiple lines must be 'No phone service' when phone service is 'No'.")
    if values["PhoneService"] == "Yes" and values["MultipleLines"] == "No phone service":
        errors.append("Multiple lines cannot be 'No phone service' when phone service is 'Yes'.")
    has_internet = values["InternetService"] != "No"
    wrong = [k for k in INTERNET_ADDONS if (values[k] == "No internet service") == has_internet]
    if wrong:
        errors.append("Internet add-ons must be 'No internet service' when internet service is 'No', "
                      f"and cannot be when the customer has internet. Check: {', '.join(wrong)}.")
    if nums["TotalCharges"] < nums["MonthlyCharges"] and nums["tenure"] >= 1:
        errors.append("Total charges cannot be lower than monthly charges for a customer with 1+ months of tenure.")
    return errors, nums


@app.route("/", methods=["GET", "POST"])
def index():
    values, errors, result = dict(DEFAULTS), [], None
    if request.method == "POST":
        values = {k: request.form.get(k, "").strip() for k in FEATURES}
        errors, nums = validate(values)
        if not errors:
            if not MODEL_PATH.exists():
                errors.append("Model not ready. Run `python train_model.py` (see docs/SETUP.md) before predicting.")
            else:
                row = {**values, **nums, "SeniorCitizen": int(values["SeniorCitizen"])}
                try:
                    model = joblib.load(MODEL_PATH)
                    x = pd.DataFrame([row], columns=FEATURES)
                    churn_prob = float(model.predict_proba(x)[0][list(model.classes_).index(1)])
                    churn = int(model.predict(x)[0]) == 1
                    result = {"churn": churn, "label": "Churn likely" if churn else "No churn predicted",
                              "probability": round(churn_prob * 100, 1)}
                except Exception:
                    app.logger.exception("Prediction failed")
                    errors.append("Prediction failed. Re-run `python train_model.py` to rebuild the model "
                                  "with your installed scikit-learn version.")
    return render_template("index.html", values=values, options=OPTIONS, errors=errors, result=result,
                           info=load_model_info())


@app.route("/privacy")
def privacy():
    return render_template("privacy.html")


@app.route("/terms")
def terms():
    return render_template("terms.html")


@app.route("/health")
def health():
    return {"status": "ok", "model_ready": MODEL_PATH.exists()}


if __name__ == "__main__":
    app.run(debug=True)
