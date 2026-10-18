# Customer Churn Prediction Web Application — Assignment 2

Flask web app built on the Assignment 1 Telco Customer Churn notebook. It trains the same pipeline
(StandardScaler + OneHotEncoder, Logistic Regression and Random Forest with `class_weight="balanced"`,
80/20 stratified split, `random_state=42`), selects the model with the higher F1-score
(Logistic Regression: F1 0.607 vs Random Forest 0.538) and serves it through a web form.

## Quick start
```
pip install -r requirements.txt
python train_model.py      # optional: a trained model is already in model/
python app.py              # open http://127.0.0.1:5000
```

Pages: `/` (prediction form), `/privacy`, `/terms`, `/health`.
The form defaults to the sample customer from Assignment 1 (section 16).

## Layout
- `train_model.py` — recreates Assignment 1 and saves `model/churn_pipeline.joblib` and `model/metrics.json`
- `app.py` — Flask app with input validation
- `data/` — dataset CSV (`WA_Fn-UseC_-Telco-Customer-Churn.csv`)
- `docs/SETUP.md` — local and PythonAnywhere instructions
