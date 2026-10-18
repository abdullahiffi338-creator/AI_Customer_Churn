"""Rebuilds the Assignment 1 pipeline: trains Logistic Regression and Random Forest,
selects the model with the higher F1-score and saves it for the Flask app."""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--data", default=str(ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"))
path = Path(parser.parse_args().data)
if not path.exists():
    raise SystemExit(f"Dataset not found: {path}. Put the Telco Customer Churn CSV in data/ or pass --data.")

# Cleaning (Assignment 1, sections 7 and 9)
df = pd.read_csv(path)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna(subset=["TotalCharges"]).drop_duplicates()
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})
df = df.drop("customerID", axis=1)
X, y = df.drop("Churn", axis=1), df["Churn"]
categorical = X.select_dtypes(exclude="number").columns.tolist()
numerical = X.select_dtypes(include="number").columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
])
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"),
}

results, fitted = {}, {}
for name, clf in classifiers.items():
    fitted[name] = Pipeline([("preprocessor", preprocessor), ("classifier", clf)]).fit(X_train, y_train)
    pred = fitted[name].predict(X_test)
    results[name] = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
    }
    print(f"\n{name}")
    for metric, value in results[name].items():
        print(f"  {metric:<9}: {value:.4f}")
    print("  confusion matrix:", confusion_matrix(y_test, pred).tolist())

best = max(results, key=lambda n: results[n]["f1"])
out = ROOT / "model"
out.mkdir(exist_ok=True)
joblib.dump(fitted[best], out / "churn_pipeline.joblib")
(out / "metrics.json").write_text(json.dumps({"selected_model": best, "results": results}, indent=2))
print(f"\nSelected model (highest F1): {best}\nSaved to {out / 'churn_pipeline.joblib'}")
