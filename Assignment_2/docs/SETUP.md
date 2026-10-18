# Setup and deployment

## Model
`train_model.py` reproduces the Assignment 1 notebook on `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`:
cleaning (`TotalCharges` to numeric, drop missing and duplicate rows, drop `customerID`), 80/20 stratified split
(`random_state=42`), both models, selection by F1-score. The saved `model/churn_pipeline.joblib` was trained
with scikit-learn 1.8. If you use a different scikit-learn version and see "Prediction failed", re-run
`python train_model.py`.

## Run on Windows
1. Install Python 3.10+ and extract this project.
2. `py -m venv .venv`
3. `.venv\Scripts\activate`
4. `pip install -r requirements.txt`
5. `python train_model.py` (rebuilds the model; prints test metrics)
6. `python app.py`
7. Open `http://127.0.0.1:5000`

## PythonAnywhere
1. Upload the project (including `data/` and `model/`) via the Files tab or `git clone`.
2. In a Bash console: `mkvirtualenv churn --python=python3.10` then `pip install -r requirements.txt`.
3. Run `python train_model.py` once so the model matches PythonAnywhere's scikit-learn version.
4. Web tab: add a new web app (Manual configuration), set the virtualenv path and source directory.
5. Edit the WSGI file to contain:
   ```python
   import sys
   path = "/home/<username>/Customer_Churn_Assignment2"
   if path not in sys.path:
       sys.path.insert(0, path)
   from app import app as application
   ```
6. Reload and open `https://<username>.pythonanywhere.com`.

A custom domain is optional and must be purchased and configured by the owner.

## Before submission
- Replace legal-page contact details with the actual project contact.
- Capture real screenshots (a churn result and a no-churn result) for the report.
- The dataset and trained model are committed on purpose (the assignment requires the saved model). Never commit secrets or private data.
