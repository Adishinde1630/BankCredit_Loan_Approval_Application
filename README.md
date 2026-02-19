# Credit Risk Streamlit App

This repository contains a Streamlit app that exposes exploratory data analysis and model prediction functionality for the credit risk dataset.

Files added:

- `app.py` — Streamlit application with three top-level buttons: EDA, Dataset, Predictions. Under EDA there are Univariate, Bivariate and Correlation views.
- `requirements.txt` — Python dependencies for the app.

How to run:

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the Streamlit app from the project root:

```bash
streamlit run app.py
```

Notes:

- Place `credit_risk_dataset.csv` in the same folder (already present).
- The app will try to load a trained model from `outputs/best_model.joblib` and a list of features from `outputs/selected_features.txt` for predictions. If these files are missing, the Predictions page will show an error.
- EDA Correlation will use `outputs/correlation_matrix.csv` if present; otherwise it computes correlation from the dataset.

