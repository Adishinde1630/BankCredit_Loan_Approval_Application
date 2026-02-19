import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Any


DATA_PATH = "credit_risk_dataset.csv"
OUTPUTS_DIR = "outputs"


def load_dataset():
    # Load dataset from CSV
    if os.path.exists(DATA_PATH):
        try:
            return pd.read_csv(DATA_PATH)
        except Exception:
            return None
    return None


def _ensure_column_transformer_compat(obj: object):
    """Ensure ColumnTransformer/Pipeline objects have minimal attrs for compatibility."""
    try:
        from sklearn.compose import ColumnTransformer
    except Exception:
        ColumnTransformer = None

    # If it's a ColumnTransformer, set missing internals
    if ColumnTransformer is not None and isinstance(obj, ColumnTransformer):
        if not hasattr(obj, "_name_to_fitted_passthrough"):
            try:
                obj._name_to_fitted_passthrough = {}
            except Exception:
                pass
        return

    # If it's a pipeline-like object with steps
    if hasattr(obj, "steps"):
        try:
            for _name, step in getattr(obj, "steps"):
                _ensure_column_transformer_compat(step)
        except Exception:
            pass

    # Recurse into common containers and attributes
    if isinstance(obj, (list, tuple, set)):
        for item in obj:
            _ensure_column_transformer_compat(item)
    if hasattr(obj, "__dict__"):
        for v in vars(obj).values():
            _ensure_column_transformer_compat(v)


def load_model():
    """Load the trained model from outputs/best_model.joblib with compatibility fixes."""
    model_path = os.path.join(OUTPUTS_DIR, "best_model.joblib")
    if not os.path.exists(model_path):
        return None
    try:
        model = joblib.load(model_path)
        try:
            _ensure_column_transformer_compat(model)
        except Exception:
            pass
        return model
    except AttributeError as err:
        # Try adding compatibility shim for sklearn internals commonly renamed
        try:
            import sklearn.compose._column_transformer as _ct

            class _RemainderColsList(list):
                pass

            setattr(_ct, "_RemainderColsList", _RemainderColsList)
            model = joblib.load(model_path)
            try:
                _ensure_column_transformer_compat(model)
            except Exception:
                pass
            return model
        except Exception:
            raise err
    except Exception:
        return None


def load_selected_features():
    """Load selected features from outputs/selected_features.txt if present."""
    sf = os.path.join(OUTPUTS_DIR, "selected_features.txt")
    if os.path.exists(sf):
        try:
            with open(sf, "r", encoding="utf-8") as f:
                contents = f.read().strip()
                if not contents:
                    return None
                # If comma separated
                if "," in contents:
                    return [c.strip() for c in contents.split(",") if c.strip()]
                # otherwise line separated
                return [c.strip() for c in contents.splitlines() if c.strip()]
        except Exception:
            return None
    return None


def show_dataset(df):
    st.subheader("Dataset")
    if df is None:
        st.error("Dataset file not found: credit_risk_dataset.csv")
        return
    st.markdown(f"**Rows:** {df.shape[0]}  —  **Columns:** {df.shape[1]}")
    # Show full dataset in an interactive dataframe
    st.dataframe(df)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv, file_name="credit_risk_dataset.csv")


def show_univariate(df):
    st.subheader("Univariate Analysis")
    if df is None:
        st.error("Dataset not loaded.")
        return

    # The EDA notebook includes these specific univariate plots:
    # - Histogram + boxplot for `loan_amnt`
    # - Value counts / bar plots for `loan_status`, `loan_grade`, `loan_intent`
    # - Boxplot of `person_income` grouped by `person_home_ownership`

    options = [
        "loan_amnt: histogram & boxplot",
        "loan_status: counts",
        "loan_grade: counts",
        "loan_intent: counts",
        "person_income by person_home_ownership: boxplot",
    ]

    chosen = st.multiselect("Choose plots (matching EDA)", options, default=[options[0]])

    # loan_amnt histogram + boxplot
    if "loan_amnt: histogram & boxplot" in chosen:
        if 'loan_amnt' in df.columns:
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            sns.histplot(df['loan_amnt'].dropna(), bins=30, ax=ax[0], kde=True)
            ax[0].set_title('Histogram of loan_amnt')
            sns.boxplot(x=df['loan_amnt'], ax=ax[1])
            ax[1].set_title('Boxplot of loan_amnt')
            st.pyplot(fig)
        else:
            st.warning('Column `loan_amnt` not found in dataset')

    # categorical counts (as in notebook)
    if "loan_status: counts" in chosen:
        if 'loan_status' in df.columns:
            st.markdown('**loan_status counts**')
            vc = df['loan_status'].value_counts()
            fig, ax = plt.subplots(figsize=(6, 3))
            sns.barplot(x=vc.index.astype(str), y=vc.values, ax=ax)
            ax.set_ylabel('count')
            ax.set_xlabel('loan_status')
            st.pyplot(fig)
            st.dataframe(vc)
        else:
            st.warning('Column `loan_status` not found in dataset')

    if "loan_grade: counts" in chosen:
        if 'loan_grade' in df.columns:
            st.markdown('**loan_grade counts**')
            vc = df['loan_grade'].value_counts()
            fig, ax = plt.subplots(figsize=(6, 3))
            sns.barplot(x=vc.index.astype(str), y=vc.values, ax=ax)
            ax.set_ylabel('count')
            ax.set_xlabel('loan_grade')
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            st.pyplot(fig)
            st.dataframe(vc)
        else:
            st.warning('Column `loan_grade` not found in dataset')

    if "loan_intent: counts" in chosen:
        if 'loan_intent' in df.columns:
            st.markdown('**loan_intent counts**')
            vc = df['loan_intent'].value_counts()
            fig, ax = plt.subplots(figsize=(6, 3))
            sns.barplot(x=vc.index.astype(str), y=vc.values, ax=ax)
            ax.set_ylabel('count')
            ax.set_xlabel('loan_intent')
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            st.pyplot(fig)
            st.dataframe(vc)
        else:
            st.warning('Column `loan_intent` not found in dataset')

    # boxplot of person_income grouped by home ownership (matches notebook grouping)
    if "person_income by person_home_ownership: boxplot" in chosen:
        if 'person_home_ownership' in df.columns and 'person_income' in df.columns:
            grouped = df.groupby('person_home_ownership')['person_income'].apply(list)
            # create a boxplot using matplotlib to mirror the notebook approach
            fig, ax = plt.subplots(figsize=(8, 4))
            try:
                ax.boxplot(grouped.values, labels=grouped.index)
                ax.set_title('person_income by person_home_ownership')
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
                st.pyplot(fig)
            except Exception as e:
                st.error(f'Error drawing grouped boxplot: {e}')
        else:
            st.warning('Required columns `person_home_ownership` or `person_income` not found')


def show_bivariate(df):
    st.subheader("Bivariate Analysis (Notebook plots)")
    if df is None:
        st.error("Dataset not loaded.")
        return

    choices = [
        "person_home_ownership vs loan_status",
        "loan_intent vs loan_status",
        "loan_grade vs loan_status",
    ]

    choice = st.selectbox("Choose plot", choices)

    fig, ax = plt.subplots(figsize=(8, 5))
    try:
        if choice == "person_home_ownership vs loan_status":
            xcol = 'person_home_ownership'
            sns.countplot(x=xcol, hue='loan_status', data=df, ax=ax)
            ax.set_title('person_home_ownership vs loan_status')
            for p in ax.patches:
                height = int(p.get_height())
                ax.text(p.get_x() + p.get_width() / 2., height + 1, height, ha="center", va="bottom", fontsize=9)
            if 'loan_status' in df.columns:
                counts = pd.crosstab(df[xcol], df['loan_status'])
                st.markdown("**Counts:**")
                st.dataframe(counts)

        elif choice == "loan_intent vs loan_status":
            xcol = 'loan_intent'
            sns.countplot(x=xcol, hue='loan_status', data=df, ax=ax)
            ax.set_title('loan_intent vs loan_status')
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            for p in ax.patches:
                height = int(p.get_height())
                ax.text(p.get_x() + p.get_width() / 2., height + 1, height, ha="center", va="bottom", fontsize=9)
            if 'loan_status' in df.columns:
                counts = pd.crosstab(df[xcol], df['loan_status'])
                st.markdown("**Counts:**")
                st.dataframe(counts)

        elif choice == "loan_grade vs loan_status":
            xcol = 'loan_grade'
            sns.countplot(x=xcol, hue='loan_status', data=df, ax=ax)
            ax.set_title('loan_grade vs loan_status')
            for p in ax.patches:
                height = int(p.get_height())
                ax.text(p.get_x() + p.get_width() / 2., height + 1, height, ha="center", va="bottom", fontsize=9)
            if 'loan_status' in df.columns:
                counts = pd.crosstab(df[xcol], df['loan_status'])
                st.markdown("**Counts:**")
                st.dataframe(counts)

        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error drawing plot: {e}")

def show_correlation(df):
    st.subheader("Correlation (Notebook heatmap)")
    if df is None:
        st.error("Dataset not loaded.")
        return
    corr_path = os.path.join(OUTPUTS_DIR, "correlation_matrix.csv")
    if os.path.exists(corr_path):
        corr = pd.read_csv(corr_path, index_col=0)
    else:
        # Use numeric_only to match notebook
        corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    # Use the same styling as the notebook (annotated, linewidth=8, center=0)
    sns.heatmap(corr, annot=True, linewidths=8, center=0, cmap="RdBu_r", ax=ax)
    ax.set_title("Correlation matrix")
    st.pyplot(fig)


def show_predictions_page(model, selected_features, df):
    st.subheader("Predictions")
    if model is None:
        st.error("Trained model not found in outputs/best_model.joblib")
        return

    fi_path = os.path.join(OUTPUTS_DIR, "feature_importances.csv")
    if os.path.exists(fi_path):
        fi = pd.read_csv(fi_path)
        st.table(fi.head(20))

    categories_map = {}
    numeric_stats = {}

    if df is not None:
        for feat in (selected_features or []):
            try:
                if feat in df.columns and not pd.api.types.is_numeric_dtype(df[feat].dtype):
                    vals = df[feat].dropna().unique().astype(str).tolist()
                    categories_map[feat] = sorted(list(dict.fromkeys(vals)))
                elif feat in df.columns:
                    col_series = df[feat].dropna()
                    if not col_series.empty:
                        numeric_stats[feat] = {
                            "min": float(col_series.min()),
                            "max": float(col_series.max()),
                            "mean": float(col_series.mean()),
                            "median": float(col_series.median())
                        }
            except Exception:
                pass

    st.markdown("**Single input prediction**")

    if selected_features is None:
        if hasattr(model, "feature_names_in_"):
            selected_features = list(model.feature_names_in_)
        else:
            st.error("Selected features list not found.")
            return

    with st.form("single_predict"):
        inputs = {}
        cols = st.columns(2)

        for i, feat in enumerate(selected_features):
            col = cols[i % 2]

            if feat in categories_map:
                val = col.selectbox(feat, [""] + categories_map[feat])
            elif feat in numeric_stats:
                default = numeric_stats[feat]["mean"]
                val = col.number_input(feat, value=float(default))
            else:
                val = col.text_input(feat)

            inputs[feat] = val

        submitted = st.form_submit_button("Predict")

        if submitted:
            try:
                # ======================================
                # BUILD INPUT ROW
                # ======================================
                input_row = {}
                for f in selected_features:
                    v = inputs.get(f, "")
                    input_row[f] = np.nan if v == "" else v

                input_df = pd.DataFrame([input_row], columns=selected_features)

                # ======================================
                # 🔥 FIX: CREATE loan_percent_income
                # ======================================
                if hasattr(model, "feature_names_in_"):

                    model_features = list(model.feature_names_in_)

                    if "loan_percent_income" in model_features:

                        if "loan_percent_income" not in input_df.columns:

                            if "loan_amnt" in input_df.columns and "person_income" in input_df.columns:
                                try:
                                    loan_amnt = float(input_df["loan_amnt"].iloc[0])
                                    income = float(input_df["person_income"].iloc[0])

                                    if income > 0:
                                        input_df["loan_percent_income"] = loan_amnt / income
                                    else:
                                        input_df["loan_percent_income"] = 0
                                except:
                                    input_df["loan_percent_income"] = 0

                # reorder columns exactly like training
                if hasattr(model, "feature_names_in_"):
                    input_df = input_df.reindex(columns=model.feature_names_in_)

                # ======================================
                # PREDICT (robust handling for class labels / probabilities)
                # ======================================
                pred = model.predict(input_df)
                pred0 = pred[0]

                decision_text = None
                prob_text = None

                # If model provides probabilities, compute confidence
                if hasattr(model, "predict_proba"):
                    try:
                        probs = model.predict_proba(input_df)
                        classes = list(getattr(model, "classes_", []))
                        prob_of_1 = None
                        if 1 in classes:
                            idx1 = classes.index(1)
                            prob_of_1 = float(probs[0][idx1])
                        else:
                            # try string '1' class
                            str_classes = [str(c) for c in classes]
                            if '1' in str_classes:
                                idx1 = str_classes.index('1')
                                prob_of_1 = float(probs[0][idx1])

                        # confidence for predicted class
                        try:
                            pred_idx = int(np.argmax(probs, axis=1)[0])
                            prob_of_pred = float(probs[0][pred_idx])
                            prob_text = f" (confidence {prob_of_pred*100:.1f}%)"
                        except Exception:
                            prob_text = None
                    except Exception:
                        prob_text = None

                # Map predicted value to human recommendation
                try:
                    if isinstance(pred0, (np.integer, int)):
                        decision_text = "❌ Recommendation: Deny loan" if int(pred0) == 1 else "✅ Recommendation: Approve loan"
                    else:
                        sval = str(pred0).strip().lower()
                        if any(k in sval for k in ["deny", "denied", "reject"]):
                            decision_text = "❌ Recommendation: Deny loan"
                        elif any(k in sval for k in ["approve", "approved", "accept"]):
                            decision_text = "✅ Recommendation: Approve loan"
                        else:
                            # fallback: if we computed prob_of_1, use threshold
                            if 'prob_of_1' in locals() and prob_of_1 is not None:
                                decision_text = "❌ Recommendation: Deny loan" if prob_of_1 >= 0.5 else "✅ Recommendation: Approve loan"
                            elif prob_text is not None and hasattr(model, "classes_"):
                                # pick highest-probability class
                                try:
                                    probs = model.predict_proba(input_df)
                                    pred_idx = int(np.argmax(probs, axis=1)[0])
                                    cls = model.classes_[pred_idx]
                                    decision_text = "❌ Recommendation: Deny loan" if int(cls) == 1 else "✅ Recommendation: Approve loan"
                                except Exception:
                                    decision_text = f"Prediction: {pred0}"
                            else:
                                decision_text = f"Prediction: {pred0}"
                except Exception:
                    decision_text = f"Prediction: {pred0}"

                st.markdown("**Recommendation & Explanation**")
                if decision_text is not None:
                    if decision_text.startswith("❌"):
                        st.error(decision_text + (prob_text or ""))
                    elif decision_text.startswith("✅"):
                        st.success(decision_text + (prob_text or ""))
                    else:
                        st.info(decision_text + (prob_text or ""))
                else:
                    st.info(f"Predicted: {pred0}{prob_text or ''}")

            except Exception as e:
                st.error(f"Error running prediction: {e}")

# ==============================
# HOME PAGE FUNCTION
# ==============================
def show_home_page(df):
    st.markdown("# 🏠 Credit Risk Analysis Dashboard")

    st.markdown("## 📋 Project Overview")
    st.write("""
This application predicts whether a loan should be Approved or Denied
using Machine Learning models trained on credit risk data.
""")

    st.markdown("## 📊 Dataset Overview")

    if df is not None:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Records", len(df))

        with col2:
            st.metric("Total Features", len(df.columns))

        with col3:
            if "loan_status" in df.columns:
                approval_rate = (
                    df["loan_status"].value_counts(normalize=True).get(0, 0) * 100
                )
                st.metric("Approval Rate", f"{approval_rate:.1f}%")

        st.dataframe(df.head(10))
    else:
        st.warning("Dataset not loaded")
    st.markdown("### By: Gorakh Shinde")



def main():

    # Load resources
    df = load_dataset()
    model = load_model()
    selected_features = load_selected_features()

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home","Dataset", "EDA", "Predictions"], index=0)

    if page == "Home":
        st.header("Welcome to the Credit Risk Analysis Dashboard")
        show_home_page(df)
       

    # EDA sub-navigation in sidebar
    eda_mode = None
    if page == "EDA":
        st.sidebar.markdown("**Exploratory Data Analysis**")
        eda_mode = st.sidebar.radio("View", ["Univariate", "Bivariate", "Correlation"], index=0)

    # Show selected page
    if page == "Dataset":
        st.header("Dataset")
        show_dataset(df)

    elif page == "EDA":
        st.header("Exploratory Data Analysis")
        if eda_mode == "Univariate":
            show_univariate(df)
        elif eda_mode == "Bivariate":
            show_bivariate(df)
        elif eda_mode == "Correlation":
            show_correlation(df)

    elif page == "Predictions":
        st.header("Predictions")
        show_predictions_page(model, selected_features, df)


if __name__ == "__main__":
    main()