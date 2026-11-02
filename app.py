import streamlit as st
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# ----------------------------------
# Page Configuration
# ----------------------------------

st.set_page_config(
    page_title="Stroke Prediction Dashboard",
    page_icon="🩺",
    layout="wide"
)

st.title("Stroke Prediction - Machine Learning Dashboard")

st.markdown("""
This application compares the performance of five machine learning models:

- Logistic Regression
- Decision Tree
- KNN
- Naive Bayes
- Support Vector Machine (SVM)
""")

# ----------------------------------
# Model File Paths
# ----------------------------------

model_files = {
    "Logistic Regression":
        "model/logistic_regression.pkl",

    "Decision Tree":
        "model/decision_tree.pkl",

    "KNN":
        "model/knn.pkl",

    "Naive Bayes":
        "model/naive_bayes.pkl",

    "SVM":
        "model/svm.pkl"
}

# ----------------------------------
# Display Model Comparison
# ----------------------------------

st.subheader("Model Comparison Metrics")

try:
    metrics_df = pd.read_csv("model_metrics.csv")
    st.dataframe(metrics_df, use_container_width=True)

except Exception as e:
    st.warning(
        "model_metrics.csv not found."
    )

# ----------------------------------
# Model Selection
# ----------------------------------

st.subheader("Evaluate a Model")

selected_model = st.selectbox(
    "Choose Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "SVM"
    ]
)

# ----------------------------------
# Upload Test CSV
# ----------------------------------

uploaded_file = st.file_uploader(
    "Upload Test Dataset (CSV)",
    type=["csv"]
)

# ----------------------------------
# Prediction Section
# ----------------------------------

if uploaded_file is not None:

    try:

        data = pd.read_csv(uploaded_file)

        TARGET = "stroke"

        if TARGET not in data.columns:

            st.error(
                f"Target column '{TARGET}' not found in uploaded file."
            )

        else:

            X = data.drop(
                TARGET,
                axis=1
            )

            y = data[TARGET]

            model = joblib.load(
                model_files[selected_model]
            )

            predictions = model.predict(X)

            try:
                probabilities = model.predict_proba(X)[:, 1]

                auc = roc_auc_score(
                    y,
                    probabilities
                )

            except Exception:
                auc = 0

            accuracy = accuracy_score(
                y,
                predictions
            )

            precision = precision_score(
                y,
                predictions,
                average="weighted",
                zero_division=0
            )

            recall = recall_score(
                y,
                predictions,
                average="weighted",
                zero_division=0
            )

            f1 = f1_score(
                y,
                predictions,
                average="weighted",
                zero_division=0
            )

            mcc = matthews_corrcoef(
                y,
                predictions
            )

            st.subheader(
                f"Results - {selected_model}"
            )

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Accuracy",
                round(accuracy, 4)
            )

            col2.metric(
                "Precision",
                round(precision, 4)
            )

            col3.metric(
                "Recall",
                round(recall, 4)
            )

            col4, col5, col6 = st.columns(3)

            col4.metric(
                "F1 Score",
                round(f1, 4)
            )

            col5.metric(
                "AUC Score",
                round(auc, 4)
            )

            col6.metric(
                "MCC Score",
                round(mcc, 4)
            )

            # ------------------------
            # Classification Report
            # ------------------------

            st.subheader(
                "Classification Report"
            )

            report = classification_report(
                y,
                predictions,
                zero_division=0
            )

            st.text(report)

            # ------------------------
            # Confusion Matrix
            # ------------------------

            st.subheader(
                "Confusion Matrix"
            )

            cm = confusion_matrix(
                y,
                predictions
            )

            cm_df = pd.DataFrame(cm)

            st.dataframe(
                cm_df,
                use_container_width=True
            )

    except Exception as e:

        st.error(
            f"Error: {e}"
        )

# ----------------------------------
# Footer
# ----------------------------------

st.markdown("---")

st.markdown(
    "Machine Learning Assignment 2"
)
