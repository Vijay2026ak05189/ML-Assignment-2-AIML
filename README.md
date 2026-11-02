# Stroke Prediction Using Machine Learning

## Problem Statement

The objective of this project is to predict whether a patient is likely to suffer a stroke based on demographic and health-related information using Machine Learning classification algorithms.

---

## Dataset Description

Dataset Name: Stroke Prediction Dataset

Source:
https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset

Number of Records: 5110

Number of Attributes: 12

Target Variable:
stroke

Problem Type:
Binary Classification

---

## Models Used

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Naive Bayes Classifier
5. Support Vector Machine (SVM)

---

## Evaluation Metrics

- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

---

## Repository Structure

```text
README.md
Stroke_Prediction_Train.ipynb
app.py
dataset.csv
model_metrics.csv
requirements.txt
test_data.csv

model/
├── logistic_regression.pkl
├── decision_tree.pkl
├── knn.pkl
├── naive_bayes.pkl
├── svm.pkl
```

---

## Model Comparison Table

Refer model_metrics.csv

---

## Observations

| Model | Observation |
|---------|---------|
| Logistic Regression | Good baseline performance and interpretability. |
| Decision Tree | Easier to understand but may overfit. |
| KNN | Sensitive to nearest neighbors and scaling. |
| Naive Bayes | Fast and computationally efficient. |
| SVM | Effective in handling complex decision boundaries. |
| Overall Winner | Based on the highest Accuracy, AUC, F1 and MCC. |

---

## Streamlit Application Link

Add your Streamlit Cloud URL here after deployment.

---

## How to Run

### Run Training Notebook

Open and execute all cells:

```text
Stroke_Prediction_Train.ipynb
```

Run all cells sequentially.

### Run Streamlit App

```bash
streamlit run app.py
```

---

## Author

Vijayabhaskar Reddy
BITS Pilani WILP
