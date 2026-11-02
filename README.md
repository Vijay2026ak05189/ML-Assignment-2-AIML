# Stroke Prediction Using Machine Learning

## Problem Statement

The objective of this assignment is to predict whether a patient is likely to suffer a stroke based on demographic and health-related information using Machine Learning classification algorithms.

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

---

## Github Repository link

https://github.com/Vijay2026ak05189/ML-Assignment-2-AIML/edit/main
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

## Model Comparison Table

Refer model_metrics.csv from Github repository or below 

| Model | Observation |

| Model	| Accuracy	AUC	Precision	Recall	F1	MCC
Logistic Regression	0.7456	0.8364	0.9448	0.7456	0.8175	0.2597
Decision Tree	0.9335	0.5477	0.9187	0.9335	0.9255	0.1218
KNN	0.9501	0.6028	0.9217	0.9501	0.9286	0.0715
Naive Bayes	0.8669	0.8018	0.9283	0.8669	0.8933	0.202
SVM	0.7613	0.776	0.9347	0.7613	0.8276	0.2023

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
