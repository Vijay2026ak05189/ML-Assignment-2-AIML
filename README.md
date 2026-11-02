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
<img width="540" height="225" alt="image" src="https://github.com/user-attachments/assets/ad6a0aeb-1f77-4bcf-8042-61f9dd6d8876" />


---

## Observations

| Model | Observation |
|---------|---------|
Logistic Regression	Achieved moderate Accuracy (74.56%) and the highest Precision (94.48%) among all models. It is effective at minimizing false positives but missed several positive cases, resulting in lower Recall and Accuracy.
Decision Tree	Delivered strong Accuracy (93.35%) and Recall (93.35%), indicating good classification performance. However, its AUC score (54.77%) was the lowest, suggesting limited ability to distinguish between classes across different thresholds.
KNN	Achieved the highest Accuracy (95.01%), Recall (95.01%), and F1 Score (92.86%). Although its AUC and MCC values were relatively low, it provided the best overall prediction performance on this dataset.
Naive Bayes	Produced balanced results with Accuracy (86.69%), AUC (80.18%), and F1 Score (89.33%). It demonstrated stable performance and good generalization but did not outperform KNN or Decision Tree.
Support Vector Machine (SVM)	Achieved moderate Accuracy (76.13%) and high Precision (93.47%). While effective at reducing false positives, its Recall and F1 Score were lower than those of KNN and Decision Tree.
Overall, Winner for your dataset?	K-Nearest Neighbors (KNN) is the best-performing model for this Stroke Prediction dataset, achieving the highest Accuracy (95.01%), Recall (95.01%), and F1 Score (92.86%). Therefore, KNN was selected as the final model for deployment in the Streamlit application.

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
