Stroke Prediction Using Machine Learning
Problem Statement
The objective of this project is to predict whether a patient is likely to suffer a stroke based on demographic and health-related information using Machine Learning classification algorithms.

Dataset Description
Dataset Name: Stroke Prediction Dataset

Source: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset

Number of Records: 5110

Number of Attributes: 12

Target Variable: stroke

Problem Type: Binary Classification

Features:

gender
age
hypertension
heart_disease
ever_married
work_type
Residence_type
avg_glucose_level
bmi
smoking_status
GitHub Repository Link
Add your repository URL here:

https://github.com/YOUR_USERNAME/YOUR_REPOSITORY

Models Used
Logistic Regression
Decision Tree Classifier
K-Nearest Neighbors (KNN)
Naive Bayes Classifier
Support Vector Machine (SVM)
Evaluation Metrics
The following metrics are used:

Accuracy
AUC Score
Precision
Recall
F1 Score
Matthews Correlation Coefficient (MCC)
Repository Structure
README.md
Stroke_Prediction_Train.ipynb
app.py
dataset.csv
requirements.txt
test_data.csv

model/
├── logistic_regression.pkl
├── decision_tree.pkl
├── knn.pkl
├── naive_bayes.pkl
├── svm.pkl
└── model_metrics.csv
Model Comparison Table
Model	Accuracy	AUC	Precision	Recall	F1	MCC
Logistic Regression						
Decision Tree						
KNN						
Naive Bayes						
SVM						
Observations
Model	Observation
Logistic Regression	Good baseline performance and interpretability.
Decision Tree	Easier to understand but may overfit.
KNN	Sensitive to nearest neighbors and scaling.
Naive Bayes	Fast and computationally efficient.
SVM	Effective in handling complex decision boundaries.
Overall Winner	Based on the highest Accuracy, AUC, F1 and MCC.
Streamlit Application Link
Add your Streamlit Cloud URL here after deployment:

https://your-app-name.streamlit.app

Run Training Notebook
Open and execute:

Stroke_Prediction_Train.ipynb
Run all cells sequentially to:

Load and preprocess the dataset
Train all five machine learning models
Generate evaluation metrics
Create test_data.csv
Generate model_metrics.csv
Save trained model files (.pkl)
Run Streamlit App
streamlit run app.py
