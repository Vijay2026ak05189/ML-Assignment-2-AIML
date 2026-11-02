import streamlit as st, pandas as pd, joblib
from sklearn.metrics import *
st.set_page_config(page_title='Stroke Prediction')
st.title('Stroke Prediction ML Dashboard')
files={'Logistic Regression':'model/logistic_regression.pkl','Decision Tree':'model/decision_tree.pkl','KNN':'model/knn.pkl','Naive Bayes':'model/naive_bayes.pkl','SVM':'model/svm.pkl'}
model_name=st.selectbox('Select Model',list(files.keys()))
uploaded=st.file_uploader('Upload Test CSV',type=['csv'])
if uploaded:
 data=pd.read_csv(uploaded)
 X=data.drop('stroke',axis=1); y=data['stroke']
 model=joblib.load(files[model_name])
 pred=model.predict(X)
 try: auc=roc_auc_score(y,model.predict_proba(X)[:,1])
 except: auc=0
 st.write({'Accuracy':accuracy_score(y,pred),'Precision':precision_score(y,pred,average='weighted'),'Recall':recall_score(y,pred,average='weighted'),'F1':f1_score(y,pred,average='weighted'),'AUC':auc,'MCC':matthews_corrcoef(y,pred)})
 st.text(classification_report(y,pred))
 st.dataframe(pd.DataFrame(confusion_matrix(y,pred)))
