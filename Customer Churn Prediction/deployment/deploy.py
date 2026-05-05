import streamlit as st
import pandas as pd
import tensorflow as tf
import pickle

preprocess = pickle.load(open('preprocess.pkl', 'rb'))
model = tf.keras.models.load_model('churn_model.h5')

st.header('TELCO ISP USER CHURN PREDICTION')

st.write('Please enter the information below:')

tenure = st.number_input('Customer tenure duration')
MonthlyCharges = st.number_input('Customer Monthly bill')
SeniorCitizen = st.selectbox('Senior citizenship status', ['no', 'yes'])
PaymentMethod = st.selectbox('Subcription payment method', ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'])
PaperlessBilling = st.selectbox('paperless billing', ['Yes', 'No'])
Contract = st.selectbox('Subscription contract terms', ['Month-to-month', 'One year', 'Two year'])
Partner = st.selectbox('Customer partner present', ['Yes', 'No'])
Dependents = st.selectbox('Customer dependents present', ['Yes', 'No'])
TechSupport = st.selectbox('Tech support service', ['Yes', 'No', 'No internet service'])
OnlineSecurity = st.selectbox('Online security service', ['Yes', 'No', 'No internet service'])
OnlineBackup = st.selectbox('Online backup service', ['Yes', 'No', 'No internet service'])
DeviceProtection = st.selectbox('Device protection service', ['Yes', 'No', 'No internet service'])



if st.button('submit'):
    
    num_cols = ['tenure', 'MonthlyCharges']
    cat_cols = ['SeniorCitizen','Partner', 'Dependents', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'Contract', 'PaperlessBilling', 'PaymentMethod']

    num_df = pd.DataFrame([[tenure, MonthlyCharges]], columns=num_cols)

    cat_df = pd.DataFrame([[SeniorCitizen, Partner, Dependents,
               OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
               Contract, PaperlessBilling,
               PaymentMethod]], columns=cat_cols)

    x = pd.concat([num_df, cat_df], axis=1)

    xs = pd.DataFrame(preprocess.transform(x))

    pred = model.predict(xs)

   
    if pred[0][0] < 0.5:
      st.write('Not churn')
    else:
      st.write('Churn')