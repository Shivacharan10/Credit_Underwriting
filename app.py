import streamlit as st
import pickle
import numpy as np

# Load the saved model, scaler, and label encoder
with open('rf_model.pkl', 'rb') as model_file:
    rf_model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

with open('label_encoder.pkl', 'rb') as le_file:
    label_encoder = pickle.load(le_file)

# Streamlit app
st.title("Loan Approval Prediction")

st.write("Enter the applicant's details to predict whether the loan will be approved or rejected.")

# Input fields
no_of_dependents = st.number_input("Number of Dependents", min_value=0, step=1, value=0)
income_annum = st.number_input("Annual Income (₹)", min_value=0, step=1000, value=500000)
loan_amount = st.number_input("Loan Amount (₹)", min_value=0, step=1000, value=1000000)
loan_term = st.number_input("Loan Term (months)", min_value=1, step=1, value=12)
cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900, step=1, value=700)

# Predict button
if st.button("Predict Loan Status"):
    # Prepare input data
    input_data = np.array([[no_of_dependents, income_annum, loan_amount, loan_term, cibil_score]])
    input_data_scaled = scaler.transform(input_data)

    # Predict loan status
    prediction = rf_model.predict(input_data_scaled)
    loan_status = label_encoder.inverse_transform(prediction)

    # Display result
    st.write(f"The loan status is: **{loan_status[0]}**")
