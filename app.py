import streamlit as st
import pickle
import numpy as np

# Load the trained fraud detection model
with open("fraud_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("💳 Fraud Detection System")
st.write("Enter transaction details to check if it's fraudulent:")

# All the input features
features = [
    "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9",
    "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18",
    "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount"
]

# Arrange inputs into 3 columns
user_inputs = []
cols = st.columns(3)
for i, feature in enumerate(features):
    with cols[i % 3]:
        value = st.number_input(f"{feature}", value=0.0)
        user_inputs.append(value)

# Predict on button click
if st.button("Predict Fraud"):
    input_array = np.array([user_inputs])
    prediction = model.predict(input_array)
    result = "🔴 FRAUD" if prediction[0] == 1 else "🟢 GENUINE"
    st.success(f"Prediction Result: {result}")
