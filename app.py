import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load trained model
with open("fraud_model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Credit Card Fraud Detection", layout="centered")

st.title("💳 Credit Card Fraud Detection App")

# Sidebar for mode selection
option = st.sidebar.radio("Choose input type:", ["Manual Entry", "Upload CSV"])

# Manual Entry
if option == "Manual Entry":
    st.subheader("Enter Transaction Details")

    # List of features from the dataset
    features = [
        "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9",
        "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18",
        "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount"
    ]

    user_input = []
    for feature in features:
        val = st.number_input(f"{feature}", value=0.0, format="%.5f")
        user_input.append(val)

    if st.button("Predict"):
        x = np.array(user_input).reshape(1, -1)
        prediction = model.predict(x)[0]
        if prediction == 1:
            st.error("🔴 FRAUDULENT Transaction Detected!")
        else:
            st.success("🟢 GENUINE Transaction")

# CSV Upload
else:
    st.subheader("Upload CSV File")

    uploaded_file = st.file_uploader("Upload a CSV file with transaction(s)", type=["csv"])
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("📊 Uploaded Data Preview:", data)

        if st.button("Predict"):
            try:
                predictions = model.predict(data)
                data["Prediction"] = ["FRAUD" if p == 1 else "GENUINE" for p in predictions]
                st.write("🔍 Prediction Results:")
                st.dataframe(data)
            except Exception as e:
                st.error(f"Error in prediction: {e}")
                st.stop()
