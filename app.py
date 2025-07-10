import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
with open("fraud_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("💳 Credit Card Fraud Detection App")
st.write("Detect fraudulent credit card transactions using a trained ML model.")

# Choose mode
mode = st.sidebar.radio("Choose input method", ["Manual Entry", "Upload CSV"])

# Manual Entry
if mode == "Manual Entry":
    st.subheader("📝 Enter Transaction Details")

    # Create input fields (Time + V1 to V28 + Amount = 30 total features)
    input_values = []
    labels = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

    for label in labels:
        val = st.number_input(label, value=0.0, format="%.6f")
        input_values.append(val)

    # Predict button
    if st.button("🔍 Predict"):
        input_df = pd.DataFrame([input_values], columns=labels)
        prediction = model.predict(input_df)[0]

        if prediction == 1:
            st.error("🚨 This transaction is predicted to be FRAUDULENT.")
        else:
            st.success("✅ This transaction is predicted to be GENUINE.")

# CSV Upload
else:
    st.subheader("📤 Upload CSV File")
    uploaded_file = st.file_uploader("Upload a CSV file with transaction data", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("📊 Uploaded Data Preview", data.head())

        try:
            predictions = model.predict(data)
            data["Prediction"] = predictions

            st.success(f"✅ Predictions completed: {sum(predictions==0)} Genuine, {sum(predictions==1)} Fraud")
            st.dataframe(data)

            csv = data.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Predicted CSV", data=csv, file_name="predicted_output.csv", mime="text/csv")

        except Exception as e:
            st.error(f"⚠️ Error during prediction: {str(e)}")
