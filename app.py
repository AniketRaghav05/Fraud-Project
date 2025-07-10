import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load trained model
with open("fraud_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("💳 Credit Card Fraud Detection")
st.write("This app predicts whether a transaction is **Fraudulent or Genuine** using AI.")

# Select input mode
mode = st.sidebar.radio("Select Input Mode", ["🧾 Upload CSV", "✍️ Manual Entry"])

# Required feature columns (same as training)
feature_columns = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

# 📂 Mode 1: CSV Upload
if mode == "🧾 Upload CSV":
    st.subheader("Upload a CSV file (30 columns: Time, V1-V28, Amount)")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file:
        try:
            data = pd.read_csv(uploaded_file)

            # Validate columns
            if list(data.columns) != feature_columns:
                st.error("❌ Column mismatch. Your CSV must have the same columns as the training data.")
            else:
                # Predict
                predictions = model.predict(data)
                data["Prediction"] = predictions
                st.success("✅ Prediction complete!")

                # Show results
                st.dataframe(data)

                frauds = sum(predictions == 1)
                genuines = sum(predictions == 0)

                st.warning(f"🔍 Genuine: {genuines}")
                st.error(f"🚨 Fraudulent: {frauds}")

                # Download result
                csv = data.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download Results", data=csv, file_name="predicted_output.csv", mime="text/csv")

        except Exception as e:
            st.error(f"Error: {e}")

# 🖊️ Mode 2: Manual Entry
else:
    st.subheader("Manually Enter Transaction Details")

    user_inputs = []
    for col in feature_columns:
        val = st.number_input(col, format="%.6f")
        user_inputs.append(val)

    if st.button("🔍 Predict Transaction"):
        input_df = pd.DataFrame([user_inputs], columns=feature_columns)
        prediction = model.predict(input_df)[0]

        if prediction == 1:
            st.error("🚨 This transaction is predicted to be **FRAUDULENT**.")
        else:
            st.success("✅ This transaction is predicted to be **GENUINE**.")
