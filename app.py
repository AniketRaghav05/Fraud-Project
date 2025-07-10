import streamlit as st
import pandas as pd
import pickle

# Set Streamlit title
st.title("💳 Credit Card Fraud Detection App")

# Load trained model
with open("fraud_model.pkl", "rb") as f:
    model = pickle.load(f)

# Upload CSV file
uploaded_file = st.file_uploader("📤 Upload transaction CSV file", type=["csv"])

if uploaded_file is not None:
    # Read uploaded CSV
    data = pd.read_csv(uploaded_file)
    
    # Show preview
    st.subheader("📊 Uploaded Data Preview")
    st.write(data.head())

    # Predict on uploaded data
    predictions = model.predict(data)

    # Add prediction column
    data["Prediction"] = predictions

    # Show results
    st.subheader("🔍 Prediction Results")
    st.write(data)

    # Count frauds and genuine
    fraud_count = (data["Prediction"] == 1).sum()
    genuine_count = (data["Prediction"] == 0).sum()

    st.success(f"✅ Genuine Transactions: {genuine_count}")
    st.error(f"🚨 Fraudulent Transactions: {fraud_count}")

    # Download result as CSV
    csv_output = data.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Predicted CSV",
        data=csv_output,
        file_name="predicted_output.csv",
        mime="text/csv"
    )
