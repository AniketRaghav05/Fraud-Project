import streamlit as st
import numpy as np
import pickle
import pandas as pd

# Load model
with open("fraud_model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Fraud Detection", layout="wide")
st.title("💳 Credit Card Fraud Detection")
st.markdown("This app predicts whether a transaction is **Fraudulent (🔴)** or **Genuine (🟢)** using an AI model.")

# --- Test Samples ---
fraud_sample = [406, 1.1918, 0.2661, 0.1664, 0.4481, 0.0600, -0.0823, -0.0788, 0.0851, -0.2554,
                -0.1669, 1.6127, 1.0652, 0.4890, -0.1437, 0.6355, 0.4639, -0.1148, -0.1833, -0.1457,
                -0.0690, -0.2257, -0.6386, 0.1012, -0.3398, 0.1671, 0.1258, -0.0089, 0.0147, 212.0]

genuine_sample = [12345, -1.3598, -0.0727, 2.5363, 1.3781, -0.3383, 0.4623, 0.2395, 0.0986, 0.3637,
                  0.0907, -0.5515, -0.6178, -0.9913, -0.3111, 1.4681, -0.4704, 0.2079, 0.0257, 0.4039,
                  0.2514, -0.0183, 0.2778, -0.1104, 0.0669, 0.1285, -0.1891, 0.1335, -0.0210, 2.69]

# --- Input Mode Selection ---
st.write("### 🧪 Choose an Input Method")
test_case = st.radio("Select:", ["Manual Input", "Load Genuine Transaction", "Load Fraudulent Transaction"])

feature_names = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
inputs = []
cols = st.columns(3)

# --- Input Fields ---
for i, name in enumerate(feature_names):
    with cols[i % 3]:
        if test_case == "Load Genuine Transaction":
            val = st.number_input(name, value=genuine_sample[i], key=f"input_{i}")
        elif test_case == "Load Fraudulent Transaction":
            val = st.number_input(name, value=fraud_sample[i], key=f"input_{i}")
        else:
            val = st.number_input(name, value=0.0, key=f"input_{i}")
        inputs.append(val)

# --- Prediction ---
if st.button("🔍 Predict Fraud"):
    x = np.array([inputs])
    prediction = model.predict(x)[0]
    label = "🟢 GENUINE" if prediction == 0 else "🔴 FRAUDULENT"
    st.subheader(f"**Prediction Result: {label}**")

# --- CSV Upload Section ---
st.write("---")
st.write("### 📁 Upload a CSV File for Bulk Prediction")

csv_file = st.file_uploader("Upload CSV with same columns (Time, V1 to V28, Amount)", type=["csv"])
if csv_file is not None:
    df = pd.read_csv(csv_file)
    predictions = model.predict(df)
    df["Prediction"] = ["FRAUD" if p == 1 else "GENUINE" for p in predictions]
    st.dataframe(df)

    # Optional download
    csv = df.to_csv(index=False).encode()
    st.download_button("📥 Download Results as CSV", data=csv, file_name="predictions.csv", mime="text/csv")
