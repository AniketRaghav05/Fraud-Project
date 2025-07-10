import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load trained model
with open("fraud_model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Credit Card Fraud Detection", layout="centered")
st.title("💳 Credit Card Fraud Detection App")

st.markdown("Use this app to detect whether a credit card transaction is **fraudulent** or **genuine**.")

# List of input features (from creditcard.csv)
features = [
    "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9",
    "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18",
    "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount"
]

# Sample values (real examples from Kaggle dataset)
fraud_sample = [
    406, 1.19185711131486, 0.26615071205963, 0.16648011335321, 0.448154078460911,
    0.0600176492822243, -0.0823608088155687, -0.0788029833323113, 0.0851016549148105,
    -0.255425128109186, -0.166974414004614, 1.61272666105479, 1.06523531137273,
    0.48909501589608, -0.143772296441519, 0.635558093258208, 0.463917041022171,
    -0.114804663102346, -0.183361270123993, -0.145783041325259, -0.0690831352230203,
    -0.225775248033138, -0.638671952771851, 0.101288021253234, -0.339846475529127,
    0.167170404418143, 0.125894532368176, -0.00898309914322813, 0.0147241691924927,
    212.0
]

genuine_sample = [
    0, -1.3598071336738, -0.0727811733098497, 2.53634673796914, 1.37815522427443,
    -0.338320769942518, 0.462387777762292, 0.239598554061257, 0.0986979012610507,
    0.363786969611213, 0.0907941719789316, -0.551599533260813, -0.617800855762348,
    -0.991389847235408, -0.311169353699879, 1.46817697209427, -0.470400525259478,
    0.207971241929242, 0.0257905801985591, 0.403992960255733, 0.251412098239705,
    -0.018306777944153, 0.277837575558899, -0.110473910188767, 0.0669280749146731,
    0.128539358273528, -0.189114843888824, 0.133558376740387, -0.0210530534538215,
    2.69
]

# Sidebar mode selection
option = st.sidebar.radio("Choose Input Type", ["Manual Entry", "Upload CSV"])

if option == "Manual Entry":
    st.subheader("Enter Transaction Details or Load a Sample")

    # Add button to load sample values
    if st.button("🔴 Load Fraud Sample"):
        user_input = fraud_sample
    elif st.button("🟢 Load Genuine Sample"):
        user_input = genuine_sample
    else:
        user_input = [0.0] * len(features)

    # Create number inputs dynamically
    inputs = []
    for i, feature in enumerate(features):
        value = st.number_input(f"{feature}", value=user_input[i], format="%.5f")
        inputs.append(value)

    if st.button("🧠 Predict"):
        x = np.array(inputs).reshape(1, -1)
        prediction = model.predict(x)[0]
        if prediction == 1:
            st.error("🔴 FRAUDULENT Transaction Detected!")
        else:
            st.success("🟢 GENUINE Transaction")

else:
    st.subheader("Upload CSV File")

    uploaded_file = st.file_uploader("Upload a CSV file with transactions (same columns as training)", type=["csv"])
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("📊 Uploaded Data Preview:", data.head())

        if st.button("🧠 Predict for Uploaded File"):
            try:
                predictions = model.predict(data)
                data["Prediction"] = ["FRAUD" if p == 1 else "GENUINE" for p in predictions]
                st.write("🔍 Prediction Results:")
                st.dataframe(data)
            except Exception as e:
                st.error(f"❌ Error during prediction: {e}")
