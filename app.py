import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model
with open("fraud_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("💳 Credit Card Fraud Detection App")
st.write("Detect fraudulent transactions using AI (RandomForest + SMOTE)")

mode = st.sidebar.selectbox("Choose input mode", ["🔘 Manual Entry", "📂 Upload CSV"])

# --- Sample values for autofill ---
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
    149.62
]

# --- 1️⃣ Manual Mode ---
if mode == "🔘 Manual Entry":
    st.subheader("Manually Enter a Transaction")

    # Buttons for autofill
    if st.button("🔴 Fill Sample Fraud"):
        st.session_state.manual_input = fraud_sample

    if st.button("🟢 Fill Sample Genuine"):
        st.session_state.manual_input = genuine_sample

    # Prepare input fields
    if "manual_input" not in st.session_state:
        st.session_state.manual_input = [0.0] * 30

    labels = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    input_values = []

    for i, label in enumerate(labels):
        val = st.number_input(label, value=st.session_state.manual_input[i], format="%.6f", key=f"input_{i}")
        input_values.append(val)

    if st.button("🔍 Predict Transaction"):
        input_df = pd.DataFrame([input_values], columns=labels)
        prediction = model.predict(input_df)[0]
        if prediction == 1:
            st.error("🚨 Prediction: FRAUD")
        else:
            st.success("✅ Prediction: GENUINE")

# --- 2️⃣ CSV Upload Mode ---
elif mode == "📂 Upload CSV":
    st.subheader("Upload a CSV with Transactions")
    uploaded_file = st.file_uploader("Upload .csv", type="csv")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("📊 Uploaded Data Preview", data.head())

        data["Prediction"] = model.predict(data)

        st.write("🔎 Result:")
        st.success(f"Genuine: {(data['Prediction'] == 0).sum()}")
        st.error(f"Fraud: {(data['Prediction'] == 1).sum()}")

        st.dataframe(data)

        csv_output = data.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Results", csv_output, "predicted_output.csv", "text/csv")
