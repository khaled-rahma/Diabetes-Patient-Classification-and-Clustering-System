import streamlit as st
import pandas as pd

st.title("🧾 Patient Data Input")

st.markdown("### Enter patient health information:")

example = {
    "Age": 55,
    "BMI": 29.5,
    "HighBP": 1,
    "HighChol": 1,
    "GenHlth": 3,
    "PhysHlth": 5
}

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", 18, 100, example["Age"])
    bmi = st.number_input("BMI", 10.0, 60.0, example["BMI"])

with col2:
    high_bp = st.selectbox("High Blood Pressure", [0, 1], index=example["HighBP"])
    high_chol = st.selectbox("High Cholesterol", [0, 1], index=example["HighChol"])

with col3:
    gen_hlth = st.slider("General Health (1 = Excellent → 5 = Poor)", 1, 5, example["GenHlth"])
    phys_hlth = st.slider("Physical Health Days (last 30 days)", 0, 30, example["PhysHlth"])

if st.button("➡️ Analyze Patient"):
    patient = pd.DataFrame([{
        "Age": age,
        "BMI": bmi,
        "HighBP": high_bp,
        "HighChol": high_chol,
        "GenHlth": gen_hlth,
        "PhysHlth": phys_hlth
    }])

    st.session_state["patient_data"] = patient
    st.success("✅ Patient data saved. Go to the Results page.")
