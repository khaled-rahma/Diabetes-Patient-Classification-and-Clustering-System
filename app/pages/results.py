import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Patient Results", layout="wide")

st.title("📊 Patient Analysis Results")

# =========================
# Check if patient data exists
# =========================
if "patient_data" not in st.session_state:
    st.warning("⚠️ Please enter patient data first.")
    st.stop()

patient = st.session_state["patient_data"].copy()

# =========================
# Load Models
# =========================
kmeans = joblib.load("models/kmeans_model.joblib")
cluster_scaler = joblib.load("models/cluster_scaler.joblib")

model_diabetes = joblib.load("models/best_model_Diabetes.joblib")
model_heart = joblib.load("models/best_model_HeartDisease.joblib")
model_stroke = joblib.load("models/best_model_Stroke.joblib")

# =========================
# Cluster Names
# =========================
CLUSTER_NAMES = {
    0: "Healthy / Low Risk",
    1: "Overweight / Prediabetic",
    2: "Young / High BMI",
    3: "Elderly / Hypertensive",
    4: "Diabetic / High Metabolic Risk"
}

# =========================
# Clustering
# =========================
X_cluster = cluster_scaler.transform(
    patient[["Age", "BMI", "HighBP", "HighChol", "GenHlth", "PhysHlth"]]
)

cluster_id = int(kmeans.predict(X_cluster)[0])
cluster_name = CLUSTER_NAMES.get(cluster_id, "Unknown Cluster")

patient["Cluster_KMeans"] = cluster_id

st.subheader("🧩 Patient Health Cluster")

col1, col2 = st.columns(2)
with col1:
    st.metric("Cluster ID", cluster_id)
with col2:
    st.metric("Cluster Name", cluster_name)

# =========================
# Supervised Predictions
# =========================
st.subheader("📈 Complication Risk Probabilities")

probs = {
    "Diabetes": model_diabetes.predict_proba(patient)[0][1],
    "Heart Disease": model_heart.predict_proba(patient)[0][1],
    "Stroke": model_stroke.predict_proba(patient)[0][1]
}

df_probs = pd.DataFrame.from_dict(
    probs, orient="index", columns=["Probability"]
)

st.bar_chart(df_probs)

# =========================
# Medical Recommendations
# =========================
st.subheader("🩺 Personalized Recommendations")

if probs["Diabetes"] > 0.6:
    st.error("🔴 High diabetes risk – strict glucose monitoring is recommended.")
elif probs["Diabetes"] > 0.4:
    st.warning("🟠 Moderate diabetes risk – lifestyle improvements advised.")
else:
    st.success("🟢 Diabetes risk is currently low.")

if probs["Heart Disease"] > 0.6:
    st.error("🔴 High cardiovascular risk – monitor blood pressure and cholesterol.")
elif probs["Heart Disease"] > 0.4:
    st.warning("🟠 Moderate heart disease risk – regular checkups advised.")
else:
    st.success("🟢 Heart disease risk is low.")

if probs["Stroke"] > 0.6:
    st.error("🔴 High stroke risk – physical activity and diet control are critical.")
elif probs["Stroke"] > 0.4:
    st.warning("🟠 Moderate stroke risk – increase physical activity.")
else:
    st.success("🟢 Stroke risk is low.")

# Cluster-based advice
st.subheader("📌 Cluster-Specific Insights")

if cluster_id == 0:
    st.info("✔ Maintain your healthy lifestyle to stay in the low-risk group.")
elif cluster_id == 1:
    st.info("✔ Weight management and diet control are strongly recommended.")
elif cluster_id == 2:
    st.info("✔ Focus on BMI reduction through physical activity.")
elif cluster_id == 3:
    st.info("✔ Blood pressure monitoring is essential.")
elif cluster_id == 4:
    st.info("✔ Intensive medical follow-up is highly recommended.")

# =========================
# Save Patient Report
# =========================
st.subheader("💾 Download Patient Report")

report = pd.concat(
    [patient.reset_index(drop=True), df_probs.T.reset_index(drop=True)],
    axis=1
)

csv = report.to_csv(index=False).encode("utf-8")

st.download_button(
    label="⬇️ Download Patient Report (CSV)",
    data=csv,
    file_name="patient_report.csv",
    mime="text/csv"
)
