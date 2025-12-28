
import streamlit as st

st.set_page_config(
    page_title="Diabetes Patient Stratification",
    layout="wide"
)

st.title("🩺 Diabetes Patient Stratification System")

st.markdown("""
### 🎯 Project Objective
This system aims to **analyze and stratify diabetes patients** using:
- 🔹 **Unsupervised Learning (K-Means Clustering)**
- 🔹 **Supervised Learning (Classification Models)**

### 🧠 What does the system provide?
- Patient risk group identification  
- Prediction of potential complications  
- Personalized health recommendations  
- Decision support for healthcare analysis  

### 📊 Dataset Used
- **diabetes_012 (BRFSS 2015)**

### 🚀 Get Started
Use the sidebar to navigate through the application.
""")

st.success("👈 Select a page from the sidebar to begin")
