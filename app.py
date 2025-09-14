import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Load Models
# -----------------------------
linear_model = joblib.load("linear_regression_model.pkl")
kmeans_model = joblib.load("kmeans_model.pkl")

# Scaler for clustering (important for KMeans consistency)
scaler = joblib.load("scaler.pkl")

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Energy Forecasting & Anomaly Detection", layout="centered")

st.title("🏢 AI-Driven HVAC Energy Forecasting & Anomaly Detection")

st.markdown("""
This dashboard provides:
1. **Energy Forecasting** using Linear Regression  
2. **Anomaly Detection** in energy consumption using K-Means Clustering
""")

# -----------------------------
# Energy Forecasting Section
# -----------------------------
st.header("🔮 Energy Forecasting (Supervised Learning)")

col1, col2 = st.columns(2)

with col1:
    building_area = st.number_input("Building Area (m²)", min_value=10.0, value=500.0)
    floor_height = st.number_input("Floor Height (m)", min_value=2.0, value=3.0)
    win_area = st.number_input("Window Area (m²)", min_value=1.0, value=50.0)
    wall_area = st.number_input("Opaque Wall Area (m²)", min_value=10.0, value=200.0)

with col2:
    win_u = st.number_input("Window U-Value (W/m²K)", min_value=0.1, value=1.2)
    roof_u = st.number_input("Roof U-Value (W/m²K)", min_value=0.1, value=0.3)
    wall_u = st.number_input("Wall U-Value (W/m²K)", min_value=0.1, value=0.4)

if st.button("⚡ Predict Energy Consumption"):
    X = np.array([[building_area, floor_height, win_area, wall_area, win_u, roof_u, wall_u]])
    prediction = linear_model.predict(X)[0]
    st.success(f"Predicted Energy Consumption: **{prediction:.2f} MWh**")

# -----------------------------
# Anomaly Detection Section
# -----------------------------
st.header("🚨 Anomaly Detection (Unsupervised Learning)")

cooling = st.number_input("Cooling Load (MWh)", min_value=0.0, value=100.0)
heating = st.number_input("Heating Load (MWh)", min_value=0.0, value=150.0)

if st.button("🔍 Detect Anomaly"):
    total_energy = cooling + heating
    X_cluster = np.array([[cooling, heating, total_energy]])
    X_scaled = scaler.transform(X_cluster)   # scale before clustering
    cluster = kmeans_model.predict(X_scaled)[0]

    st.info(f"Assigned Cluster: **{cluster}**")
    if cluster == 0:
        st.warning("⚠️ This consumption pattern looks **unusual** (possible anomaly).")
    else:
        st.success("✅ Normal operating condition detected.")

# -----------------------------
# Footer
# -----------------------------

