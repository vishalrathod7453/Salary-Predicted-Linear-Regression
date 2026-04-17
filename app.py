import streamlit as st
import joblib
import numpy as np
import pandas as pd
import time

# --- PAGE CONFIG ---
st.set_page_config(page_title="Linear Salary AI", page_icon="📈", layout="centered")

# --- CUSTOM CSS FOR ANIMATIONS ---
st.markdown("""
    <style>
    @keyframes pulse {
        0% { transform: scale(1); box-shadow: 0 0 0 0 rgba(76, 175, 80, 0.4); }
        70% { transform: scale(1.02); box-shadow: 0 0 0 10px rgba(76, 175, 80, 0); }
        100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(76, 175, 80, 0); }
    }
    .prediction-box {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        padding: 30px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
        animation: pulse 2s infinite;
        margin-top: 20px;
    }
    .stSlider > div [data-baseweb="slider"] {
        background: linear-gradient(to right, #4CAF50, #2E7D32);
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD LINEAR MODEL ---
@st.cache_resource
def load_linear_model():
    return joblib.load("Modelli.pkl")

model = load_linear_model()

# --- HEADER ---
st.title("📈 Salary Predictor (Linear)")
st.write("Using a **Linear Regression** model to estimate salary based on experience.")

# --- UI LAYOUT ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Experience Input")
    exp = st.number_input("Years of Experience", min_value=0.0, max_value=50.0, value=2.0, step=0.5)

with col2:
    st.info("The model calculates a direct linear relationship between time and earnings.")

# --- PREDICTION ---
if st.button("Generate Projection 🚀"):
    # Create input frame matching 'YearsExperience' feature name 
    features = pd.DataFrame([[exp]], columns=['YearsExperience'])
    
    with st.status("Calculating linear trend...", expanded=True) as status:
        time.sleep(0.8)
        prediction = model.predict(features)[0]
        status.update(label="Projection Ready!", state="complete", expanded=False)

    st.markdown(f"""
        <div class="prediction-box">
            <p style="color: #888; font-size: 1.2rem; margin-bottom: 5px;">Estimated Salary</p>
            <h1 style="color: #4CAF50; font-size: 3.5rem; margin: 0;">${prediction:,.2f}</h1>
            <p style="margin-top: 10px; font-style: italic; opacity: 0.7;">Based on Modelli.pkl analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.snow()

# --- SIDEBAR DETAILS ---
with st.sidebar:
    st.title("Project Details")
    st.write("**Model Type:** Linear Regression ")
    st.write("**Feature:** YearsExperience ")
    st.write("**Version:** 2026 Portfolio Edition")
    st.divider()
    st.caption("Deployment target: Streamlit Cloud")
