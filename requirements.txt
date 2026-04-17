import streamlit as st
import pickle
import numpy as np
from streamlit_lottie import st_lottie
import requests

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(page_title="ML Predictor", page_icon="🚀", layout="centered")

# ---------------------------
# Load Model
# ---------------------------
model = pickle.load(open("Modelli.pkl", "rb"))

# ---------------------------
# Lottie Animation Loader
# ---------------------------
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_ai = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_kyu7xb1v.json")

# ---------------------------
# Custom CSS for Attractive UI
# ---------------------------
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
    }
    h1 {
        color: #00C9A7;
        text-align: center;
    }
    .stButton>button {
        background-color: #00C9A7;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 18px;
    }
    .stNumberInput label {
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------
# Title Section
# ---------------------------
st.markdown("<h1>🚀 ML Prediction App</h1>", unsafe_allow_html=True)

st_lottie(lottie_ai, height=250)

st.write("### Enter Input Features")

# ---------------------------
# INPUTS (Adjust based on your model)
# ---------------------------
# ⚠️ IMPORTANT: Change number of inputs based on your model

feature1 = st.number_input("Feature 1")
feature2 = st.number_input("Feature 2")
feature3 = st.number_input("Feature 3")

# ---------------------------
# Prediction
# ---------------------------
if st.button("🔮 Predict"):
    try:
        features = np.array([[feature1, feature2, feature3]])
        prediction = model.predict(features)

        st.success(f"✅ Prediction: {prediction[0]}")

    except Exception as e:
        st.error(f"❌ Error: {e}")
