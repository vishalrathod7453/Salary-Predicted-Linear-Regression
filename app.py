import streamlit as st
import pickle
import numpy as np
from streamlit_lottie import st_lottie
import requests

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(page_title="ML Prediction App", page_icon="🚀", layout="centered")

# ---------------------- LOAD MODEL ----------------------
model = pickle.load(open("Modelli.pkl", "rb"))

# ---------------------- LOTTIE ANIMATION ----------------------
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_ai = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_kyu7xb1v.json")

# ---------------------- CUSTOM CSS ----------------------
st.markdown("""
    <style>
    .main {
        background: linear-gradient(to right, #667eea, #764ba2);
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 18px;
    }
    .stTextInput>div>div>input {
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------- TITLE ----------------------
st.title("🤖 ML Prediction App")
st.write("Enter input values to get prediction")

# ---------------------- ANIMATION ----------------------
st_lottie(lottie_ai, height=200)

# ---------------------- INPUT FIELDS ----------------------
# Change these based on your model features
feature1 = st.number_input("Feature 1", value=0.0)
feature2 = st.number_input("Feature 2", value=0.0)
feature3 = st.number_input("Feature 3", value=0.0)

# ---------------------- PREDICTION ----------------------
if st.button("🔮 Predict"):
    input_data = np.array([[feature1, feature2, feature3]])
    prediction = model.predict(input_data)

    st.success(f"✅ Prediction: {prediction[0]}")

# ---------------------- FOOTER ----------------------
st.markdown("---")
st.write("Made with ❤️ using Streamlit")
