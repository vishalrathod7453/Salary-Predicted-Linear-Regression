import streamlit as st
import pickle
import numpy as np
import requests
from streamlit_lottie import st_lottie

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="AI Prediction App",
    page_icon="🚀",
    layout="centered"
)

# ------------------ LOAD MODEL ------------------
model = pickle.load(open("Modelli.pkl", "rb"))

# ------------------ LOTTIE FUNCTION ------------------
def load_lottie(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Animation URL
lottie_animation = load_lottie("https://assets5.lottiefiles.com/packages/lf20_zrqthn6o.json")

# ------------------ CUSTOM CSS ------------------
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to right, #141E30, #243B55);
        color: white;
    }
    h1 {
        text-align: center;
        color: #00FFD1;
    }
    .stButton>button {
        background: linear-gradient(to right, #00FFD1, #00C9A7);
        color: black;
        font-size: 18px;
        border-radius: 12px;
        height: 3em;
        width: 100%;
    }
    .stNumberInput input {
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------ TITLE ------------------
st.title("🤖 AI Prediction App")
st.write("### Enter your data below")

# ------------------ ANIMATION ------------------
if lottie_animation:
    st_lottie(lottie_animation, height=200)

# ------------------ INPUT SECTION ------------------
st.subheader("📥 Input Features")

# 👉 CHANGE NUMBER OF FEATURES HERE if needed
num_features = 3  

inputs = []
cols = st.columns(num_features)

for i in range(num_features):
    with cols[i]:
        value = st.number_input(f"Feature {i+1}", value=0.0)
        inputs.append(value)

# ------------------ PREDICTION ------------------
st.markdown("### 🔍 Prediction")

if st.button("🚀 Predict"):
    try:
        input_array = np.array([inputs])
        prediction = model.predict(input_array)

        st.success(f"✅ Prediction: {prediction[0]}")

    except Exception as e:
        st.error(f"❌ Error: {e}")

# ------------------ FOOTER ------------------
st.markdown("---")
st.markdown("<center>Made with ❤️ using Streamlit</center>", unsafe_allow_html=True)
