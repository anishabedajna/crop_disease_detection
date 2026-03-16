import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import urllib.request
import json

# --- 1. CONFIG & MODEL SETUP ---
MODEL_URL = "https://github.com/anishabedajna/crop_disease_detection/releases/download/v1/plant_disease_model.h5"
MODEL_PATH = "plant_disease_model.h5"

st.set_page_config(page_title="Crop Disease Detection System", page_icon="🌿", layout="centered")

# --- 2. CACHED MODEL LOADING ---
@st.cache_resource
def load_trained_model():
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1000000:
        with st.spinner("🚀 Initializing AI Brain..."):
            try:
                opener = urllib.request.build_opener()
                opener.addheaders = [('User-agent', 'Mozilla/5.0')]
                urllib.request.install_opener(opener)
                urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            except Exception as e:
                st.error(f"Error loading model: {e}")
                return None
    return tf.keras.models.load_model(MODEL_PATH)

# --- 3. CUSTOM CSS ---
st.markdown("""
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1542601906990-b4d3fb778b09?q=80&w=2026&auto=format&fit=crop");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    .main-title {
        text-align: center;
        color: white;
        font-size: 48px;
        font-weight: bold;
        text-decoration: underline;
        margin-top: -80px;
        padding-bottom: 30px;
        text-shadow: 2px 2px 10px rgba(0,0,0,0.8);
    }
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .img-container {
        border: 8px solid white;
        border-radius: 4px;
        margin-bottom: 20px;
    }
    .management-popup {
        background: rgba(255, 255, 255, 0.95);
        color: #1b5e20;
        padding: 25px;
        border-radius: 15px;
        border-left: 10px solid #d32f2f;
        margin-top: 20px;
    }
    .diagnosis-text {
        color: white;
        font-size: 24px;
        font-weight: bold;
        text-align: center;
        background: rgba(0,0,0,0.5);
        padding: 10px;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 4. MAIN PAGE ---
st.markdown('<div class="main-title">Crop Disease Detection System</div>', unsafe_allow_html=True)

model = load_trained_model()
try:
    with open('class_indices.json', 'r') as f:
        class_indices = json.load(f)
    class_names = list(class_indices.values())
except:
    class_names = []

uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.markdown('<div class="img-container">', unsafe_allow_html=True)
    st.image(image, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    if st.button("RUN DETECTION"):
        with st.spinner("Analyzing..."):
            img = image.resize((224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0
            preds = model.predict(img_array)
            result = class_names[np.argmax(preds)]
            
            st.markdown(f'<div class="diagnosis-text">Status: {result}</div>', unsafe_allow_html=True)

            if "healthy" not in result.lower():
                st.markdown(f"""
                    <div class="management-popup">
                        <h2 style='margin-top: 0; color: #d32f2f;'>🛡️ Management Required</h2>
                        <hr>
                        <p><b>Issue:</b> {result}</p>
                        <ul>
                            <li><b>Action:</b> Remove infected leaves immediately.</li>
                            <li><b>Treatment:</b> Apply organic fungicide or Neem oil.</li>
                            <li><b>Sanitation:</b> Sterilize tools after use.</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.success("Crop is healthy!")
