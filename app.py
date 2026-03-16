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
        with st.spinner("🚀 Initializing System..."):
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
        font-size: 50px;
        font-weight: bold;
        text-decoration: underline;
        margin-top: -60px;
        text-shadow: 2px 2px 10px rgba(0,0,0,0.8);
    }
    .intro-text {
        text-align: center;
        color: #f0f0f0;
        font-size: 19px;
        margin: 20px auto;
        max-width: 700px;
        line-height: 1.6;
        background: rgba(0,0,0,0.4);
        padding: 20px;
        border-radius: 12px;
    }
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .img-container {
        border: 8px solid white;
        border-radius: 4px;
        margin: 20px auto;
    }
    .management-popup {
        background: rgba(255, 255, 255, 0.98);
        color: #1b5e20;
        padding: 25px;
        border-radius: 15px;
        border-left: 10px solid #d32f2f;
        margin-top: 25px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }
    .diagnosis-text {
        color: white;
        font-size: 24px;
        font-weight: bold;
        text-align: center;
        background: rgba(0,0,0,0.6);
        padding: 12px;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 4. MAIN UI CONTENT ---

st.markdown('<div class="main-title">Crop Disease Detection System</div>', unsafe_allow_html=True)

st.markdown("""
    <div class="intro-text">
        A machine learning approach for detecting crop diseases, upload a clear photo of 
        a plant leaf below to analyze its health and receive immediate management strategies.
    </div>
    """, unsafe_allow_html=True)

st.write("##")

# Load logic
model = load_trained_model()
category_names = []
try:
    with open('class_indices.json', 'r') as f:
        indices_data = json.load(f)
    category_names = list(indices_data.values())
except Exception:
    category_names = []

uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.markdown('<div class="img-container">', unsafe_allow_html=True)
    st.image(image, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    if st.button("RUN ANALYSIS", use_container_width=True):
        if model is not None and len(category_names) > 0:
            with st.spinner("Analyzing..."):
                img_resized = image.resize((224, 224))
                img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
                img_array = np.expand_dims(img_array, axis=0) / 255.0
                predictions = model.predict(img_array)
                result_label = category_names[np.argmax(predictions)]
                
                st.markdown(f'<div class="diagnosis-text">Status: {result_label}</div>', unsafe_allow_html=True)

                if "healthy" not in result_label.lower():
                    st.markdown(f"""
                        <div class="management-popup">
                            <h2 style='margin-top: 0; color: #d32f2f;'>🛡️ Management Strategies</h2>
                            <hr>
                            <p><b>Detected Disease:</b> {result_label}</p>
                            <p><b>Recommended Actions:</b></p>
                            <ul>
                                <li><b>Pruning:</b> Remove infected leaves immediately to prevent spread.</li>
                                <li><b>Treatment:</b> Apply organic fungicides like Neem oil or sulfur sprays.</li>
                                <li><b>Sanitation:</b>
