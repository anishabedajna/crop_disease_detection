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

st.set_page_config(page_title="Crop Disease Detection", page_icon="🌿", layout="centered")

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

# --- 3. CUSTOM CSS (EXACT REPLICATION) ---
st.markdown("""
    <style>
    /* Background setup */
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1542601906990-b4d3fb778b09?q=80&w=2026&auto=format&fit=crop");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }

    /* Main Dark Box Container */
    .main-box {
        background-color: rgba(0, 0, 0, 0.82);
        padding: 40px;
        border-radius: 8px;
        color: white;
        text-align: center;
        margin-top: 30px;
    }

    /* Title Styling */
    .header-text {
        font-size: 38px;
        font-weight: 500;
        margin-bottom: 25px;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Predict Button Styling */
    div.stButton > button {
        background-color: #388e3c !important;
        color: white !important;
        border-radius: 4px !important;
        border: none !important;
        padding: 8px 20px !important;
        font-size: 16px !important;
        transition: 0.3s;
    }
    
    div.stButton > button:hover {
        background-color: #2e7d32 !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }

    /* Prediction Result Text */
    .prediction-output {
        font-size: 22px;
        font-weight: bold;
        margin-top: 25px;
    }

    /* Management Card - Visible for diseased results */
    .mgmt-card {
        background-color: #ffffff;
        color: #1b5e20;
        text-align: left;
        padding: 20px;
        border-radius: 6px;
        margin-top: 25px;
        border-left: 8px solid #c62828;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }

    /* Minimalist Uploader Hide labels */
    .stFileUploader label { display: none; }
    
    footer, header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- 4. MAIN UI CONTENT ---

st.markdown('<div class="main-box">', unsafe_allow_html=True)
st.markdown('<div class="header-text">Crop Disease Detection</div>', unsafe_allow_html=True)

# Model Loading
model = load_trained_model()
try:
    with open('class_indices.json', 'r') as f:
        class_names = list(json.load(f).values())
except:
    class_names = []

# Action Row
col_file, col_btn = st.columns([2, 1])

with col_file:
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

with col_btn:
    st.write("<br>", unsafe_allow_html=True) # Alignment spacing
    predict_clicked = st.button("Predict Leaf Disease")

# Output Section
if uploaded_file:
    image = Image.open(uploaded_file)
    st.markdown('<p style="font-size: 20px; margin-top:15px;">Original Image</p>', unsafe_allow_html=True)
    st.image(image, width=350)
    
    if predict_clicked:
        with st.spinner("Processing..."):
            # Image Preprocessing
            img = image.resize((224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Prediction
            preds = model.predict(img_array)
            result = class_names[np.argmax(preds)]
            
            # Display Prediction
            st.markdown(f'<div class="prediction-output">Predicted Disease: {result}</div>', unsafe_allow_html=True)

            # --- CONDITIONAL MANAGEMENT LOGIC ---
            if "healthy" not in result.lower():
                st.markdown(f"""
                    <div class="mgmt-card">
                        <h3 style="color: #c62828; margin-top: 0; font-size: 20px;">📋 Management Action Plan</h3>
                        <p style="font-size: 15px;">Steps to treat <b>{result}</b>:</p>
                        <ul style="line-height: 1.6; font-size: 14px;">
                            <li><b>Sanitation:</b> Prune and burn infected foliage immediately.</li>
                            <li><b>Treatment:</b> Apply appropriate organic fungicides (e.g., Copper or Sulfur).</li>
                            <li><b>Prevention:</b> Improve air circulation and avoid overhead watering.</li>
                            <li><b>Monitoring:</b> Inspect neighboring plants for early symptoms.</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.balloons()
                st.success("The crop is Healthy! No treatment required.")

st.markdown('</div>', unsafe_allow_html=True)
