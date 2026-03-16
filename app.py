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

st.set_page_config(page_title="Leaf Disease Detection", page_icon="🌿", layout="centered")

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

# --- 3. CUSTOM CSS (STRICT STYLE REPLICATION) ---
st.markdown("""
    <style>
    /* Background: Vibrant Green Leaves from reference style */
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1542601906990-b4d3fb778b09?q=80&w=2026&auto=format&fit=crop");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }

    /* Main Container: Dark Translucent Overlay */
    .main-box {
        background: rgba(0, 0, 0, 0.8);
        padding: 40px;
        border-radius: 10px;
        color: white;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-top: -20px;
    }

    /* Title Styling: Big and Underlined */
    .title-text {
        font-size: 42px;
        font-weight: bold;
        color: white;
        text-decoration: underline;
        margin-bottom: 15px;
    }

    /* Intro Text Styling */
    .intro-text {
        font-size: 18px;
        color: #cccccc;
        margin-bottom: 25px;
        line-height: 1.5;
    }

    /* Predict Button Styling */
    .stButton>button {
        background-color: #28a745 !important;
        color: white !important;
        font-weight: bold !important;
        border: none !important;
        padding: 10px 20px !important;
        width: 100%;
    }

    /* Management Card styling (Only for Diseased) */
    .mgmt-box {
        background: white;
        color: #1b5e20;
        text-align: left;
        padding: 20px;
        border-radius: 8px;
        margin-top: 25px;
        border-left: 8px solid #d32f2f;
    }

    /* Hide Streamlit default UI elements */
    footer, header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- 4. MAIN UI CONTENT ---
st.markdown('<div class="main-box">', unsafe_allow_html=True)

# Title and Intro inside the dark box
st.markdown('<div class="title-text">Leaf Disease Detection</div>', unsafe_allow_html=True)
st.markdown("""
    <div class="intro-text">
        A machine learning approach for detecting crop diseases, upload a clear photo of 
        a plant leaf below to analyze its health and receive immediate management strategies.
    </div>
    """, unsafe_allow_html=True)

# Spacing Rule
st.write("---")

# Model Loading
model = load_trained_model()
try:
    with open('class_indices.json', 'r') as f:
        class_names = list(json.load(f).values())
except:
    class_names = []

# Uploader
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.markdown('<p style="font-size: 20px; font-weight: bold;">Original Image</p>', unsafe_allow_html=True)
    st.image(image, width=350)
    
    if st.button("Predict Leaf Disease"):
        with st.spinner("Analyzing..."):
            img = image.resize((224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            preds = model.predict(img_array)
            result = class_names[np.argmax(preds)]
            
            # Prediction Result
            st.markdown(f'<h2 style="color:white; margin-top:15px;">Predicted Disease: {result}</h2>', unsafe_allow_html=True)

            # --- MANAGEMENT CONDITION ---
            # Box only pops up if result is NOT healthy
            if "healthy" not in result.lower():
                st.markdown(f"""
                    <div class="mgmt-box">
                        <h3 style="color: #d32f2f; margin-top: 0;">🛡️ Management Measures</h3>
                        <p>Recommended actions for <b>{result}</b>:</p>
                        <ul style="line-height: 1.6;">
                            <li><b>Pruning:</b> Remove and safely destroy all infected foliage.</li>
                            <li><b>Treatment:</b> Apply Neem oil or organic fungicides early.</li>
                            <li><b>Isolation:</b> Separate the plant to prevent further contamination.</li>
                            <li><b>Sanitation:</b> Clean all gardening tools after contact with diseased leaves.</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.balloons()
                st.success("Result: Healthy! No management measures required.")

st.markdown('</div>', unsafe_allow_html=True)
