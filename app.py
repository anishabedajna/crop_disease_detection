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

# --- 3. CUSTOM CSS (EXACT REPLICATION OF DARK BOX STYLE) ---
st.markdown("""
    <style>
    /* Vibrant Green Leaves Background */
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1542601906990-b4d3fb778b09?q=80&w=2026&auto=format&fit=crop");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }

    /* The Center Dark Box Overlay from your reference image */
    .main-box {
        background-color: rgba(0, 0, 0, 0.85);
        padding: 50px;
        border-radius: 12px;
        color: white;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }

    /* Title Styling: Bold and Underlined */
    .title-text {
        font-size: 46px;
        font-weight: bold;
        color: white;
        text-decoration: underline;
        margin-bottom: 20px;
    }

    /* Intro Phrasing */
    .intro-text {
        font-size: 18px;
        color: #dddddd;
        margin-bottom: 30px;
        line-height: 1.6;
        font-style: italic;
    }

    /* Prediction Button */
    .stButton>button {
        background-color: #28a745 !important;
        color: white !important;
        font-weight: bold !important;
        border: none !important;
        padding: 12px 20px !important;
        width: 100%;
        font-size: 18px !important;
    }

    /* Management Measures Card (Visible for diseases) */
    .mgmt-card {
        background-color: white;
        color: #1b5e20;
        text-align: left;
        padding: 25px;
        border-radius: 8px;
        margin-top: 30px;
        border-left: 10px solid #d32f2f;
    }

    /* Clean UI - Hide Streamlit elements */
    footer, header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- 4. MAIN UI CONTENT ---

# All UI elements are wrapped in the dark box div
st.markdown('<div class="main-box">', unsafe_allow_html=True)

st.markdown('<div class="title-text">Leaf Disease Detection</div>', unsafe_allow_html=True)

st.markdown("""
    <div class="intro-text">
        A machine learning approach for detecting crop diseases, upload a clear photo of 
        a plant leaf below to analyze its health and receive immediate management strategies.
    </div>
    """, unsafe_allow_html=True)

st.write("---")

model = load_trained_model()
try:
    with open('class_indices.json', 'r') as f:
        class_names = list(json.load(f).values())
except:
    class_names = []

# Drag and Drop uploader
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.markdown('<p style="font-size: 22px; font-weight: bold; margin-top:20px;">Original Image</p>', unsafe_allow_html=True)
    st.image(image, width=400)
    
    if st.button("Predict Leaf Disease"):
        with st.spinner("Analyzing..."):
            # Resize and prepare image for ML model
            img = image.resize((224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            preds = model.predict(img_array)
            result = class_names[np.argmax(preds)]
            
            # Show Disease Name
            st.markdown(f'<h2 style="color:white; margin-top:20px;">Predicted Disease: {result}</h2>', unsafe_allow_html=True)

            # --- CONDITIONAL MANAGEMENT LOGIC ---
            if "healthy" not in result.lower():
                st.markdown(f"""
                    <div class="mgmt-card">
                        <h3 style="color: #d32f2f; margin-top: 0;">🛡️ Management Measures</h3>
                        <p style="font-size: 16px;">Specific measures for <b>{result}</b>:</p>
                        <ul style="line-height: 1.8; font-size: 16px;">
                            <li><b>Pruning:</b> Remove and destroy infected leaves immediately.</li>
                            <li><b>Treatment:</b> Apply Neem oil or targeted organic fungicides.</li>
                            <li><b>Isolation:</b> Quarantine the plant to prevent airborne spore spread.</li>
                            <li><b>Hygiene:</b> Sterilize shears and tools after every use.</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.balloons()
                st.success(f"Result: {result}. No further measures needed.")

st.markdown('</div>', unsafe_allow_html=True)
