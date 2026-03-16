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
    /* Background from reference */
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1501004318641-729e8e26bd05?q=80&w=2000&auto=format&fit=crop");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }

    /* Main Container: Dark Translucent Overlay as seen in Screenshot (42) */
    .main-container {
        background: rgba(0, 0, 0, 0.75);
        padding: 40px;
        border-radius: 15px;
        color: white;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Large Underlined Title */
    .title-text {
        font-size: 42px;
        font-weight: bold;
        color: white;
        margin-bottom: 10px;
        text-decoration: underline;
    }

    /* Intro Phrasing */
    .intro-text {
        font-size: 18px;
        color: #e0e0e0;
        margin-bottom: 30px;
        line-height: 1.6;
    }

    /* Predict Button Styling */
    .stButton>button {
        background-color: #28a745 !important;
        color: white !important;
        font-weight: bold !important;
        border: none !important;
        padding: 10px 24px !important;
        border-radius: 5px !important;
        width: 100%;
    }

    /* Management Card styling */
    .management-card {
        background: rgba(255, 255, 255, 0.95);
        color: #1b5e20;
        text-align: left;
        padding: 25px;
        border-radius: 10px;
        margin-top: 30px;
        border-left: 10px solid #d32f2f;
    }

    footer, header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- 4. MAIN UI CONTENT ---
st.markdown('<div class="main-container">', unsafe_allow_html=True)

st.markdown('<div class="title-text">Leaf Disease Detection</div>', unsafe_allow_html=True)

st.markdown("""
    <div class="intro-text">
        A machine learning approach for detecting crop diseases, upload a clear photo of 
        a plant leaf below to analyze its health and receive immediate management strategies.
    </div>
    """, unsafe_allow_html=True)

# Spacing
st.write("---")

model = load_trained_model()
try:
    with open('class_indices.json', 'r') as f:
        class_names = list(json.load(f).values())
except:
    class_names = []

uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.markdown('<p style="font-size: 20px; font-weight: bold;">Original Image</p>', unsafe_allow_html=True)
    st.image(image, width=350)
    
    if st.button("Predict Leaf Disease"):
        with st.spinner("Analyzing..."):
            # Model Processing
            img = image.resize((224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            preds = model.predict(img_array)
            result = class_names[np.argmax(preds)]
            
            # Result Label
            st.markdown(f'<h2 style="color:white;">Predicted Disease: {result}</h2>', unsafe_allow_html=True)

            # --- CONDITIONAL MANAGEMENT ---
            # Measures only "pop up" if NOT healthy
            if "healthy" not in result.lower():
                st.markdown(f"""
                    <div class="management-card">
                        <h3 style="color: #d32f2f; margin-top: 0;">🛡️ Management Measures</h3>
                        <p>Measures for <b>{result}</b>:</p>
                        <ul style="line-height: 1.8;">
                            <li><b>Pruning:</b> Remove and destroy infected leaves.</li>
                            <li><b>Treatment:</b> Apply Neem oil or recommended fungicides.</li>
                            <li><b>Isolation:</b> Keep infected plants away from healthy ones.</li>
                            <li><b>Sanitation:</b> Clean tools after handling diseased foliage.</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.balloons()
                st.success("The leaf is healthy! No management measures needed.")

st.markdown('</div>', unsafe_allow_html=True)
