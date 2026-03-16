import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import urllib.request
import json

# --- 1. CONFIG ---
MODEL_URL = "https://github.com/anishabedajna/crop_disease_detection/releases/download/v1/plant_disease_model.h5"
MODEL_PATH = "plant_disease_model.h5"

st.set_page_config(page_title="Crop Disease Detection", layout="centered")

# --- 2. MODEL LOADING ---
@st.cache_resource
def load_trained_model():
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1000000:
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return tf.keras.models.load_model(MODEL_PATH)

# --- 3. STYLING (Simple & Clean) ---
st.markdown("""
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1542601906990-b4d3fb778b09?q=80&w=2026&auto=format&fit=crop");
        background-size: cover;
    }
    .title-text {
        text-align: center; color: white; font-size: 45px; 
        font-weight: bold; text-decoration: underline; margin-top: -50px;
    }
    .intro {
        text-align: center; color: white; background: rgba(0,0,0,0.5); 
        padding: 15px; border-radius: 10px; margin: 20px 0;
    }
    footer, header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- 4. UI CONTENT ---
st.markdown('<p class="title-text">Crop Disease Detection System</p>', unsafe_allow_html=True)

st.markdown('<div class="intro">A machine learning approach for detecting crop diseases, upload a clear photo of a plant leaf below to analyze its health and receive immediate management strategies.</div>', unsafe_allow_html=True)

# Spacers to push uploader down
st.write("##")
st.write("##")

# Load logic
model = load_trained_model()
try:
    with open('class_indices.json', 'r') as f:
        category_names = list(json.load(f).values())
except:
    category_names = []

# Uploader
file = st.file_uploader("", type=["jpg", "jpeg", "png"])

if file:
    img = Image.open(file)
    st.image(img, use_container_width=True)
    
    if st.button("RUN ANALYSIS", use_container_width=True):
        # Image Process
        img_res = img.resize((224, 224))
        arr = tf.keras.preprocessing.image.img_to_array(img_res) / 255.0
        arr = np.expand_dims(arr, axis=0)
        
        # Predict
        preds = model.predict(arr)
        label = category_names[np.argmax(preds)]
        
        st.subheader(f"Result: {label}")

        # Management Box (Standard Streamlit Container for safety)
        if "healthy" not in label.lower():
            with st.expander("🛡️ VIEW MANAGEMENT STRATEGIES", expanded=True):
                st.error(f"Issue Detected: {label}")
                st.write("**Recommended Actions:**")
                st.write("- **Pruning:** Remove infected leaves immediately.")
                st.write("- **Treatment:** Use organic fungicides like Neem oil.")
                st.write("- **Sanitation:** Sterilize tools after handling.")
                st.write("- **Airflow:** Increase spacing between plants.")
        else:
            st.balloons()
            st.success("This leaf is healthy!")
