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
        font-style: italic;
    }
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .
