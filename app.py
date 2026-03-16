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

# Using centered layout to keep everything focused
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

# --- 3. CUSTOM CSS FOR LAYOUT ---
st.markdown("""
    <style>
    /* Background Image */
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1542601906990-b4d3fb778b09?q=80&w=2026&auto=format&fit=crop");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }

    /* Title: Upper, Centered, Underlined */
    .main-title {
        text-align: center;
        color: white;
        font-size: 48px;
        font-weight: bold;
        text-decoration: underline;
        margin-top: -80px; /* Moves it significantly up */
        padding-bottom: 30px;
        text-shadow: 2px 2px 10px rgba(0,0,0,0.8);
    }

    /* Remove the 'A Machine Learning...' tag and Streamlit branding */
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* White Image Border */
    .img-container {
        border: 8px solid white;
        border-radius: 4px;
        margin-bottom: 20px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
    }

    /* Management Pop-up Box Styling */
    .management-popup {
        background: rgba(255, 255, 255, 0.95);
        color: #1b5e20;
        padding: 25px;
        border-radius: 15px;
        border-left: 10px solid #d32f2f; /* Red stripe for diseased alert */
        margin-top: 20px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.5);
    }
    
    .diagnosis-text {
        color
