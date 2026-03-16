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

st.set_page_config(page_title="Plant Disease Detection App", page_icon="🌿", layout="centered")

# --- 2. CACHED MODEL LOADING ---
@st.cache_resource
def load_trained_model():
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1000000:
        with st.spinner("🚀 Initializing System..."):
            try:
                # Add headers to avoid bot detection during model download
                opener = urllib.request.build_opener()
                opener.addheaders = [('User-agent', 'Mozilla/5.0')]
                urllib.request.install_opener(opener)
                urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            except Exception as e:
                st.error(f"Error loading model: {e}")
                return None
    return tf.keras.models.load_model(MODEL_PATH)

# --- 3. CUSTOM CSS (STYLING & VISIBILITY FIXES) ---
st.markdown("""
    <style>
    /* New Light Green Background from Second Reference */
    .stApp {
        background-color: #f1f8e9;
        background-image: url("https://images.unsplash.com/photo-1599676648316-29a32c668914?q=80&w=2000&auto=format&fit=crop");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    
    /* Clean up the main content container */
    [data-testid="stVerticalBlock"] > div > div > div > div > div {
        background: rgba(0,0,0,0) !important;
        border: none !important;
        box-shadow: none !important;
        padding-top: 0px !important;
    }

    /* 1. Title: Large, Dark Green, Underlined */
    .main-title {
        text-align: center;
        color: #1b5e20; /* Dark Green */
        font-size: 60px; /* Big Size */
        font-weight: bold;
        text-decoration: underline;
        margin-top: -60px;
        padding-bottom: 30px;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
    }

    /* 2. Smaller Drag & Drop and Upload Info Text */
    div.stFileUploader > div > div > div > label {
        font-size: 14px !important;
        color: #555555;
    }
    
    /* Small info text below the title */
    .intro-text {
        text-align: center;
        color: #1b5e20;
        font-size: 17px;
        margin-top: -15px;
        padding-bottom: 40px;
        max-width: 650px;
        margin-left: auto;
        margin-right: auto;
        line-height: 1.5;
    }

    /* 3. Result Label Styling (Visibility) */
    .diagnosis-label {
        color: white;
        font-size: 28px;
        font-weight: bold;
        text-align: center;
        margin-top: 25px;
        background-color: #1b5e20;
        padding: 10px;
        border-radius: 8px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }

    /* 4. RUN ANALYSIS Button (Visibility Check) */
    .stButton>button {
        width: 100% !important;
        background-color: #28a745 !important;
        color: white !important;
        font-weight: bold !important;
        border: 1px solid white !important;
        font-size: 18px !important;
    }
    .stButton>button:hover {
        background-color: white !important;
        color: #28a745 !important;
        border: 1px solid #28a745 !important;
    }

    /* Clean Image Container */
    .img-container {
        border: 10px solid white;
        margin-top: 20px;
        margin-bottom: 20px;
    }

    /* 5. Strategy Container (Bullet Point Setup, removed boxes) */
    .strategy-card {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 12px;
        padding: 30px;
        margin-top: 20px;
        color: #1E5128;
        border-left: 10px solid #28a745;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .strategy-card h3 {
        color: #d9534f; /* Red-ish for disease alert */
        margin-top: 0;
    }
    
    /* Clean Bullet Points for Management Strategies */
    .strategy-list {
        list-style-type: disc;
        margin-left: 20px;
        line-height: 1.8;
    }
    .strategy-list li {
        margin-bottom: 10px;
        font-size: 16px;
    }

    /* Hide Streamlit default elements */
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- 4. MAIN UI CONTENT ---

# The Big, Dark Green, Underlined Title
st.markdown('<div class="main-title">Plant Disease Detection App</div>', unsafe_allow_html=True)

# Spaced Info Text (removed "AI" term)
st.markdown("""
    <div class="intro-text">
        Upload one or more plant leaf images below for analysis. This system uses a 
        machine learning approach to detect potential diseases and provide immediate, 
        bullet-point management strategies.
    </div>
    """, unsafe_allow_html=True)

# Load logic
model = load_trained_model()
try:
    with open('class_indices.json', 'r') as f:
        class_indices = json.load(f)
    class_names = list(class_indices.values())
except Exception:
    class_names = []

# Centered Spacing
st.write("##")

# File Uploader (Text is smaller and centered)
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    
    # Original Image display with white border container
    _, center_img, _ = st.columns([1, 3, 1])
    with center_img:
        st.markdown('<div class="img-container">', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Run Analysis Button (with visible text)
    if st.button("🔍 RUN DISEASE ANALYSIS"):
        if model is not None and len(class_names) > 0:
            with st.spinner("Analyzing leaf patterns..."):
                img = image.resize((224, 224))
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0) / 255.0
                preds = model.predict(img_array)
                # Ensure the predicted name is handled properly if it has spaces or is "healthy"
                result = class_names[np.argmax(preds)]
                
                # Show Diagnosis Label (Highly Visible)
                st.markdown(f'<div class="diagnosis-label">Result: {result}</div>', unsafe_allow_html=True)

                # DYNAMIC MANAGEMENT SECTION
                if "healthy" in result.lower():
                    st.balloons()
                    st.success("🌱 Great! The leaf is identified as **Healthy**. No management action or treatment is required.")
                else:
                    # DYNAMIC BULLET POINTS FOR DISEASED LEAVES
                    st.markdown(f"""
                        <div class="strategy-card">
                            <h3 style='margin-bottom: 10px;'>🛡️ Immediate Management for {result}</h3>
                            <p style='font-size: 16px; margin-bottom: 20px;'>
                                To protect your harvest from the spread of this infection, take the following actions:
                            </p>
                            <hr style="border: 0.5px solid #28a745;">
                            <ul class="strategy-list">
                                <li><b>Pruning:</b> Remove the infected leaves immediately and destroy them (do not compost).</li>
                                <li><b>Treatment:</b> Spray organic or chemical fungicides like Neem oil or a sulfur-based solution.</li>
                                <li><b>Sanitation:</b> Sterilize all tools, like pruning shears, used near the infected plant.</li>
                                <li><b>Airflow:</b> Increase plant spacing or thin out leaves to improve air circulation and reduce humidity.</li>
                            </ul>
                            <p style='font-style: italic; font-size: 14px; margin-top: 20px; color: #555;'>Check the rest of your crop daily for signs of similar symptoms.</p>
                        </div>
                    """, unsafe_allow_html=True)
        else:
            st.error("Model or class names could not be initialized.")
