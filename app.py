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

st.set_page_config(page_title="Crop Disease Detection System", page_icon="🌿", layout="wide")

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
    /* Home Page Dark Background */
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1542601906990-b4d3fb778b09?q=80&w=2026&auto=format&fit=crop");
        background-size: cover;
        background-position: center;
    }

    /* Remove Black Boxes */
    [data-testid="stVerticalBlock"] > div > div > div > div > div {
        background: rgba(0,0,0,0) !important;
        border: none !important;
    }

    /* Title Styling: Top, Centered, Underlined */
    .main-title {
        text-align: center;
        color: white;
        font-size: 50px;
        font-weight: bold;
        text-decoration: underline;
        margin-top: -50px;
        padding-bottom: 20px;
    }

    /* Navigation Buttons */
    .stButton>button {
        border-radius: 5px !important;
        background-color: #28a745 !important;
        color: white !important;
        font-weight: bold !important;
        border: 1px solid white !important;
    }
    
    /* Result Text Styling */
    .result-text {
        color: white;
        font-size: 24px;
        text-align: center;
        font-weight: bold;
        margin-top: 20px;
    }

    /* Measures Box */
    .measures-container {
        background: rgba(255, 255, 255, 0.9);
        color: #1E5128;
        padding: 30px;
        border-radius: 15px;
        border-left: 10px solid #28a745;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 4. SESSION STATE & NAVIGATION ---
if 'page' not in st.session_state:
    st.session_state.page = 'Home'
if 'prediction' not in st.session_state:
    st.session_state.prediction = None

# Centered Nav Bar
_, n1, n2, _ = st.columns([2, 1, 1, 2])
with n1:
    if st.button("🏠 HOME"): st.session_state.page = 'Home'
with n2:
    if st.button("🛡️ MANAGEMENT"): st.session_state.page = 'Management'

# --- 5. PAGE LOGIC ---

# --- PAGE: HOME (Detection) ---
if st.session_state.page == 'Home':
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
        _, img_col, _ = st.columns([1, 1.5, 1])
        with img_col:
            # White border effect from reference
            st.markdown('<div style="border: 8px solid white; display: inline-block;">', unsafe_allow_html=True)
            st.image(image, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("Predict Leaf Disease"):
            with st.spinner("Analyzing..."):
                img = image.resize((224, 224))
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0) / 255.0
                preds = model.predict(img_array)
                st.session_state.prediction = class_names[np.argmax(preds)]
        
        if st.session_state.prediction:
            st.markdown(f'<div class="result-text">Predicted Disease: {st.session_state.prediction}</div>', unsafe_allow_html=True)

# --- PAGE: MANAGEMENT (Dynamic Measures) ---
elif st.session_state.page == 'Management':
    # Light Leaf Background for this page
    st.markdown("""
        <style>
        .stApp {
            background-image: url("https://images.unsplash.com/photo-1501004318641-729e8e26bd05?q=80&w=2000&auto=format&fit=crop") !important;
        }
        h2, h3, p, li { color: #1E5128 !important; }
        </style>
        """, unsafe_allow_html=True)
    
    st.markdown("<h2 style='text-align: center;'>Management & Preventive Measures</h2>", unsafe_allow_html=True)

    if st.session_state.prediction:
        if "healthy" in st.session_state.prediction.lower():
            st.success(f"✅ The leaf is identified as **{st.session_state.prediction}**. It is a healthy leaf; no preventive measures are required at this time.")
        else:
            st.markdown(f"""
                <div class="measures-container">
                    <h3>🛡️ Safety Measures for {st.session_state.prediction}</h3>
                    <p>Based on the AI diagnosis, please follow these steps to save your crop:</p>
                    <ul>
                        <li><b>Quarantine:</b> Immediately remove and isolate the infected plant to prevent spores from spreading.</li>
                        <li><b>Organic Treatment:</b> Apply Neem Oil or a copper-based fungicide to the affected areas.</li>
                        <li><b>Pruning:</b> Cut off the diseased leaves using sterilized tools and destroy them (do not compost).</li>
                        <li><b>Watering:</b> Avoid overhead watering; water the soil directly to keep leaves dry.</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.info("⚠️ No diagnosis found. Please go to the Home page and upload a leaf image first.")

# Footer
st.markdown("<p style='text-align: center; color: white; margin-top: 50px;'>A Machine Learning Approach for Crop Disease Management © 2026</p>", unsafe_allow_html=True)
