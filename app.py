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
    /* Default Background (Dark Leaf) */
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1542601906990-b4d3fb778b09?q=80&w=2026&auto=format&fit=crop");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }

    /* Remove Streamlit Default Boxes */
    [data-testid="stVerticalBlock"] > div > div > div > div > div {
        background: rgba(0,0,0,0) !important;
        border: none !important;
        box-shadow: none !important;
    }

    /* Title: Top, Centered, Underlined */
    .main-title {
        text-align: center;
        color: white;
        font-size: 52px;
        font-weight: bold;
        text-decoration: underline;
        margin-top: -60px;
        padding-bottom: 30px;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.7);
    }

    /* Navigation & Predict Buttons */
    .stButton>button {
        border-radius: 4px !important;
        background-color: #28a745 !important;
        color: white !important;
        font-weight: bold !important;
        border: 1px solid white !important;
        width: auto !important;
        padding: 10px 25px !important;
    }
    
    .stButton>button:hover {
        background-color: #ffffff !important;
        color: #28a745 !important;
    }

    /* White Image Border matching reference image */
    .img-frame {
        border: 10px solid white;
        display: inline-block;
        margin-top: 20px;
    }

    /* Result Text Styling */
    .diagnosis-label {
        color: white;
        font-size: 26px;
        font-weight: bold;
        margin-top: 25px;
        text-align: center;
    }

    /* Measures Container */
    .measures-card {
        background: rgba(255, 255, 255, 0.95);
        color: #1E5128;
        padding: 35px;
        border-radius: 12px;
        border-left: 10px solid #28a745;
        margin-top: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 4. NAVIGATION STATE ---
if 'page' not in st.session_state:
    st.session_state.page = 'Home'
if 'result_stored' not in st.session_state:
    st.session_state.result_stored = None

# Centered Navigation Bar
_, n1, n2, _ = st.columns([2, 1, 1, 2])
with n1:
    if st.button("🏠 HOME"): st.session_state.page = 'Home'
with n2:
    if st.button("🛡️ MANAGEMENT"): st.session_state.page = 'Management'

# --- 5. PAGE CONTENT LOGIC ---

# --- PAGE 1: HOME (Detection Section) ---
if st.session_state.page == 'Home':
    st.markdown('<div class="main-title">Crop Disease Detection System</div>', unsafe_allow_html=True)
    
    model = load_trained_model()
    try:
        with open('class_indices.json', 'r') as f:
            class_indices = json.load(f)
        class_names = list(class_indices.values())
    except:
        class_names = []

    # Upload Section
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        
        # Original Image display with white border container
        st.markdown('<div style="text-align: center; color: white; font-size: 20px; margin-top: 20px;">Original Image</div>', unsafe_allow_html=True)
        _, img_col, _ = st.columns([1, 1, 1])
        with img_col:
            st.markdown('<div class="img-frame">', unsafe_allow_html=True)
            st.image(image, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Prediction Trigger
        if st.button("Predict Leaf Disease"):
            with st.spinner("AI is analyzing leaf patterns..."):
                img = image.resize((224, 224))
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0) / 255.0
                preds = model.predict(img_array)
                st.session_state.result_stored = class_names[np.argmax(preds)]
        
        # Result Display
        if st.session_state.result_stored:
            st.markdown(f'<div class="diagnosis-label">Predicted Disease: {st.session_state.result_stored}</div>', unsafe_allow_html=True)

# --- PAGE 2: MANAGEMENT (Specific Measures Section) ---
elif st.session_state.page == 'Management':
    # Dynamic styling for light leaf background
    st.markdown("""
        <style>
        .stApp {
            background-image: url("https://images.unsplash.com/photo-1501004318641-729e8e26bd05?q=80&w=2000&auto=format&fit=crop") !important;
        }
        h2, h3, p, li { color: #1E5128 !important; }
        </style>
        """, unsafe_allow_html=True)
    
    st.markdown("<h2 style='text-align: center; font-weight: bold;'>Prevention & Management Strategies</h2>", unsafe_allow_html=True)

    if st.session_state.result_stored:
        # Check if the detected result is "Healthy"
        if "healthy" in st.session_state.result_stored.lower():
            st.success(f"✨ Success! The leaf is identified as **{st.session_state.result_stored}**. This is a healthy leaf, so no preventive measures or treatments are required.")
        else:
            # Display specific management measures for diseased leaves
            st.markdown(f"""
                <div class="measures-card">
                    <h3 style='margin-top: 0;'>🛡️ Management for {st.session_state.result_stored}</h3>
                    <p>To protect your harvest from the spread of this infection, take the following actions immediately:</p>
                    <hr style="border: 0.5px solid #28a745;">
                    <ul>
                        <li><b>Sanitation:</b> Sterilize all pruning shears and farming tools used near the infected plant.</li>
                        <li><b>Disposal:</b> Carefully remove the diseased leaves and burn them or bury them deep in soil. <b>Do not</b> use them for compost.</li>
                        <li><b>Application:</b> Spray organic fungicides like Neem oil or a mild sulfur-based solution.</li>
                        <li><b>Airflow:</b> Thin out the plants to increase air circulation, which reduces moisture levels that help fungi grow.</li>
                    </ul>
                    <p><i>Check the rest of your crop daily for signs of similar symptoms.</i></p>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ No Analysis Data Found. Please go to the Home page and analyze a leaf image first.")

# --- FOOTER ---
st.markdown("<p style='text-align: center; color: white; margin-top: 80px; font-size: 14px;'>A Machine Learning Approach for Crop Disease Management © 2026</p>", unsafe_allow_html=True)
