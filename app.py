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

st.set_page_config(page_title="CropCare AI", page_icon="🌿", layout="wide")

# --- 2. CACHED MODEL LOADING ---
@st.cache_resource
def load_trained_model():
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1000000:
        with st.spinner("🚀 Initializing AI Engine..."):
            try:
                opener = urllib.request.build_opener()
                opener.addheaders = [('User-agent', 'Mozilla/5.0')]
                urllib.request.install_opener(opener)
                urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            except Exception as e:
                st.error(f"Error: {e}")
                return None
    return tf.keras.models.load_model(MODEL_PATH)

# --- 3. CUSTOM CSS FOR FULL LEAF BACKGROUND & CENTERED OPTIONS ---
st.markdown("""
    <style>
    /* Full Page Leaf Background */
    [data-testid="stAppViewContainer"] {
        background-image: url("https://images.unsplash.com/photo-1518531933037-91b2f5f229cc?q=80&w=2000&auto=format&fit=crop");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }

    /* Hide the Sidebar entirely to match your request */
    [data-testid="stSidebar"] {
        display: none;
    }

    /* Main Container Styling */
    .main-box {
        background: rgba(0, 0, 0, 0.6); /* Darker overlay for better text contrast */
        padding: 50px;
        border-radius: 30px;
        backdrop-filter: blur(8px);
        text-align: center;
        color: white;
        max-width: 900px;
        margin: auto;
    }

    /* Centered Title */
    .huge-title {
        font-size: 60px;
        font-weight: 900;
        color: #ffffff;
        text-shadow: 3px 3px 10px rgba(0,0,0,0.8);
        margin-bottom: 10px;
    }

    /* Horizontal Menu Buttons */
    .stButton>button {
        border-radius: 50px !important;
        background-color: #28a745 !important; /* Bright Green */
        color: white !important;
        font-weight: bold !important;
        padding: 10px 30px !important;
        border: 2px solid white !important;
        transition: 0.4s;
    }
    .stButton>button:hover {
        background-color: white !important;
        color: #28a745 !important;
        transform: scale(1.05);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 4. TOP TITLE & NAVIGATION (CENTERED) ---
st.markdown('<h1 class="huge-title">🌿 CROPCARE AI</h1>', unsafe_allow_html=True)

# Initialize "page" state if it doesn't exist
if 'page' not in st.session_state:
    st.session_state.page = 'Home'

# Centered Navigation Menu
col1, col2, col3 = st.columns([1,1,1])
with col1:
    if st.button("🏠 HOME"): st.session_state.page = 'Home'
with col2:
    if st.button("🔍 ANALYZE LEAF"): st.session_state.page = 'Detection'
with col3:
    if st.button("🛡️ MEASURES"): st.session_state.page = 'Measures'

st.markdown("<br>", unsafe_allow_html=True)

# --- 5. PAGE LOGIC ---

# PAGE 1: HOME
if st.session_state.page == 'Home':
    st.markdown("""
        <div class="main-box">
            <h2 style='color: #4CAF50;'>Your AI-Powered Crop Doctor</h2>
            <p style='font-size: 18px;'>Welcome to the next generation of agriculture. 
            Upload leaf photos to detect 38 different plant diseases instantly using 
            advanced Deep Learning technology.</p>
            <hr style='border: 0.5px solid #555;'>
            <p>Click <b>ANALYZE LEAF</b> above to start your diagnosis.</p>
        </div>
        """, unsafe_allow_html=True)

# PAGE 2: DETECTION
elif st.session_state.page == 'Detection':
    st.markdown('<div class="main-box">', unsafe_allow_html=True)
    st.header("🔍 Leaf Analysis")
    
    model = load_trained_model()
    try:
        with open('class_indices.json', 'r') as f:
            class_indices = json.load(f)
        class_names = list(class_indices.values())
    except:
        st.error("Missing class_indices.json file!")
        class_names = []

    uploaded_file = st.file_uploader("Drop your leaf photo here", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption='Target Leaf', width=300)
        
        if st.button("START AI DIAGNOSIS"):
            with st.spinner("Processing..."):
                img = image.resize((224, 224))
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0) / 255.0
                
                preds = model.predict(img_array)
                result = class_names[np.argmax(preds)]
                st.success(f"Diagnosis: {result}")
                st.balloons()
    st.markdown('</div>', unsafe_allow_html=True)

# PAGE 3: MEASURES
elif st.session_state.page == 'Measures':
    st.markdown("""
        <div class="main-box">
            <h2 style='color: #4CAF50;'>Prevention & Management</h2>
            <ul style='text-align: left; display: inline-block;'>
                <li><b>Crop Rotation:</b> Avoid planting same families together.</li>
                <li><b>Sanitation:</b> Keep tools clean of soil pathogens.</li>
                <li><b>Early Action:</b> Remove spotted leaves immediately.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<p style='text-align: center; color: white; margin-top: 50px;'>CropCare AI Major Project © 2026</p>", unsafe_allow_html=True)
