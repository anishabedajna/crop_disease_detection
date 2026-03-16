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

st.set_page_config(page_title="Crop Disease Detector", page_icon="🌿", layout="wide")

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
                st.error(f"Error loading model: {e}")
                return None
    return tf.keras.models.load_model(MODEL_PATH)

# --- 3. CUSTOM CSS: THE "A TO Z" LOOK ---
st.markdown("""
    <style>
    /* Home Background */
    [data-testid="stAppViewContainer"] {
        background-image: url("https://images.unsplash.com/photo-1518531933037-91b2f5f229cc?q=80&w=2000&auto=format&fit=crop");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }

    [data-testid="stSidebar"] { display: none; }

    /* The Central Black Box from Third Image */
    .matching-overlay {
        background: rgba(0, 0, 0, 0.82) !important;
        padding: 40px;
        border-radius: 15px;
        max-width: 850px;
        margin: auto;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-top: 30px;
    }

    /* Green Button from Reference */
    .stButton>button {
        border-radius: 4px !important;
        background-color: #28a745 !important;
        color: white !important;
        font-weight: bold !important;
        padding: 8px 20px !important;
        border: none !important;
    }

    /* White Image Border from Reference */
    .img-border {
        border: 8px solid white;
        display: inline-block;
        margin-bottom: 20px;
    }

    /* Prediction Text */
    .pred-text {
        color: white;
        font-size: 24px;
        font-family: 'Segoe UI', sans-serif;
        margin-top: 20px;
    }

    /* Safety Measures Box */
    .measures-box {
        background: rgba(255, 255, 255, 0.95);
        color: #1E5128;
        padding: 20px;
        border-radius: 10px;
        margin-top: 25px;
        text-align: left;
        border-left: 8px solid #ff4b4b;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 4. NAVIGATION ---
if 'page' not in st.session_state:
    st.session_state.page = 'Home'
if 'current_pred' not in st.session_state:
    st.session_state.current_pred = None

# Navigation Header
_, col1, col2, col3, _ = st.columns([1.5, 2, 2, 2, 1.5])
with col1:
    if st.button("🏠 HOME"): st.session_state.page = 'Home'
with col2:
    if st.button("🔍 ANALYZE LEAF"): 
        st.session_state.page = 'Detection'
        st.session_state.current_pred = None
with col3:
    if st.button("🛡️ MEASURES"): st.session_state.page = 'Measures'

# --- 5. PAGE LOGIC ---

if st.session_state.page == 'Home':
    st.markdown("<h1 style='text-align:center; color:white; font-size:60px;'>CropCare AI</h1>", unsafe_allow_html=True)

elif st.session_state.page == 'Detection':
    # Switch to Lighter Background
    st.markdown("<style>[data-testid='stAppViewContainer'] { background-image: url('https://images.unsplash.com/photo-1501004318641-729e8e26bd05?q=80&w=2000&auto=format&fit=crop') !important; }</style>", unsafe_allow_html=True)
    
    st.markdown('<div class="matching-overlay">', unsafe_allow_html=True)
    st.markdown("<h2 style='color:white;'>Leaf Disease Detection</h2>", unsafe_allow_html=True)
    
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
        st.markdown("<p style='color:white; font-size:20px;'>Original Image</p>", unsafe_allow_html=True)
        
        # Center the white-bordered image
        st.markdown('<div class="img-border">', unsafe_allow_html=True)
        st.image(image, width=300)
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("Predict Leaf Disease"):
            with st.spinner("Analyzing..."):
                img = image.resize((224, 224))
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0) / 255.0
                preds = model.predict(img_array)
                st.session_state.current_pred = class_names[np.argmax(preds)]
        
        if st.session_state.current_pred:
            st.markdown(f'<div class="pred-text">Predicted Disease: {st.session_state.current_pred}</div>', unsafe_allow_html=True)
            
            # Smart Safety Measures Logic
            if "healthy" in st.session_state.current_pred.lower():
                st.success("✨ This leaf is healthy! No safety measures needed.")
            else:
                st.markdown(f"""
                <div class="measures-box">
                    <h3 style='color:#d9534f;'>🚨 Safety Measures for {st.session_state.current_pred}</h3>
                    <ul>
                        <li><b>Isolate:</b> Move the plant away from healthy ones.</li>
                        <li><b>Pruning:</b> Cut off and burn the infected leaves.</li>
                        <li><b>Sanitation:</b> Clean your hands and tools after touching the plant.</li>
                        <li><b>Organic Spray:</b> Use a Neem oil or copper-based fungicide.</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.page == 'Measures':
    st.markdown("<style>[data-testid='stAppViewContainer'] { background-image: url('https://images.unsplash.com/photo-1501004318641-729e8e26bd05?q=80&w=2000&auto=format&fit=crop') !important; }</style>", unsafe_allow_html=True)
    st.markdown("<div style='text-align:center;'><h2>🛡️ General Prevention</h2><p>Proper spacing and soil health are key to avoiding disease.</p></div>", unsafe_allow_html=True)
