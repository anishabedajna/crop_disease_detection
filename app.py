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

st.set_page_config(page_title="Crop Disease Detection", page_icon="🌿", layout="centered")

@st.cache_resource
def load_trained_model():
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1000000:
        with st.spinner("🚀 Initializing..."):
            opener = urllib.request.build_opener()
            opener.addheaders = [('User-agent', 'Mozilla/5.0')]
            urllib.request.install_opener(opener)
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return tf.keras.models.load_model(MODEL_PATH)

# --- 2. THE "EXACT LOOK" CSS ---
st.markdown("""
    <style>
    /* Background */
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1542601906990-b4d3fb778b09?q=80&w=2026&auto=format&fit=crop");
        background-size: cover;
        background-position: center;
    }

    /* The Main Container (Dark Box) */
    .main-box {
        background-color: rgba(0, 0, 0, 0.85);
        padding: 50px;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }

    /* Title - No change, exactly like pic 1 */
    .header-text {
        font-size: 42px;
        font-weight: bold;
        margin-bottom: 30px;
    }

    /* KILLING THE DEFAULT DRAG-AND-DROP DESIGN */
    /* This section hides the "Drag and drop" text and the big gray border */
    [data-testid="stFileUploadDropzone"] {
        background-color: white !important;
        color: black !important;
        border: none !important;
        padding: 5px !important;
        border-radius: 4px !important;
    }
    
    [data-testid="stFileUploadDropzone"] div div {
        display: none; /* Hides the 'Limit 200MB' and icon */
    }

    /* Predict Button Styling - Vibrant Green */
    div.stButton > button {
        background-color: #28a745 !important;
        color: white !important;
        font-weight: bold !important;
        border: none !important;
        padding: 10px 20px !important;
        font-size: 16px !important;
        height: 45px;
    }

    /* Management Card styling */
    .mgmt-card {
        background-color: white;
        color: #1b5e20;
        text-align: left;
        padding: 20px;
        border-radius: 8px;
        margin-top: 20px;
        border-left: 10px solid #d32f2f;
    }

    footer, header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- 3. UI LAYOUT ---

st.markdown('<div class="main-box">', unsafe_allow_html=True)
st.markdown('<div class="header-text">Crop Disease Detection</div>', unsafe_allow_html=True)

# Loading model/classes
model = load_trained_model()
try:
    with open('class_indices.json', 'r') as f:
        class_names = list(json.load(f).values())
except:
    class_names = []

# Action Row: Choose File and Predict Button
col1, col2 = st.columns([2, 1])

with col1:
    # This now looks like a simple white "Choose File" button
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

with col2:
    # Aligns the button vertically with the uploader
    st.markdown('<div style="height: 3px;"></div>', unsafe_allow_html=True)
    predict_clicked = st.button("Predict Leaf Disease")

# --- 4. PREDICTION & CONDITIONAL MGMT ---
if uploaded_file:
    image = Image.open(uploaded_file)
    st.markdown('<p style="font-size: 20px; font-weight: bold; margin-top:20px;">Original Image</p>', unsafe_allow_html=True)
    st.image(image, width=380)
    
    if predict_clicked:
        # Preprocessing
        img = image.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        preds = model.predict(img_array)
        result = class_names[np.argmax(preds)]
        
        # Result Text
        st.markdown(f'<div style="font-size: 22px; font-weight: bold; margin-top: 20px;">Predicted Disease: {result}</div>', unsafe_allow_html=True)

        # Management Logic
        if "healthy" not in result.lower():
            st.markdown(f"""
                <div class="mgmt-card">
                    <h3 style="color: #d32f2f; margin-top: 0;">🛡️ Management Measures</h3>
                    <p>Since the crop is diagnosed with <b>{result}</b>, follow these steps:</p>
                    <ul style="line-height: 1.6;">
                        <li><b>Isolation:</b> Separate the plant to avoid spore spread.</li>
                        <li><b>Pruning:</b> Cut off and safely discard infected leaves.</li>
                        <li><b>Treatment:</b> Use a fungicide suitable for {result.split('___')[-1].replace('_', ' ')}.</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.balloons()
            st.success("Result: Healthy. No further management needed!")

st.markdown('</div>', unsafe_allow_html=True)
