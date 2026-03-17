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
        with st.spinner("🚀 Initializing Model..."):
            opener = urllib.request.build_opener()
            opener.addheaders = [('User-agent', 'Mozilla/5.0')]
            urllib.request.install_opener(opener)
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return tf.keras.models.load_model(MODEL_PATH)

# --- 2. CUSTOM CSS (FINAL PERFECT DESIGN) ---
st.markdown("""
<style>

/* BACKGROUND */
.stApp {
    background-image: url("https://images.unsplash.com/photo-1542601906990-b4d3fb778b09?q=80&w=2026&auto=format&fit=crop");
    background-size: cover;
}

/* MAIN BOX */
.main-box {
    background: rgba(0,0,0,0.85);
    padding: 40px;
    border-radius: 12px;
    text-align: center;
    width: 60%;
    margin: auto;
    color: white;
}

/* TITLE */
.header-text {
    font-size: 38px;
    font-weight: bold;
    margin-bottom: 25px;
}

/* 🔥 HIDE DEFAULT FILE UPLOADER COMPLETELY */
[data-testid="stFileUploader"] {
    opacity: 0;
    position: absolute;
    z-index: -1;
}

/* FAKE BUTTON */
.upload-btn {
    background: white;
    color: black;
    padding: 8px 15px;
    border-radius: 5px;
    display: inline-block;
    cursor: pointer;
    font-size: 14px;
}

/* GREEN BUTTON */
div.stButton > button {
    background-color: #28a745 !important;
    color: white !important;
    font-weight: bold;
    border-radius: 5px;
}

/* REMOVE HEADER FOOTER */
header, footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)



# --- 3. UI LAYOUT ---
st.markdown('<div class="main-box">', unsafe_allow_html=True)
st.markdown('<div class="header-text">Crop Disease Detection</div>', unsafe_allow_html=True)

# Fake button text
st.markdown('<div class="upload-btn">Choose File</div>', unsafe_allow_html=True)

# REAL uploader (hidden)
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

predict_clicked = st.button("Predict Leaf Disease")

# --- 4. PREDICTION ---
if uploaded_file:
    image = Image.open(uploaded_file)

    st.markdown('<p style="font-size: 20px; font-weight: bold; margin-top:20px;">Original Image</p>', unsafe_allow_html=True)
    st.image(image, width=320)

    if predict_clicked:
        # Preprocess
        img = image.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        preds = model.predict(img_array)
        result = class_names[np.argmax(preds)]

        # Result
        st.markdown(f'<div style="font-size: 22px; font-weight: bold; margin-top: 20px;">Predicted Disease: {result}</div>', unsafe_allow_html=True)

        # Management
        if "healthy" not in result.lower():
            st.markdown(f"""
            <div class="mgmt-card">
                <h3 style="color:#d32f2f;">🛡️ Management Measures</h3>
                <ul>
                    <li><b>Isolation:</b> Separate infected plant.</li>
                    <li><b>Pruning:</b> Remove affected leaves.</li>
                    <li><b>Fungicide:</b> Use treatment for {result.split('___')[-1].replace('_',' ')}</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.balloons()
            st.success("Healthy leaf detected!")

st.markdown('</div>', unsafe_allow_html=True)
