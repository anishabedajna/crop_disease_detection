import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import urllib.request
import json

# --- CONFIG ---
MODEL_URL = "https://github.com/anishabedajna/crop_disease_detection/releases/download/v1/plant_disease_model.h5"
MODEL_PATH = "plant_disease_model.h5"

st.set_page_config(page_title="Plant Disease Detection", page_icon="🌿", layout="centered")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Loading model..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# --- LOAD CLASS NAMES ---
try:
    with open("class_indices.json", "r") as f:
        class_names = list(json.load(f).values())
except:
    class_names = []

# --- CUSTOM CSS ---
st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(rgba(220,255,220,0.7), rgba(220,255,220,0.7)),
                url("https://images.unsplash.com/photo-1501004318641-b39e6451bec6");
    background-size: cover;
    background-position: center;
}

/* Center content */
.block-container {
    text-align: center;
}

/* Title */
.title {
    font-size: 42px;
    font-weight: bold;
    color: #1a1a1a;
    margin-top: 30px;
}

/* Subtitle */
.subtitle {
    font-size: 18px;
    color: #333333;
    margin-bottom: 25px;
}

/* File uploader label */
label, .stFileUploader label {
    color: #222222 !important;
    font-weight: 600;
}

/* Smaller uploader */
section[data-testid="stFileUploader"] {
    max-width: 400px;
    margin: auto;
}

/* Drag box */
section[data-testid="stFileUploader"] div {
    font-size: 14px !important;
    padding: 10px !important;
}

/* Center images */
img {
    display: block;
    margin-left: auto;
    margin-right: auto;
}

/* Info box */
.info-box {
    background-color: rgba(255,255,255,0.75);
    padding: 10px;
    border-radius: 6px;
    margin-top: 10px;
    color: #222;
    font-weight: 500;
}

/* Button */
div.stButton > button {
    background-color: #2e7d32 !important;
    color: white !important;
    font-weight: bold;
    border-radius: 6px;
}

/* Result styling */
.stSuccess {
    background-color: rgba(255,255,255,0.9) !important;
    color: #111 !important;
    font-weight: bold;
    border-radius: 6px;
    text-align: center;
}

.stWarning {
    background-color: rgba(255,255,200,0.9) !important;
    color: #222 !important;
    font-weight: bold;
    border-radius: 6px;
    text-align: center;
}

/* Hide header/footer */
header, footer {visibility: hidden;}

</style>
""", unsafe_allow_html=True)

# --- UI TEXT ---
st.markdown('<div class="title">Plant Disease Detection App</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload one or more plant leaf images to detect diseases.</div>', unsafe_allow_html=True)

# --- FILE UPLOADER ---
uploaded_file = st.file_uploader(
    "Upload leaf image (jpg, jpeg, png)",
    type=["jpg", "jpeg", "png"]
)

st.markdown('<div class="info-box">Please upload a plant leaf image to get started.</div>', unsafe_allow_html=True)

predict_clicked = st.button("Detect Disease")

# --- PREDICTION ---
if uploaded_file:
    image = Image.open(uploaded_file)

    # Center image
    st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
    st.image(image, width=300)
    st.markdown("</div>", unsafe_allow_html=True)

    if predict_clicked:
        img = image.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        preds = model.predict(img_array)
        result = class_names[np.argmax(preds)]

        st.success(f"Prediction: {result}")

        if "healthy" not in result.lower():
            st.warning("Apply proper treatment and isolate the plant.")
        else:
            st.success("Healthy leaf detected!")
