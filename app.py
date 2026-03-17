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

# --- CUSTOM UI (LIGHT GREEN STYLE) ---
st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(rgba(200,255,200,0.6), rgba(200,255,200,0.6)),
                url("https://images.unsplash.com/photo-1501004318641-b39e6451bec6");
    background-size: cover;
    background-position: center;
}

/* Center everything */
.block-container {
    text-align: center;
}

/* Title */
.title {
    font-size: 40px;
    font-weight: bold;
    color: #2e7d32;
    margin-top: 30px;
}

/* Subtitle */
.subtitle {
    font-size: 18px;
    color: #4e944f;
    margin-bottom: 30px;
}

/* Button */
div.stButton > button {
    background-color: #2e7d32 !important;
    color: white !important;
    font-weight: bold;
    border-radius: 6px;
}

/* Info box */
.info-box {
    background-color: rgba(200,255,200,0.7);
    padding: 10px;
    border-radius: 5px;
    margin-top: 15px;
    color: #2e7d32;
}

header, footer {visibility: hidden;}

</style>
""", unsafe_allow_html=True)

# --- UI TEXT ---
st.markdown('<div class="title">Plant Disease Detection App</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload one or more plant leaf images to detect diseases.</div>', unsafe_allow_html=True)

# --- FILE UPLOADER ---
uploaded_file = st.file_uploader(
    "Upload one or more leaf images (jpg, jpeg, png)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False
)

st.markdown('<div class="info-box">Please upload a plant leaf image to get started.</div>', unsafe_allow_html=True)

predict_clicked = st.button("Detect Disease")

# --- PREDICTION ---
if uploaded_file:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", width=300)

    if predict_clicked:
        img = image.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        preds = model.predict(img_array)
        result = class_names[np.argmax(preds)]

        st.success(f"Prediction: {result}")

        if "healthy" not in result.lower():
            st.warning("Apply proper treatment and isolate plant.")
        else:
            st.success("Healthy leaf detected!")
