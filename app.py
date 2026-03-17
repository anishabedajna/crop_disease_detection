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

# --- DISEASE SOLUTIONS ---
solutions = {
    "Tomato___Late_blight": [
        "Remove infected leaves immediately",
        "Apply fungicide like Mancozeb",
        "Avoid overhead watering",
    ],
    "Tomato___Early_blight": [
        "Use crop rotation",
        "Apply copper-based fungicide",
        "Remove affected leaves",
    ],
    "Potato___Late_blight": [
        "Use certified seeds",
        "Improve air circulation",
        "Apply fungicide regularly",
    ],
}

# --- CSS ---
st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(rgba(220,255,220,0.7), rgba(220,255,220,0.7)),
                url("https://images.unsplash.com/photo-1501004318641-b39e6451bec6");
    background-size: cover;
    background-position: center;
}

/* Center everything */
.block-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}

/* File uploader */
section[data-testid="stFileUploader"] {
    max-width: 350px;
    margin: auto;
    text-align: center;
}

/* File name color */
[data-testid="stFileUploaderFileName"] {
    color: black !important;
    font-weight: 600;
    text-align: center;
}

/* Center image */
img {
    display: block;
    margin-left: auto !important;
    margin-right: auto !important;
}

/* Button center */
div.stButton {
    display: flex;
    justify-content: center;
}

div.stButton > button {
    background-color: #2e7d32 !important;
    color: white !important;
    font-weight: bold;
    border-radius: 6px;
}

/* Result box */
.result-box {
    background-color: rgba(255,255,255,0.9);
    padding: 12px;
    border-radius: 8px;
    margin-top: 15px;
    color: #111;
    font-weight: bold;
    text-align: center;
}

/* Measures title */
.measures-title {
    color: #000000;
    font-weight: bold;
    margin-top: 15px;
    text-align: center;
}

/* Bullet points */
ul {
    display: inline-block;
    text-align: left;
    color: #222;
}

/* Hide header/footer */
header, footer {visibility: hidden;}

</style>
""", unsafe_allow_html=True)

# --- TITLE + SUBTITLE (NO ERROR VERSION) ---
st.markdown("""
<div style="text-align: center; margin-top: -10px;">

    <h1 style="color:#111; text-decoration: underline; margin-bottom: 0px;">
        Plant Disease Detection App
    </h1>

    <p style="color:#222; font-size:18px; margin-top: 20px;">
        A machine learning-based approach to detect plant crop diseases and recommend appropriate measures.
    </p>

</div>
""", unsafe_allow_html=True)

# --- FILE UPLOADER ---
uploaded_file = st.file_uploader("Upload leaf image", type=["jpg", "jpeg", "png"])

# --- BUTTON ---
analyze = st.button("Analyze")

# --- PREDICTION ---
if uploaded_file:
    image = Image.open(uploaded_file)

    # Center image
    st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
    st.image(image, width=300)
    st.markdown("</div>", unsafe_allow_html=True)

    if analyze:
        img = image.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        preds = model.predict(img_array)
        result = class_names[np.argmax(preds)]

        if "healthy" in result.lower():
            st.markdown('<div class="result-box">✅ This crop is Healthy</div>', unsafe_allow_html=True)

        else:
            st.markdown(f'<div class="result-box">❌ This crop is Diseased: {result}</div>', unsafe_allow_html=True)

            st.markdown('<div class="measures-title">🌿 Recommended Measures</div>', unsafe_allow_html=True)

            if result in solutions:
                for step in solutions[result]:
                    st.markdown(f"- {step}")
            else:
                st.markdown("- Remove infected parts")
                st.markdown("- Apply general fungicide")
                st.markdown("- Keep plant in dry conditions")
