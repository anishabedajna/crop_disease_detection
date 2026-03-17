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
.stApp {
    background: linear-gradient(rgba(220,255,220,0.7), rgba(220,255,220,0.7)),
                url("https://images.unsplash.com/photo-1501004318641-b39e6451bec6");
    background-size: cover;
}

.block-container {
    text-align: center;
}

[data-testid="stFileUploaderFileName"] {
    color: black !important;
    font-weight: 600;
}

div.stButton > button {
    background-color: #2e7d32 !important;
    color: white !important;
    font-weight: bold;
}

header, footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- TITLE ---
st.markdown("""
<div style="text-align:center; margin-top:-40px;">
    <h1 style="color:#111; text-decoration: underline;">
        Plant Disease Detection App
    </h1>
</div>
""", unsafe_allow_html=True)

# --- SUBTITLE ---
st.markdown("""
<p style='text-align:center; font-size:18px; color:#222; margin-top:20px;'>
A machine learning-based approach to detect plant crop diseases and recommend appropriate measures.
</p>
""", unsafe_allow_html=True)

# --- UPLOADER ---
uploaded_file = st.file_uploader("Upload leaf image", type=["jpg", "jpeg", "png"])
analyze = st.button("Analyze")

# --- PREDICTION ---
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, width=300)

    if analyze:
        img = image.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        preds = model.predict(img_array)
        result = class_names[np.argmax(preds)]

        # --- RESULT BOX ---
        if "healthy" in result.lower():
            st.markdown(f"""
            <div style="background-color:white; padding:12px; border-radius:10px; width:60%; margin:auto;">
                <p style="color:#1b5e20; font-size:18px; font-weight:bold; text-align:center;">
                    ✅ This crop is Healthy
                </p>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown(f"""
            <div style="background-color:white; padding:12px; border-radius:10px; width:60%; margin:auto;">
                <p style="color:#8b0000; font-size:18px; font-weight:bold; text-align:center;">
                    ❌ This crop is Diseased: {result}
                </p>
            </div>
            """, unsafe_allow_html=True)

            # --- MEASURES BOX ---
            st.markdown("""
            <div style="background-color:white; padding:15px; border-radius:10px; width:65%; margin:15px auto; text-align:center;">
                <h3 style="color:black;">Recommended Measures</h3>
            """, unsafe_allow_html=True)

            if result in solutions:
                for step in solutions[result]:
                    st.markdown(f"<p style='color:black;'>• {step}</p>", unsafe_allow_html=True)
            else:
                st.markdown("<p style='color:black;'>• Remove infected parts</p>", unsafe_allow_html=True)
                st.markdown("<p style='color:black;'>• Apply fungicide</p>", unsafe_allow_html=True)
                st.markdown("<p style='color:black;'>• Maintain proper care</p>", unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)
