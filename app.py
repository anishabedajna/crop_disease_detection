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

# --- DISEASE SOLUTIONS (IMPORTANT ADDITION) ---
solutions = {
    "Tomato___Late_blight": "Remove infected leaves. Apply fungicide like Mancozeb. Avoid overhead watering.",
    "Tomato___Early_blight": "Use crop rotation. Apply copper-based fungicides. Remove affected leaves.",
    "Potato___Late_blight": "Use certified seeds. Apply fungicide. Improve air circulation.",
    "Apple___Black_rot": "Prune infected branches. Apply fungicide. Remove fallen fruits.",
    "Corn___Common_rust": "Use resistant varieties. Apply fungicide if severe.",
}

# --- CUSTOM CSS (same as before) ---
st.markdown("""
<style>
.stApp {
    background: linear-gradient(rgba(220,255,220,0.7), rgba(220,255,220,0.7)),
                url("https://images.unsplash.com/photo-1501004318641-b39e6451bec6");
    background-size: cover;
    background-position: center;
}
.block-container { text-align: center; }

.title {
    font-size: 42px;
    font-weight: bold;
    color: #1a1a1a;
    margin-top: 30px;
}

.subtitle {
    font-size: 18px;
    color: #333333;
    margin-bottom: 25px;
}

section[data-testid="stFileUploader"] {
    max-width: 400px;
    margin: auto;
}

img {
    display: block;
    margin-left: auto;
    margin-right: auto;
}

.stSuccess {
    background-color: rgba(255,255,255,0.9) !important;
    color: #111 !important;
    font-weight: bold;
    text-align: center;
}

.stWarning {
    background-color: rgba(255,255,200,0.9) !important;
    color: #222 !important;
    font-weight: bold;
    text-align: center;
}

header, footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- UI ---
st.markdown('<div class="title">Plant Disease Detection App</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a plant leaf image to detect disease instantly.</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload leaf image", type=["jpg", "jpeg", "png"])

# --- AUTO PREDICTION ---
if uploaded_file:
    image = Image.open(uploaded_file)

    st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
    st.image(image, width=300)
    st.markdown("</div>", unsafe_allow_html=True)

    # --- PROCESS IMAGE ---
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    result = class_names[np.argmax(preds)]

    # --- HEALTH CHECK ---
    if "healthy" in result.lower():
        st.success("✅ The plant is Healthy")
    else:
        st.error(f"❌ Disease Detected: {result}")

        # --- SHOW SOLUTION ---
        if result in solutions:
            st.warning(f"🌿 Treatment: {solutions[result]}")
        else:
            st.warning("Apply general fungicide and remove infected parts.")                                             
