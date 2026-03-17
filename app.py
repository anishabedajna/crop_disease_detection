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

st.set_page_config(page_title="Plant Disease Detection", layout="centered")

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
        # Assuming JSON is { "0": "DiseaseName", "1": "Healthy" }
        data = json.load(f)
        class_names = [data[str(i)] for i in range(len(data))]
except:
    class_names = []

# --- DISEASE SOLUTIONS ---
solutions = {
    "Tomato___Late_blight": "Remove infected leaves. Apply fungicide like Mancozeb. Avoid overhead watering.",
    "Tomato___Early_blight": "Use crop rotation. Apply copper-based fungicides. Remove affected leaves.",
    "Potato___Late_blight": "Use certified seeds. Apply fungicide. Improve air circulation.",
    "Apple___Black_rot": "Prune infected branches. Apply fungicide. Remove fallen fruits.",
    "Corn___Common_rust": "Use resistant varieties. Apply fungicide if severe.",
}

# --- EXACT UI CSS ---
st.markdown("""
<style>
    /* Background of the whole app */
    .stApp {
        background-color: #D9D9D9 !important;
    }
    
    /* Center text alignment */
    .block-container {
        text-align: center;
        padding-top: 2rem;
    }

    /* Main Title Styling */
    .main-title {
        color: #1B3022 !important;
        font-family: 'Arial Black', sans-serif;
        font-size: 45px !important;
        font-weight: 900;
        margin-bottom: 0px !important;
        text-transform: uppercase;
    }
    
    .sub-title {
        color: #1B3022;
        font-size: 18px;
        margin-bottom: 30px;
    }

    /* Analyze Button Styling */
    div.stButton > button {
        background-color: #1B3022 !important;
        color: white !important;
        border-radius: 5px !important;
        padding: 10px 40px !important;
        font-size: 18px !important;
        font-weight: bold;
        border: none !important;
        width: 100%;
        margin-top: 10px;
    }

    /* Result Box (White label) */
    .result-label {
        background-color: white;
        padding: 15px;
        border-radius: 5px;
        border: 1px solid #ccc;
        color: black;
        font-weight: bold;
        text-align: left;
        margin-top: 20px;
    }

    /* Measures Popup Box */
    .measures-box {
        background-color: #f8f9fa;
        border-left: 5px solid #1B3022;
        padding: 15px;
        text-align: left;
        margin-top: 15px;
        border-radius: 4px;
        color: #333;
    }

    /* Hide Streamlit branding */
    header, footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- UI HEADER ---
st.markdown('<p class="main-title">PLANT DISEASE DETECTOR</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Please upload the image of the plant leaf for the analysis</p>', unsafe_allow_html=True)

# --- FILE UPLOADER ---
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    
    # Display image exactly like the preview
    st.image(image, use_container_width=True)

    # --- ANALYZE BUTTON ---
    if st.button("ANALYZE"):
        # Image Processing
        img = image.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        preds = model.predict(img_array)
        result = class_names[np.argmax(preds)]

        # 1. Show the Result box exactly as requested
        st.markdown(f'<div class="result-label">Result = {result}</div>', unsafe_allow_html=True)

        # 2. Condition: If diseased, show measures
        if "healthy" not in result.lower():
            measure_text = solutions.get(result, "Apply general fungicide and remove infected parts.")
            
            st.markdown(f"""
            <div class="measures-box">
                <h4 style="margin-top:0; color:#1B3022;">⚠️ Recommended Measures:</h4>
                <p>{measure_text}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.balloons()
            st.success("This plant is healthy!")
