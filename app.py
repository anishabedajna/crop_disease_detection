import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import urllib.request
import json

# --- CONFIG ---
MODEL_URL = "https://github.com/anishabedajna/crop_disease_detection/releases/download/v1/plant_disease_model.h5"
CLASS_INDICES_URL = "https://github.com/anishabedajna/crop_disease_detection/releases/download/v1/class_indices.json"
MODEL_PATH = "plant_disease_model.h5"
CLASS_INDICES_PATH = "class_indices.json"

st.set_page_config(page_title="Plant Disease Detector", layout="centered")

# --- CUSTOM CSS FOR EXACT DESIGN ---
st.markdown("""
<style>
    /* 1. Full page background with leaf image and light overlay */
    .stApp {
        background: linear-gradient(rgba(255,255,255,0.5), rgba(255,255,255,0.5)), 
                    url("https://images.unsplash.com/photo-1599385552300-85f2fa6b3068?q=80");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }

    /* 2. Main Card Container */
    .block-container {
        max-width: 700px !important;
        margin: auto;
        padding: 50px !important;
        background-color: rgba(235, 235, 235, 0.96); /* Light grey card */
        border-radius: 30px;
        box-shadow: 0px 15px 40px rgba(0,0,0,0.3);
        text-align: center;
    }

    /* 3. Title: Bold, Green, Underlined, Centered */
    .main-title {
        color: #0E2416 !important;
        font-family: 'Arial Black', sans-serif;
        font-size: 42px !important;
        text-decoration: underline;
        font-weight: 900;
        margin-bottom: 5px !important;
    }

    .sub-title {
        color: #0E2416 !important;
        font-size: 1.2rem;
        margin-bottom: 30px;
        font-weight: 700;
    }

    /* 4. Centered and smaller file uploader */
    [data-testid="stFileUploader"] {
        width: 85% !important;
        margin: 0 auto !important;
    }

    /* 5. Analyze Button (Dark Green) */
    div.stButton > button {
        background-color: #0E2416 !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 12px 0px !important;
        font-weight: 900;
        font-size: 20px;
        width: 100%;
        border: none !important;
        margin-top: 20px !important;
    }

    /* 6. Result Label (White Box, Black Text) */
    .result-label {
        background-color: white !important;
        padding: 18px;
        border-radius: 10px;
        border: 2px solid #0E2416;
        color: #000000 !important; /* Forces Black text */
        font-weight: 900;
        text-align: left;
        margin-top: 20px;
        font-size: 22px;
    }

    /* 7. Measures Box (Pure Black Text Visibility) */
    .measures-box {
        background-color: #ffffff !important;
        border-left: 10px solid #0E2416;
        padding: 25px;
        margin-top: 20px;
        border-radius: 8px;
        text-align: left;
        box-shadow: 4px 4px 15px rgba(0,0,0,0.1);
    }

    /* Force all text in recommendation box to be DARK */
    .measures-box h3 {
        color: #0E2416 !important;
        font-weight: 900 !important;
        margin-bottom: 10px !important;
    }

    .measures-box p, .measures-box li {
        color: #000000 !important;
        font-weight: 800 !important;
        font-size: 18px !important;
        line-height: 1.5;
    }

    /* Status Messages (Diseased/Healthy) */
    .status-alert {
        font-size: 24px;
        font-weight: 900;
        margin-top: 15px;
        padding: 10px;
        border-radius: 8px;
    }

    header, footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- LOAD RESOURCES ---
@st.cache_resource
def load_all():
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    if not os.path.exists(CLASS_INDICES_PATH):
        urllib.request.urlretrieve(CLASS_INDICES_URL, CLASS_INDICES_PATH)
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_INDICES_PATH, "r") as f:
        indices = json.load(f)
        class_names = [indices[str(i)] for i in range(len(indices))]
    return model, class_names

model, class_names = load_all()

# --- SOLUTIONS ---
solutions = {
    "Potato___Late_blight": "1. Use certified disease-free seeds.\n2. Apply protective fungicides like Mancozeb.\n3. Improve air circulation by spacing plants properly.",
    "Tomato___Early_blight": "1. Rotate crops annually.\n2. Apply copper-based fungicides.\n3. Remove and destroy infected lower leaves.",
    "Apple___Black_rot": "1. Prune dead wood and cankers.\n2. Remove all fallen fruit.\n3. Apply appropriate fungicides during early growth.",
}

# --- UI START ---
st.markdown('<p class="main-title">PLANT DISEASE DETECTOR</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Please upload the image of the plant leaf for analysis</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, use_container_width=True)

    # ANALYZE BUTTON - positioned directly under the image
    if st.button("ANALYZE"):
        # Prediction Process
        img_prep = image.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img_prep) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        preds = model.predict(img_array)
        result = class_names[np.argmax(preds)]

        # 1. Result Label
        st.markdown(f'<div class="result-label">Result = {result}</div>', unsafe_allow_html=True)

        # 2. Condition: Disease or Healthy
        if "healthy" not in result.lower():
            # Show "Diseased" Status
            st.markdown('<div class="status-alert" style="color: #B22222; background-color: #FFDADA;">❌ This crop is Diseased</div>', unsafe_allow_html=True)
            
            # Show Measures
            measure_content = solutions.get(result, "General: Apply appropriate fungicide and isolate infected plants.")
            st.markdown(f"""
                <div class="measures-box">
                    <h3>📋 Recommended Measures:</h3>
                    <p>{measure_content}</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            # Show "Healthy" Status
            st.markdown('<div class="status-alert" style="color: #006400; background-color: #D4EDDA;">✅ This crop is Healthy</div>', unsafe_allow_html=True)
            st.balloons()
