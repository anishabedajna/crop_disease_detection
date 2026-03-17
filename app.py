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

# --- CUSTOM CSS ---
st.markdown("""
<style>
    /* 1. Full page background with leaf image */
    .stApp {
        background: linear-gradient(rgba(255,255,255,0.4), rgba(255,255,255,0.4)), 
                    url("https://images.unsplash.com/photo-1599385552300-85f2fa6b3068?q=80");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }

    /* 2. Main Container Card */
    .block-container {
        max-width: 700px !important;
        margin: auto;
        padding: 50px !important;
        background-color: rgba(235, 235, 235, 0.98);
        border-radius: 30px;
        box-shadow: 0px 15px 40px rgba(0,0,0,0.4);
        text-align: center;
    }

    /* 3. Underlined Centered Title */
    .main-title {
        color: #0E2416 !important;
        font-family: 'Arial Black', sans-serif;
        font-size: 45px !important;
        text-decoration: underline;
        font-weight: 900;
        margin-bottom: 5px !important;
    }

    .sub-title {
        color: #0E2416 !important;
        font-size: 1.1rem;
        margin-bottom: 30px;
        font-weight: 600;
    }

    /* 4. Small centered file uploader */
    [data-testid="stFileUploader"] {
        width: 80% !important;
        margin: 0 auto !important;
    }

    /* 5. Analyze Button (Dark Green) */
    div.stButton > button {
        background-color: #0E2416 !important;
        color: white !important;
        border-radius: 5px !important;
        padding: 12px 0px !important;
        font-weight: bold;
        font-size: 20px;
        width: 100%;
        border: none !important;
        margin-top: 25px !important;
    }

    /* 6. Result Label (White Box, Black Text) */
    .result-label {
        background-color: white !important;
        padding: 15px;
        border-radius: 8px;
        border: 2px solid #0E2416;
        color: #000000 !important;
        font-weight: 900;
        text-align: left;
        margin-top: 15px;
        font-size: 22px;
    }

    /* 7. Measures Box (Strict Dark Text) */
    .measures-box {
        background-color: #ffffff !important;
        border-left: 10px solid #0E2416;
        padding: 25px;
        margin-top: 20px;
        border-radius: 5px;
        text-align: left;
        box-shadow: 3px 3px 15px rgba(0,0,0,0.1);
    }

    /* Force all text inside measures box to be BLACK */
    .measures-box h3 {
        color: #0E2416 !important;
        font-weight: 900 !important;
        margin-bottom: 15px !important;
    }

    .measures-box p, .measures-box li {
        color: #000000 !important;
        font-weight: 700 !important;
        font-size: 18px !important;
    }

    /* 8. Hide Streamlit clutter */
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

# --- SOLUTIONS DICTIONARY ---
solutions = {
    "Potato___Late_blight": "1. Use certified disease-free seeds. \n2. Apply fungicides like Mancozeb. \n3. Improve air circulation.",
    "Tomato___Early_blight": "1. Rotate crops annually. \n2. Apply copper-based fungicides. \n3. Remove infected lower leaves.",
    "Apple___Black_rot": "1. Prune out dead wood. \n2. Remove fallen fruit. \n3. Use appropriate fungicides.",
}

# --- UI START ---
st.markdown('<p class="main-title">PLANT DISEASE DETECTOR</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Please upload the image of the plant leaf for analysis</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, use_container_width=True)

    # ANALYZE BUTTON - sits directly below image
    if st.button("ANALYZE"):
        # Processing
        img = image.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        preds = model.predict(img_array)
        result = class_names[np.argmax(preds)]

        # 1. Result Display
        st.markdown(f'<div class="result-label">Result = {result}</div>', unsafe_allow_html=True)

        # 2. Logic Check
        if "healthy" not in result.lower():
            # Show "Diseased" Status in dark red bold text
            st.markdown('<p style="color:#B22222; font-size:24px; font-weight:900; margin-top:15px;">❌ This crop is Diseased</p>', unsafe_allow_html=True)
            
            # Show Measures in the Black Text Box
            measure_text = solutions.get(result, "Apply general fungicide and remove infected plant parts immediately.")
            st.markdown(f"""
                <div class="measures-box">
                    <h3>📋 Recommended Measures:</h3>
                    <p>{measure_text}</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            # Show "Healthy" Status in dark green bold text
            st.markdown('<p style="color:#006400; font-size:24px; font-weight:900; margin-top:15px;">✅ This crop is Healthy</p>', unsafe_allow_html=True)
            st.balloons()
