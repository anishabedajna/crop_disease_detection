import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import urllib.request
import json
import time

# --- CONFIG ---
# Replace with your actual model and class indices URLs
MODEL_URL = "https://github.com/anishabedajna/crop_disease_detection/releases/download/v1/plant_disease_model.h5"
CLASS_INDICES_URL = "https://github.com/anishabedajna/crop_disease_detection/releases/download/v1/class_indices.json"
MODEL_PATH = "plant_disease_model.h5"
CLASS_INDICES_PATH = "class_indices.json"

st.set_page_config(page_title="Plant Disease Detector", layout="centered")

# --- CUSTOM CSS (This matches your image exactly) ---
# We use this to center the elements, style the button, and create the central 'card'.
st.markdown("""
<style>
    /* Background for the whole app - using the requested leaf image background */
    .stApp {
        background: linear-gradient(rgba(220,255,220,0.8), rgba(220,255,220,0.8)), 
                    url("https://images.unsplash.com/photo-1599385552300-85f2fa6b3068?q=80");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }

    /* Target the central container to make it look like a rounded 'card' */
    .block-container {
        max-width: 600px; /* limits the width for a card look */
        margin: auto;
        padding: 50px !important;
        background-color: #E6E6E6; /* Light grey card color from image */
        border-radius: 20px;
        box-shadow: 10px 10px 30px rgba(0,0,0,0.1);
        text-align: center; /* Center-aligns all standard content */
    }

    /* Main Title Styling - Bold, Dark Green, Underlined, Centered */
    .main-title {
        color: #1B3022 !important;
        font-family: 'Arial Black', sans-serif;
        font-size: 3rem !important;
        text-transform: uppercase;
        margin-bottom: 5px !important;
        text-decoration: underline; /* Add the underline */
    }

    .sub-title {
        color: #1B3022;
        font-size: 1.2rem;
        margin-bottom: 40px;
    }

    /* Custom CSS to target the st.button specifically and style it dark green */
    div.stButton > button:first-child {
        background-color: #1B3022 !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 10px 40px !important;
        font-size: 20px !important;
        font-weight: bold;
        border: none !important;
        width: 100%; /* Makes it full-width within the container */
        margin-top: 10px;
        cursor: pointer;
    }
    
    /* Button Hover effect */
    div.stButton > button:first-child:hover {
        background-color: #2D4F38 !important; /* Slightly lighter on hover */
    }

    /* Result Box Styling - a white box with left-aligned text */
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
    
    /* Recommendations Box Styling */
    .recommendations-box {
        background-color: #f1f3f5;
        border-left: 5px solid #1B3022;
        padding: 20px;
        text-align: left;
        margin-top: 20px;
        border-radius: 4px;
        color: #333;
    }

    /* Hide standard Streamlit elements */
    header, footer {visibility: hidden;}
    .css-1dp5a00 {visibility: hidden;} /* Hides the standard title if needed */
</style>
""", unsafe_allow_html=True)

# --- CACHED RESOURCES LOADING ---
@st.cache_resource
def load_and_cache_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading and loading model... this may take a moment."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_and_cache_class_names():
    if not os.path.exists(CLASS_INDICES_PATH):
        with st.spinner("Downloading class labels..."):
            urllib.request.urlretrieve(CLASS_INDICES_URL, CLASS_INDICES_PATH)
    with open(CLASS_INDICES_PATH, "r") as f:
        data = json.load(f)
        return [data[str(i)] for i in range(len(data))]

# Try-except block for loading model/classes (best practice for production)
try:
    model = load_and_cache_model()
    class_names = load_and_cache_class_names()
except Exception as e:
    st.error(f"Error loading model resources: {e}")
    class_names = []

# --- DISEASE MEASURES (DATA DICTIONARY) ---
# It's better to keep this data-focused and style it dynamically in the UI section
solutions = {
    "Tomato___Late_blight": [
        "Remove and destroy infected leaves immediately.",
        "Apply fungicides like Copper-based sprays or Mancozeb.",
        "Ensure good air circulation."
    ],
    "Tomato___Early_blight": [
        "Rotate crops. Use copper-based fungicides.",
        "Remove lower leaves to stop upward spread."
    ],
    "Potato___Late_blight": [
        "Use certified disease-free seeds.",
        "Apply protective fungicides before rainy periods.",
        "Improve air circulation."
    ],
    "Apple___Black_rot": [
        "Prune infected branches and cankers.",
        "Apply appropriate fungicides.",
        "Remove and destroy fallen fruits."
    ],
    "Corn___Common_rust": [
        "Use resistant varieties if available.",
        "Apply fungicide if necessary (though often not needed)."
    ],
}

# --- UI HEADER ---
# This markdown structure and the CSS classes ensure the exact header look.
st.markdown('<h1 class="main-title">PLANT DISEASE DETECTOR</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Please upload the image of the plant leaf for the analysis</p>', unsafe_allow_html=True)

# --- FILE UPLOADER & IMAGE HANDLING ---
# Streamlit's file_uploader doesn't fully support custom styling from standard CSS,
# but we wrap it to match the overall layout.
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # Display image at full width of our custom card container
    st.image(image, use_container_width=True)

    # --- ANALYZE BUTTON & FUNCTIONALITY ---
    if st.button("ANALYZE"):
        # Check if model/classes loaded successfully
        if not class_names or model is None:
            st.error("Cannot proceed. Model resources are missing.")
        else:
            with st.spinner('Performing analysis...'):
                # 1. Prepare Image for Prediction
                # (Assuming model expects (224, 224) and [0,1] normalization)
                img_for_model = image.resize((224, 224))
                img_array = tf.keras.preprocessing.image.img_to_array(img_for_model) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                # 2. Get Prediction
                preds = model.predict(img_array)
                result_index = np.argmax(preds)
                predicted_class = class_names[result_index]

                # 3. Create Custom Result Label (Matches image exactly)
                st.markdown(f'<div class="result-label">Result = {predicted_class}</div>', unsafe_allow_html=True)

                # 4. Conditional Check: If diseased, show measures in a conditional popup.
                if "healthy" not in predicted_class.lower():
                    st.error("This crop is Diseased")
                    
                    # 5. Show Recommended Measures
                    measures_list = solutions.get(predicted_class, ["Consult an agricultural expert."])
                    
                    st.markdown("""
                        <div class="recommendations-box">
                            <h3 style="margin-top:0; color:#1B3022;">📋 Recommended Measures:</h3>
                            <ul>
                    """, unsafe_allow_html=True)
                    
                    for measure in measures_list:
                        st.markdown(f"<li>{measure}</li>", unsafe_allow_html=True)
                        
                    st.markdown("""
                            </ul>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.success("This crop is Healthy")
                    # Add a nice celebratory effect for health
                    st.balloons()
