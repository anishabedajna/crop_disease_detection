import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import urllib.request
import json
import time

# --- CONFIG ---
MODEL_URL = "https://github.com/anishabedajna/crop_disease_detection/releases/download/v1/plant_disease_model.h5"
CLASS_INDICES_URL = "https://github.com/anishabedajna/crop_disease_detection/releases/download/v1/class_indices.json"
MODEL_PATH = "plant_disease_model.h5"
CLASS_INDICES_PATH = "class_indices.json"

# Setting layout="wide" helps with full-page designs, although we mainly use CSS.
st.set_page_config(page_title="Plant Disease Detector", layout="centered")

# --- CUSTOM CSS (Enforces a full-page, high-contrast look) ---
st.markdown("""
<style>
    /* 1. Force the leaf image to be the full-page background with proper overlay */
    .stApp {
        background: linear-gradient(rgba(220,255,220,0.85), rgba(220,255,220,0.85)), 
                    url("https://images.unsplash.com/photo-1599385552300-85f2fa6b3068?q=80");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }

    /* 2. Remove the Card Look - Make the main container transparent and remove padding */
    .block-container {
        max-width: 700px !important;
        margin: auto;
        padding: 1rem !important; /* Minimum padding for elements */
        background-color: transparent !important; /* Removes the grey box */
        box-shadow: none !important; /* Removes the shadow from the card */
        text-align: center; /* Center-aligns all standard content */
    }

    /* 3. Main Title Styling - Bold, Dark Green, Underlined, Centered */
    .main-title {
        color: #1B3022 !important;
        font-family: 'Arial Black', sans-serif;
        font-size: 3rem !important;
        text-transform: uppercase;
        margin-bottom: 5px !important;
        text-decoration: underline;
    }

    .sub-title {
        color: #1B3022 !important;
        font-size: 1.2rem;
        margin-bottom: 40px;
        font-weight: 700;
    }

    /* 4. Smallered file uploader for centered aesthetic */
    [data-testid="stFileUploader"] {
        width: 85% !important;
        margin: 0 auto !important;
    }

    /* 5. Custom CSS for dark green full-width 'ANALYZE' button */
    div.stButton > button:first-child {
        background-color: #1B3022 !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 10px 40px !important;
        font-size: 20px !important;
        font-weight: bold;
        border: none !important;
        width: 100%; /* Makes it full-width within the element stack */
        margin-top: 10px;
    }
    
    /* Button Hover effect */
    div.stButton > button:first-child:hover {
        background-color: #2D4F38 !important; /* Slightly lighter on hover */
    }

    /* 6. Result Label Styling (White box) */
    .result-label {
        background-color: white !important;
        padding: 15px;
        border-radius: 5px;
        border: 1px solid #ccc;
        color: #000000 !important; /* Pure Black text for contrast */
        font-weight: bold;
        text-align: left;
        margin-top: 20px;
        font-size: 1.2rem;
    }
    
    /* 7. Recommendations Box Styling with dark text colors */
    .recommendations-box {
        background-color: #f1f3f5 !important;
        border-left: 5px solid #1B3022;
        padding: 20px;
        text-align: left;
        margin-top: 20px;
        border-radius: 4px;
        color: #333333 !important; /* Specific dark text color */
    }
    
    /* Direct bullet points color styling */
    .recommendations-box li {
        color: #333333 !important; 
        font-weight: bold;
    }

    /* Custom CSS to style st.error for contrast */
    .stError {
        color: #8C2222 !important;
        background-color: rgba(255,200,200,0.9) !important;
        text-align: center;
        font-weight: bold;
    }

    /* Hide standard Streamlit header and footer branding */
    header, footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- CACHED RESOURCES LOADING ---
@st.cache_resource
def load_and_cache_model():
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_and_cache_class_names():
    if not os.path.exists(CLASS_INDICES_PATH):
        urllib.request.urlretrieve(CLASS_INDICES_URL, CLASS_INDICES_PATH)
    with open(CLASS_INDICES_PATH, "r") as f:
        data = json.load(f)
        return [data[str(i)] for i in range(len(data))]

try:
    model = load_and_cache_model()
    class_names = load_and_cache_class_names()
except Exception as e:
    st.error(f"Error loading model resources: {e}")
    class_names = []

# --- DISEASE MEASURES DICTIONARY ---
# We keep this focused on data and dynamic HTML styling.
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
# These specific markdown strings and CSS classes recreate the centered header look.
st.markdown('<h1 class="main-title">PLANT DISEASE DETECTOR</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Please upload the image of the plant leaf for the analysis</p>', unsafe_allow_html=True)

# --- FILE UPLOADER & IMAGE HANDLING ---
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # We display the image and ensure it's centered
    st.image(image, use_container_width=True)

    # --- ANALYZE BUTTON & CONDITIONAL FUNCTIONALITY ---
    if st.button("ANALYZE"):
        # Check if model/classes loaded successfully
        if not class_names or model is None:
            st.error("Cannot proceed. Model resources are missing.")
        else:
            with st.spinner('Performing analysis...'):
                # 1. Prepare Image for Prediction
                # TensorFlow usually normalizes pixel values [0,1].
                img_for_model = image.resize((224, 224))
                img_array = tf.keras.preprocessing.image.img_to_array(img_for_model) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                # 2. Get Prediction
                preds = model.predict(img_array)
                result_index = np.argmax(preds)
                predicted_class = class_names[result_index]

                # 3. Create Custom Result Label (Matches image, left-aligned in white box)
                st.markdown(f'<div class="result-label">Result = {predicted_class}</div>', unsafe_allow_html=True)

                # 4. Condition Check: Show detailed status message
                if "healthy" not in predicted_class.lower():
                    # st.error's styling is overridden by custom CSS for better contrast.
                    st.error("⚠️ This crop is Diseased")
                    
                    # 5. Show Recommended Measures for treatment
                    measures_list = solutions.get(predicted_class, ["Consult an agricultural expert."])
                    
                    # We start the custom-styled recommendations box
                    st.markdown("""
                        <div class="recommendations-box">
                            <h3 style="margin-top:0; color:#1B3022;">📋 Recommended Measures:</h3>
                            <ul>
                    """, unsafe_allow_html=True)
                    
                    for measure in measures_list:
                        st.markdown(f"<li>{measure}</li>", unsafe_allow_html=True)
                        
                    # We properly close the recommendations box
                    st.markdown("""
                            </ul>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.success("✅ This crop appears to be Healthy!")
                    st.balloons()
