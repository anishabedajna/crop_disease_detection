import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import urllib.request
import json

# --- 1. CONFIG & PERMANENT RELEASE LINK ---
# Direct download link from your GitHub Release v1
MODEL_URL = "https://github.com/anishabedajna/crop_disease_detection/releases/download/v1/plant_disease_model.h5"
MODEL_PATH = "plant_disease_model.h5"

# Page Config (this sets the browser tab title and icon)
st.set_page_config(page_title="Crop Disease Detector", page_icon="🌿", layout="centered")

# --- 2. PROFESSIONAL CSS STYLING (Integrated) ---
st.markdown("""
    <style>
    .stApp { background-color: #f8fafc; } /* Lighter, clean background */
    
    /* Title Styling */
    .title-container {
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 30px;
        padding-bottom: 15px;
        border-bottom: 3px solid #4E9F3D; /* Green accent line */
    }
    .main-title { 
        color: #1E5128; 
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 42px;
        font-weight: 800;
        margin-left: 15px; /* Space after logo */
    }
    
    /* Button Styling */
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3.8em;
        background-color: #1E5128; /* Dark green */
        color: white;
        font-size: 19px;
        font-weight: bold;
        border: none;
        box-shadow: 0px 4px 10px rgba(30, 81, 40, 0.2);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #4E9F3D; /* Lighter green on hover */
        box-shadow: 0px 6px 15px rgba(78, 159, 61, 0.3);
        border: none;
        color: white;
    }
    
    /* Result Box Styling */
    .report-box {
        padding: 30px;
        border-radius: 15px;
        background-color: #ffffff;
        border-left: 12px solid #4E9F3D;
        box-shadow: 0px 10px 25px rgba(0,0,0,0.08);
        margin-top: 20px;
    }
    .diagnosis-text {
        font-size: 28px;
        font-weight: 700;
        color: #1E5128;
        margin-bottom: 10px;
    }
    .confidence-text {
        font-size: 18px;
        color: #64748b;
        margin-bottom: 20px;
    }
    .management-steps {
        font-size: 16px;
        color: #334155;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. BACKGROUND LOADING LOGIC ---
@st.cache_resource
def load_trained_model():
    # Check if the file is missing or is an empty LFS pointer (<1MB)
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1000000:
        with st.spinner("📥 Initializing AI Brain (547MB Download)... Please wait 3-5 minutes."):
            try:
                # Add headers to ensure GitHub accepts the request
                opener = urllib.request.build_opener()
                opener.addheaders = [('User-agent', 'Mozilla/5.0')]
                urllib.request.install_opener(opener)
                urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
                st.success("✅ Download complete! Loading model...")
            except Exception as e:
                st.error(f"Initialization failed: {e}")
                return None
    return tf.keras.models.load_model(MODEL_PATH)

# Load the model
model = load_trained_model()

# Load class names from JSON
try:
    with open('class_indices.json', 'r') as f:
        class_indices = json.load(f)
    class_names = list(class_indices.values())
except Exception as e:
    st.error("Error: 'class_indices.json' not found. Please upload it to GitHub.")
    class_names = []

# --- 4. USER INTERFACE ---

# Title with Leaf Logo
st.markdown("""
    <div class="title-container">
        <span style="font-size: 48px;">🌿</span>
        <h1 class="main-title">Crop Disease Detector</h1>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; color: #64748b; margin-bottom: 30px;">
    <strong>Protect your yield with instant AI diagnosis.</strong><br>
    Upload a clear photo of an affected plant leaf below for analysis.
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📸 Choose a leaf photo...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Sample', use_container_width=True)
    
    # --- AUTOMATED ANALYSIS BUTTON ---
    if st.button("🔍 START AUTOMATED ANALYSIS"):
        if model is not None and len(class_names) > 0:
            with st.spinner("Analyzing leaf patterns..."):
                # 1. Image Preprocessing
                img = image.resize((224, 224)) 
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array /= 255.0

                # 2. Prediction
                predictions = model.predict(img_array)
                predicted_class = np.argmax(predictions)
                confidence = np.max(predictions) * 100
                result = class_names[predicted_class]
                
                # 3. Results Display
                st.markdown("---")
                st.markdown(f"""
                    <div class="report-box">
                        <div class="diagnosis-text">Diagnosis: <span style='color:#4E9F3D;'>{result}</span></div>
                        <div class="confidence-text">Analysis Confidence: <strong>{confidence:.2f}%</strong></div>
                        <hr style="border-top: 1px solid #e2e8f0;">
                        <h4 style='color:#1E5128; margin-top: 15px;'>Recommended Management Steps:</h4>
                        <ul class="management-steps">
                            <li><strong>Isolate:</strong> Immediately separate infected crops to prevent further spread.</li>
                            <li><strong>Monitor:</strong> Inspect surrounding plants daily for early symptoms.</li>
                            <li><strong>Treatment:</strong> Apply appropriate organic or chemical controls validated for <em>{result}</em>.</li>
                            <li><strong>Sanitation:</strong> Thoroughly clean and disinfect all tools after use on the infected plant.</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)
                st.balloons() # Flying balloons for success!
        else:
            st.error("System Error: The AI model is still loading or a critical file is missing. Please check your logs.")

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #94a3b8; font-size: 14px; padding: 10px;">
    A Machine Learning Approach for Crop Disease Management © 2026<br>
    Built with Streamlit and TensorFlow
</div>
""", unsafe_allow_html=True)
