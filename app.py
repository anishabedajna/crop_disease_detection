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

st.set_page_config(page_title="Crop Disease Detection", page_icon="🌾", layout="centered")

# --- 2. PROFESSIONAL CSS STYLING (Integrated) ---
st.markdown("""
    <style>
    .stApp { background-color: #fcfdfc; }
    .main-title { 
        color: #1E5128; 
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        text-align: center;
        font-size: 35px;
        font-weight: bold;
        border-bottom: 2px solid #4E9F3D;
        padding-bottom: 10px;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3.5em;
        background-color: #1E5128;
        color: white;
        font-size: 18px;
        font-weight: bold;
        border: none;
        box-shadow: 0px 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button:hover {
        background-color: #4E9F3D;
        color: white;
        border: none;
    }
    .report-box {
        padding: 25px;
        border-radius: 12px;
        background-color: #ffffff;
        border-left: 10px solid #4E9F3D;
        box-shadow: 0px 8px 16px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. BACKGROUND LOADING LOGIC ---
@st.cache_resource
def load_trained_model():
    # Check if the file is missing or is an empty LFS pointer (<1MB)
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1000000:
        with st.spinner("📥 Downloading AI Model (547MB)... This takes 3-5 minutes. Please wait."):
            try:
                # Add headers to ensure GitHub accepts the request
                opener = urllib.request.build_opener()
                opener.addheaders = [('User-agent', 'Mozilla/5.0')]
                urllib.request.install_opener(opener)
                urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
                st.success("✅ Download complete! Loading model into memory...")
            except Exception as e:
                st.error(f"Download failed: {e}")
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
st.markdown('<p class="main-title">A Machine Learning Approach for Crop Disease Detection and Management</p>', unsafe_allow_html=True)

st.write("") 
st.markdown("""
**Empowering farmers with instant AI diagnosis.** Manual detection is slow and often inaccurate. Use this tool to get a precise diagnosis and take immediate action to protect your yield.
""")

uploaded_file = st.file_uploader("📸 Upload or drag a leaf photo here", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Sample', use_container_width=True)
    
    if st.button("🔍 START AUTOMATED ANALYSIS"):
        if model is not None and len(class_names) > 0:
            with st.spinner("Processing leaf patterns..."):
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
                        <h2 style='margin-top:0;'>Diagnosis: <span style='color:#4E9F3D;'>{result}</span></h2>
                        <p><b>Analysis Accuracy:</b> {confidence:.2f}%</p>
                        <hr>
                        <h4 style='color:#1E5128;'>Recommended Management Steps:</h4>
                        <ul>
                            <li><b>Isolate:</b> Prevent the spread of <b>{result}</b> by separating affected crops.</li>
                            <li><b>Monitoring:</b> Check surrounding plants for similar early symptoms.</li>
                            <li><b>Treatment:</b> Apply appropriate organic or chemical controls specifically for {result}.</li>
                            <li><b>Sanitation:</b> Clean all farming tools used on the infected plant.</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)
                st.balloons()
        else:
            st.error("System Error: Model could not be loaded. Check your logs.")

st.markdown("---")
st.caption("A Machine Learning Approach for Crop Disease Management © 2026")
