import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import urllib.request
import json

# --- 1. CONFIG & MODEL LINK ---
# PASTE YOUR LINK BELOW
MODEL_URL = "https://release-assets.githubusercontent.com/github-production-release-asset/1183591133/f1a78a68-4140-4cf0-8e8c-bc74da1572ce?sp=r&sv=2018-11-09&sr=b&spr=https&se=2026-03-16T20%3A22%3A34Z&rscd=attachment%3B+filename%3Dplant_disease_model.h5&rsct=application%2Foctet-stream&skoid=96c2d410-5711-43a1-aedd-ab1947aa7ab0&sktid=398a6654-997b-47e9-b12b-9515b896b4de&skt=2026-03-16T19%3A21%3A41Z&ske=2026-03-16T20%3A22%3A34Z&sks=b&skv=2018-11-09&sig=V4VIO2%2BHIA7dpfIRaaPvygjHIgAz7Q9MfTWFP3tvP4k%3D&jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmVsZWFzZS1hc3NldHMuZ2l0aHVidXNlcmNvbnRlbnQuY29tIiwia2V5Ijoia2V5MSIsImV4cCI6MTc3MzY5MzM0OCwibmJmIjoxNzczNjg5NzQ4LCJwYXRoIjoicmVsZWFzZWFzc2V0cHJvZHVjdGlvbi5ibG9iLmNvcmUud2luZG93cy5uZXQifQ._zHWLIUVwJhuWhKWz8PkcxVZj78-bVO6dvdzad2XkOk&response-content-disposition=attachment%3B%20filename%3Dplant_disease_model.h5&response-content-type=application%2Foctet-stream" 
MODEL_PATH = "plant_disease_model.h5"

st.set_page_config(page_title="Crop Disease Detection", page_icon="🌾", layout="centered")

# --- 2. PROFESSIONAL CSS STYLING ---
st.markdown("""
    <style>
    .stApp { background-color: #fcfdfc; }
    h1 { 
        color: #1E5128; 
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        text-align: center;
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

# --- 3. BACKGROUND LOGIC ---
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📥 Downloading AI Brain... This may take a moment due to file size."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return tf.keras.models.load_model(MODEL_PATH)

# Load data files
try:
    model = load_model()
    with open('class_indices.json', 'r') as f:
        class_indices = json.load(f)
    class_names = list(class_indices.values())
except Exception as e:
    st.error(f"System Error: {e}")

# --- 4. USER INTERFACE ---
st.title("A Machine Learning Approach for Crop Disease Detection and Management")

st.write("") # Spacer
st.markdown("""
**Empowering farmers with instant AI diagnosis.** Manual detection is slow and often inaccurate. Use this tool to get a precise diagnosis and take immediate action to protect your yield.
""")

uploaded_file = st.file_uploader("📸 Upload or drag a leaf photo here", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Sample', use_container_width=True)
    
    if st.button("🔍 START AUTOMATED ANALYSIS"):
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

st.markdown("---")
st.caption("A Machine Learning Approach for Crop Disease Management © 2026")
