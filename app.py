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

st.set_page_config(page_title="Plant Disease Detection App", layout="wide")

# --- EXACT CSS FROM SCREENSHOT ---
st.markdown("""
<style>
/* Perfect background match with leaf patterns */
.stApp {
    background: linear-gradient(135deg, #e8f5e8 0%, #f0f8f0 50%, #d4edda 100%);
    background-image: 
        radial-gradient(ellipse 30% 20% at 20% 80%, rgba(76,175,80,0.15) 0%, transparent 50%),
        radial-gradient(ellipse 25% 15% at 80% 20%, rgba(0,128,0,0.1) 0%, transparent 50%),
        radial-gradient(ellipse 20% 10% at 40% 60%, rgba(46,125,50,0.08) 0%, transparent 50%);
}

/* Main header - EXACT match */
.header {
    font-size: 3.2rem !important;
    font-weight: 900 !important;
    background: linear-gradient(45deg, #2d5a2d, #4a7c4a);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    margin: 2rem 0 0.5rem 0 !important;
    font-family: 'Arial Black', sans-serif !important;
    text-shadow: 3px 3px 6px rgba(0,0,0,0.2);
}

/* Subheader - EXACT match */
.subheader {
    font-size: 1.35rem !important;
    color: #4a7c4a !important;
    text-align: center !important;
    margin: 0 0 3rem 0 !important;
    font-weight: 500 !important;
    font-style: italic;
}

/* Upload area - PERFECT dashed border match */
.upload-area {
    background: rgba(255,255,255,0.92);
    border: 4px dashed #4a7c4a;
    border-radius: 20px;
    padding: 3.5rem 2rem !important;
    text-align: center;
    margin: 0 auto 2rem auto;
    max-width: 650px;
    box-shadow: 0 12px 30px rgba(0,0,0,0.15);
    backdrop-filter: blur(10px);
    transition: all 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94);
}

.upload-area:hover {
    border-color: #2d5a2d;
    background: rgba(255,255,255,0.98);
    transform: translateY(-5px);
    box-shadow: 0 20px 40px rgba(0,0,0,0.2);
}

/* Drag text styling */
.upload-text {
    color: #2d5a2d !important;
    font-size: 1.25rem !important;
    font-weight: 600 !important;
    margin: 1.5rem 0 2rem 0 !important;
    text-shadow: 1px 1px 2px rgba(255,255,255,0.8);
}

/* Camera icon - EXACT size/position */
.camera-icon {
    font-size: 5rem !important;
    color: #4a7c4a !important;
    margin-bottom: 1.5rem !important;
    filter: drop-shadow(0 4px 8px rgba(74,124,74,0.3));
}

/* Browse button styling */
.browse-btn {
    background: linear-gradient(45deg, #4a7c4a, #6ba86b) !important;
    color: white !important;
    border: none !important;
    border-radius: 30px !important;
    padding: 1rem 3rem !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    box-shadow: 0 6px 20px rgba(74,124,74,0.4) !important;
    transition: all 0.3s ease !important;
    width: 100% !important;
    max-width: 250px;
}

.browse-btn:hover {
    background: linear-gradient(45deg, #2d5a2d, #4a7c4a) !important;
    transform: translateY(-3px) !important;
    box-shadow: 0 10px 25px rgba(45,90,45,0.5) !important;
}

/* Footer text - EXACT match */
.footer-text {
    color: #5a8a5a !important;
    font-size: 1.1rem !important;
    margin-top: 2rem !important;
    font-style: italic !important;
    font-weight: 500 !important;
}

/* Hide Streamlit uploader default styling */
[data-testid="stFileUploader"] div[role="button"] {
    background: transparent !important;
    border: none !important;
}

/* Center everything */
.block-container {
    padding-top: 2rem !important;
}
</style>
""", unsafe_allow_html=True)

# --- CACHED MODEL LOADING (YOUR ORIGINAL CODE) ---
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
    st.stop()

# --- YOUR DISEASE SOLUTIONS (UNCHANGED) ---
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

# --- EXACT HEADER FROM SCREENSHOT ---
st.markdown('<div class="header">Plant Disease Detection App</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Upload one or more leaf images to detect diseases</div>', unsafe_allow_html=True)

# --- PERFECT UPLOAD AREA FROM SCREENSHOT ---
col1, col2, col3 = st.columns([2, 1, 2])
with col2:
    st.markdown("""
    <div class="upload-area">
        <div class="camera-icon">📷</div>
        <div class="upload-text">Drag & drop images here</div>
        <div class="footer-text">or click Browse</div>
    </div>
    """, unsafe_allow_html=True)

# --- FUNCTIONAL FILE UPLOADER (HIDDEN BUT WORKS) ---
uploaded_files = st.file_uploader(
    "Choose leaf images...", 
    type=['jpg', 'jpeg', 'png'],
    accept_multiple_files=True,
    key="hidden_uploader",
    help="Upload leaf images for disease detection"
)

# --- YOUR EXISTING LOGIC (IMPROVED LAYOUT) ---
if uploaded_files:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_container_width=True)
        
        if st.button("🔍 ANALYZE", key=f"analyze_{uploaded_file.name}"):
            with st.spinner('Analyzing...'):
                # YOUR MODEL PREDICTION CODE
                img_for_model = image.resize((224, 224))
                img_array = tf.keras.preprocessing.image.img_to_array(img_for_model) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                preds = model.predict(img_array)
                result_index = np.argmax(preds)
                predicted_class = class_names[result_index]
                
                # Results with your styling
                st.markdown(f"""
                <div style='background: white; padding: 20px; border-radius: 10px; 
                           border-left: 5px solid #4a7c4a; margin: 20px 0;'>
                    <h3 style='color: #2d5a2d; margin: 0;'>Result: {predicted_class}</h3>
                </div>
                """, unsafe_allow_html=True)
                
                if "healthy" not in predicted_class.lower():
                    st.error("⚠️ This crop is Diseased")
                    measures_list = solutions.get(predicted_class, ["Consult an agricultural expert."])
                    
                    st.markdown(f"""
                    <div style='background: #f8f9fa; border-left: 5px solid #2d5a2d; 
                               padding: 20px; border-radius: 8px; margin: 20px 0;'>
                        <h4 style='color: #2d5a2d; margin-top: 0;'>📋 Recommended Measures:</h4>
                        <ul style='color: #333; font-weight: 500;'>
                    """, unsafe_allow_html=True)
                    
                    for measure in measures_list:
                        st.markdown(f"<li style='margin: 8px 0;'>{measure}</li>", unsafe_allow_html=True)
                    
                    st.markdown("</ul></div>", unsafe_allow_html=True)
                else:
                    st.success("✅ This crop appears to be Healthy!")
                    st.balloons()
