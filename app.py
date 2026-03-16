import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import urllib.request
import json

# --- 1. CONFIG & MODEL SETUP ---
MODEL_URL = "https://github.com/anishabedajna/crop_disease_detection/releases/download/v1/plant_disease_model.h5"
MODEL_PATH = "plant_disease_model.h5"

st.set_page_config(page_title="CropCare AI | Disease Detector", page_icon="🌿", layout="wide")

# --- 2. CACHED MODEL LOADING ---
@st.cache_resource
def load_trained_model():
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1000000:
        with st.spinner("🚀 Initializing AI Engine (547MB)... Please wait 3-5 minutes."):
            try:
                opener = urllib.request.build_opener()
                opener.addheaders = [('User-agent', 'Mozilla/5.0')]
                urllib.request.install_opener(opener)
                urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
                st.success("✅ Engine Loaded!")
            except Exception as e:
                st.error(f"Error loading model: {e}")
                return None
    return tf.keras.models.load_model(MODEL_PATH)

# --- 3. PROFESSIONAL CSS STYLING ---
st.markdown("""
    <style>
    /* Main Background */
    .stApp { background-color: #f8fafc; }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] { background-color: #1E5128; border-right: 2px solid #4E9F3D; }
    [data-testid="stSidebar"] * { color: white !important; font-weight: 500; }
    
    /* Title Section */
    .main-header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #1E5128, #4E9F3D);
        color: white;
        border-radius: 15px;
        margin-bottom: 30px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Card Styling for Measures */
    .info-card {
        background: white;
        padding: 25px;
        border-radius: 15px;
        border-left: 8px solid #4E9F3D;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    
    /* Button Styling */
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3.5em;
        background-color: #1E5128;
        color: white;
        font-weight: bold;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #4E9F3D;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 4. SIDEBAR NAVIGATION ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/628/628283.png", width=80)
    st.title("CropCare AI")
    st.markdown("---")
    page = st.radio("Navigation", ["🏠 Home", "🔍 Disease Detection", "🛡️ Prevention Measures"])
    st.markdown("---")
    st.info("This AI tool helps farmers identify 38 types of plant diseases instantly.")

# --- PAGE 1: HOME ---
if page == "🏠 Home":
    st.markdown("""
        <div class="main-header">
            <h1>🌿 Crop Disease Detection System</h1>
            <p>A Major Project in Machine Learning & Agriculture</p>
        </div>
        """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.write("### Project Overview")
        st.write("""
        Agriculture is the backbone of the economy, yet billions are lost annually due to crop diseases. 
        This project utilizes a **Deep Convolutional Neural Network (CNN)** to provide 
        automated, high-speed diagnosis of leaf infections.
        
        **Key Technologies:**
        * TensorFlow / Keras
        * MobileNetV2 Architecture
        * Streamlit Cloud Hosting
        """)
        
    with col2:
        # A nice placeholder image for the home page
        st.image("https://images.unsplash.com/photo-1597113366853-9a93ad3fed2e?auto=format&fit=crop&w=800&q=80", 
                 caption="AI in Agriculture", use_container_width=True)

    st.write("---")
    st.write("### Supported Diseases Table")
    st.markdown("""
    | Crop | Supported Classes |
    | :--- | :--- |
    | **Apple** | Scab, Black rot, Cedar rust, Healthy |
    | **Tomato** | Bacterial spot, Late blight, Leaf Mold, Mosaic virus, Healthy |
    | **Potato** | Early blight, Late blight, Healthy |
    | **Corn** | Common rust, Northern Leaf Blight, Healthy |
    """)

# --- PAGE 2: DETECTION ---
elif page == "🔍 Disease Detection":
    st.markdown('<h2 style="color: #1E5128;">🔍 AI Diagnostic Tool</h2>', unsafe_allow_html=True)
    
    # Load Model & Classes
    model = load_trained_model()
    try:
        with open('class_indices.json', 'r') as f:
            class_indices = json.load(f)
        class_names = list(class_indices.values())
    except:
        st.error("Error: 'class_indices.json' not found on GitHub.")
        class_names = []

    st.write("Upload a clear photo of the plant leaf for accurate analysis.")
    uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', width=450)
        
        if st.button("RUN AUTOMATED ANALYSIS"):
            if model and class_names:
                with st.spinner("Analyzing patterns..."):
                    # Preprocess
                    img = image.resize((224, 224)) 
                    img_array = tf.keras.preprocessing.image.img_to_array(img)
                    img_array = np.expand_dims(img_array, axis=0) / 255.0

                    # Predict
                    predictions = model.predict(img_array)
                    result = class_names[np.argmax(predictions)]
                    confidence = np.max(predictions) * 100
                    
                    # Result Display
                    st.markdown(f"""
                        <div class="info-card">
                            <h2 style='color:#1E5128;'>Diagnosis: {result}</h2>
                            <p>Confidence Level: <b>{confidence:.2f}%</b></p>
                        </div>
                    """, unsafe_allow_html=True)
                    st.balloons()
            else:
                st.error("The system is still initializing. Please wait a moment.")

# --- PAGE 3: MEASURES ---
elif page == "🛡️ Prevention Measures":
    st.markdown('<h2 style="color: #1E5128;">🛡️ Disease Management & Measures</h2>', unsafe_allow_html=True)
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("""
            <div class="info-card">
                <h3>🌱 Prevention (Cultural)</h3>
                <ul>
                    <li><b>Crop Rotation:</b> Don't plant the same crop in the same spot twice.</li>
                    <li><b>Sanitation:</b> Clean all farm tools before moving between fields.</li>
                    <li><b>Resistant Varieties:</b> Use seeds certified to resist local pathogens.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
    with col_b:
        st.markdown("""
            <div class="info-card">
                <h3>🧪 Treatment (Chemical/Organic)</h3>
                <ul>
                    <li><b>Fungicides:</b> Apply copper-based sprays for fungal blights.</li>
                    <li><b>Neem Oil:</b> A powerful organic alternative for soft-bodied pests.</li>
                    <li><b>Soil Health:</b> Ensure proper Nitrogen-Phosphorus-Potassium balance.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>CropCare AI Major Project © 2026</p>", unsafe_allow_html=True)
