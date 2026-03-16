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

st.set_page_config(page_title="Crop Disease Detector", page_icon="🌿", layout="wide")

# --- 2. CACHED MODEL LOADING ---
@st.cache_resource
def load_trained_model():
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1000000:
        with st.spinner("🚀 Initializing AI Engine..."):
            try:
                opener = urllib.request.build_opener()
                opener.addheaders = [('User-agent', 'Mozilla/5.0')]
                urllib.request.install_opener(opener)
                urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            except Exception as e:
                st.error(f"Error loading model: {e}")
                return None
    return tf.keras.models.load_model(MODEL_PATH)

# --- 3. CUSTOM CSS ---
st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background-image: url("https://images.unsplash.com/photo-1518531933037-91b2f5f229cc?q=80&w=2000&auto=format&fit=crop");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    [data-testid="stSidebar"] { display: none; }
    .huge-title {
        font-size: 70px;
        font-weight: 900;
        color: #ffffff;
        text-shadow: 4px 4px 15px rgba(0,0,0,0.9);
        text-align: center;
        width: 100%;
        margin-top: 40px;
    }
    .stButton>button {
        border-radius: 50px !important;
        background-color: #28a745 !important;
        color: white !important;
        font-weight: bold !important;
        width: 100%;
    }
    .result-label {
        color: #1E5128;
        font-size: 32px;
        font-weight: bold;
        text-align: center;
        margin-top: 25px;
        background: rgba(255, 255, 255, 0.85);
        padding: 15px;
        border-radius: 15px;
    }
    .measures-box {
        background: rgba(255, 255, 255, 0.9);
        color: #1E5128;
        padding: 20px;
        border-radius: 15px;
        margin-top: 20px;
        border-left: 10px solid #ff4b4b;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 4. NAVIGATION ---
if 'page' not in st.session_state:
    st.session_state.page = 'Home'
if 'last_result' not in st.session_state:
    st.session_state.last_result = None

st.markdown('<h1 class="huge-title">Crop Disease Detector</h1>', unsafe_allow_html=True)

_, col1, col2, col3, _ = st.columns([1.5, 2, 2, 2, 1.5])
with col1:
    if st.button("🏠 HOME"): st.session_state.page = 'Home'
with col2:
    if st.button("🔍 ANALYZE LEAF"): st.session_state.page = 'Detection'
with col3:
    if st.button("🛡️ MEASURES"): st.session_state.page = 'Measures'

# --- 5. PAGE CONTENT ---

if st.session_state.page == 'Home':
    st.markdown("<div style='text-align: center; color: white;'><h2>Welcome to AI-Powered Diagnosis</h2><p style='font-size: 20px;'>Upload a photo to see health status and safety measures.</p></div>", unsafe_allow_html=True)

elif st.session_state.page == 'Detection':
    st.markdown("<style>[data-testid='stAppViewContainer'] { background-image: url('https://images.unsplash.com/photo-1501004318641-729e8e26bd05?q=80&w=2000&auto=format&fit=crop') !important; }</style>", unsafe_allow_html=True)
    
    model = load_trained_model()
    try:
        with open('class_indices.json', 'r') as f:
            class_indices = json.load(f)
        class_names = list(class_indices.values())
    except:
        st.error("Missing class_indices.json")
        class_names = []

    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        _, img_col, _ = st.columns([1, 2, 1])
        with img_col:
            st.image(image, caption='Target Leaf', use_container_width=True)
        
        if st.button("RUN AI DIAGNOSIS"):
            with st.spinner("Analyzing..."):
                img = image.resize((224, 224))
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0) / 255.0
                preds = model.predict(img_array)
                st.session_state.last_result = class_names[np.argmax(preds)]
                st.balloons()

        if st.session_state.last_result:
            st.markdown(f'<div class="result-label">Diagnosis: {st.session_state.last_result}</div>', unsafe_allow_html=True)
            
            # --- DYNAMIC SAFETY MEASURES LOGIC ---
            if "healthy" in st.session_state.last_result.lower():
                st.success("✨ This leaf is Healthy! No safety measures required. Keep up the good work!")
            else:
                st.markdown(f"""
                <div class="measures-box">
                    <h3>🛡️ Safety Measures for {st.session_state.last_result}</h3>
                    <ul>
                        <li><b>Isolate:</b> Remove this plant from others immediately to stop the spread.</li>
                        <li><b>Treatment:</b> Apply appropriate organic fungicides or pesticides.</li>
                        <li><b>Sanitation:</b> Sterilize all tools used on this plant.</li>
                        <li><b>Pruning:</b> Cut off the infected leaves and destroy them (do not compost).</li>
                    </ul>
                    <p style='font-size: 14px;'><i>Note: Visit the 'Measures' page for more general farming tips.</i></p>
                </div>
                """, unsafe_allow_html=True)

elif st.session_state.page == 'Measures':
    st.markdown("<style>[data-testid='stAppViewContainer'] { background-image: url('https://images.unsplash.com/photo-1501004318641-729e8e26bd05?q=80&w=2000&auto=format&fit=crop') !important; }</style>", unsafe_allow_html=True)
    st.markdown("<div style='color: #1E5128; text-align: center;'><h2>🛡️ General Farming Safety</h2><p>Always maintain soil health and tool hygiene to prevent future outbreaks.</p></div>", unsafe_allow_html=True)
