import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import json

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Crop Disease Detection",
    page_icon="🌿",
    layout="wide"
)

# --- 2. EMBEDDED CSS STYLING ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stAlert { border-radius: 10px; }
    footer {visibility: hidden;}
    .footer-text {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: white;
        color: #6c757d;
        text-align: center;
        padding: 10px;
        font-size: 14px;
        border-top: 1px solid #e9ecef;
        z-index: 100;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. ASSET LOADING (Model & Classes) ---
@st.cache_resource
def load_assets():
    # Load Model with the specific filename you provided
    model = tf.keras.models.load_model('plant_disease_model.h5', compile=False)
    
    # Load Class Indices
    try:
        with open('class_indices.json', 'r') as f:
            class_dict = json.load(f)
        # Sort by index value to ensure correct label mapping
        class_names = [k for k, v in sorted(class_dict.items(), key=lambda item: item[1])]
    except Exception:
        # Fallback labels if JSON is missing
        class_names = ['Healthy Cotton', 'Diseased Cotton', 'Healthy Tomato', 'Tomato Bacterial Spot']
        
    return model, class_names

try:
    model, CLASS_NAMES = load_assets()
except Exception as e:
    st.error(f"⚠️ Error: Ensure 'plant_disease_model.h5' and 'class_indices.json' are in the same folder as this script.")

# --- 4. REMEDY KNOWLEDGE BASE ---
REMEDIES = {
    'Healthy Cotton': "Your cotton plant looks healthy! Keep maintaining proper irrigation and nitrogen levels.",
    'Diseased Cotton': "Possible Fungal Infection. Action: Remove infected leaves and apply a copper-based fungicide or neem oil.",
    'Healthy Tomato': "The tomato leaf is in great condition. Ensure consistent sunlight and avoid watering the leaves directly.",
    'Tomato Bacterial Spot': "Bacterial Spot detected. Action: Use copper-based sprays, avoid overhead irrigation, and rotate crops next season."
}

# --- 5. MAIN UI HEADER ---
st.title("🌿 Crop Disease Detection System")
st.markdown("##### *A Machine Learning Approach for Detecting Crop Disease*")
st.write("---")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.header("📸 Image Upload")
    uploaded_file = st.file_uploader("Upload a leaf image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption='Captured Leaf Image', use_column_width=True)

with col2:
    st.header("🔍 Diagnostic Analysis")
    
    if uploaded_file is not None:
        with st.spinner("Model is analyzing pixel patterns..."):
            # --- 6. PRE-PROCESSING ---
            img = image.resize((224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = img_array / 255.0  # Normalization
            img_array = np.expand_dims(img_array, axis=0)

            # --- 7. PREDICTION LOGIC ---
            predictions = model.predict(img_array)
            # Use softmax for probability distribution
            score = tf.nn.softmax(predictions[0]) 
            
            result_index = np.argmax(score)
            result_label = CLASS_NAMES[result_index]
            confidence = 100 * np.max(score)

            # --- 8. RESULT DISPLAY ---
            if confidence > 65: 
                st.success(f"### Prediction: **{result_label}**")
                st.metric(label="Model Confidence Score", value=f"{confidence:.2f}%")
                
                st.markdown("---")
                st.subheader("💡 Recommended Management:")
                st.info(REMEDIES.get(result_label, "Consult an agricultural specialist for a detailed treatment plan."))
            else:
                st.warning("⚠️ **Ambiguous Diagnosis.** \n\nThe model is unsure. Please provide a clearer, top-down photo with better lighting.")

    else:
        st.info("Awaiting image upload for diagnosis...")

# --- 9. TECHNICAL SIDEBAR ---
with st.sidebar:
    st.title("Project Overview")
    st.write("**Title:** A Machine Learning Approach for Detecting Crop Disease")
    st.info("Utilizes a Deep Convolutional Neural Network (CNN) to identify plant pathologies from leaf imagery.")
    st.subheader("Technical Specs")
    st.markdown(f"- **Classes:** {len(CLASS_NAMES)}")
    st.markdown("- **Framework:** TensorFlow 2.x")
    st.markdown("- **Input Size:** 224x224 RGB")
    
    st.write("---")
    st.caption("Final Year Submission")

# --- 10. FORMAL FOOTER ---
st.markdown("""
    <div class="footer-text">
        © 2026 | A Machine Learning Approach for Detecting Crop Disease
    </div>
    """, unsafe_allow_html=True)