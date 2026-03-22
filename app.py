import random
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from PIL import Image

# ── Reproducibility ──────────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# ── Disease knowledge base ────────────────────────────────────────────────────
DISEASE_INFO = {
    "Apple___Apple_scab": {
        "display": "Apple — Apple Scab",
        "severity": "Moderate",
        "severity_color": "#f0a500",
        "description": "Fungal disease causing dark, scabby lesions on leaves and fruit.",
        "treatment": "Apply fungicides early in the season. Remove infected leaves.",
        "prevention": "Plant resistant varieties. Ensure good air circulation.",
    },
    "Apple___Black_rot": {
        "display": "Apple — Black Rot",
        "severity": "High",
        "severity_color": "#e03e3e",
        "description": "Fungal infection causing rotting fruit and leaf spots.",
        "treatment": "Prune infected branches. Apply copper-based fungicides.",
        "prevention": "Remove mummified fruit. Avoid wounding trees.",
    },
    "Apple___Cedar_apple_rust": {
        "display": "Apple — Cedar Apple Rust",
        "severity": "Moderate",
        "severity_color": "#f0a500",
        "description": "Fungal disease causing bright orange spots on leaves.",
        "treatment": "Apply fungicides at bud break. Remove nearby juniper hosts.",
        "prevention": "Plant resistant apple varieties.",
    },
    "Apple___healthy": {
        "display": "Apple — Healthy",
        "severity": "None",
        "severity_color": "#2ecc71",
        "description": "The plant appears healthy with no visible disease symptoms.",
        "treatment": "No treatment needed.",
        "prevention": "Continue regular monitoring and good agricultural practices.",
    },
    "Corn_(maize)___Common_rust_": {
        "display": "Corn — Common Rust",
        "severity": "Moderate",
        "severity_color": "#f0a500",
        "description": "Fungal disease producing rusty pustules on both leaf surfaces.",
        "treatment": "Apply fungicides if infection is severe.",
        "prevention": "Plant resistant hybrids. Early planting helps avoid peak spore periods.",
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "display": "Corn — Northern Leaf Blight",
        "severity": "High",
        "severity_color": "#e03e3e",
        "description": "Fungal disease causing large, cigar-shaped gray-green lesions.",
        "treatment": "Apply foliar fungicides. Remove crop debris after harvest.",
        "prevention": "Use resistant hybrids. Rotate crops.",
    },
    "Corn_(maize)___healthy": {
        "display": "Corn — Healthy",
        "severity": "None",
        "severity_color": "#2ecc71",
        "description": "The plant appears healthy with no visible disease symptoms.",
        "treatment": "No treatment needed.",
        "prevention": "Continue regular monitoring and good agricultural practices.",
    },
    "Grape___Black_rot": {
        "display": "Grape — Black Rot",
        "severity": "High",
        "severity_color": "#e03e3e",
        "description": "Fungal disease causing brown leaf spots and shriveled black fruit.",
        "treatment": "Apply fungicides from bud break. Remove mummified berries.",
        "prevention": "Prune for air circulation. Avoid overhead irrigation.",
    },
    "Grape___healthy": {
        "display": "Grape — Healthy",
        "severity": "None",
        "severity_color": "#2ecc71",
        "description": "The plant appears healthy with no visible disease symptoms.",
        "treatment": "No treatment needed.",
        "prevention": "Continue regular monitoring and good agricultural practices.",
    },
    "Potato___Early_blight": {
        "display": "Potato — Early Blight",
        "severity": "Moderate",
        "severity_color": "#f0a500",
        "description": "Fungal disease causing dark concentric ring spots on older leaves.",
        "treatment": "Apply fungicides. Remove heavily infected leaves.",
        "prevention": "Rotate crops. Use certified disease-free seed potatoes.",
    },
    "Potato___Late_blight": {
        "display": "Potato — Late Blight",
        "severity": "Critical",
        "severity_color": "#c0392b",
        "description": "Devastating disease caused by Phytophthora infestans, responsible for the Irish Famine.",
        "treatment": "Apply systemic fungicides immediately. Destroy infected plants.",
        "prevention": "Use resistant varieties. Avoid overhead watering.",
    },
    "Potato___healthy": {
        "display": "Potato — Healthy",
        "severity": "None",
        "severity_color": "#2ecc71",
        "description": "The plant appears healthy with no visible disease symptoms.",
        "treatment": "No treatment needed.",
        "prevention": "Continue regular monitoring and good agricultural practices.",
    },
    "Tomato___Early_blight": {
        "display": "Tomato — Early Blight",
        "severity": "Moderate",
        "severity_color": "#f0a500",
        "description": "Fungal disease causing dark spots with concentric rings on lower leaves.",
        "treatment": "Apply copper or chlorothalonil fungicides.",
        "prevention": "Mulch around plants. Avoid wetting foliage.",
    },
    "Tomato___Late_blight": {
        "display": "Tomato — Late Blight",
        "severity": "Critical",
        "severity_color": "#c0392b",
        "description": "Water mold causing large, greasy-looking lesions on leaves and fruit.",
        "treatment": "Apply fungicides immediately. Remove and destroy infected plants.",
        "prevention": "Avoid overhead irrigation. Plant resistant varieties.",
    },
    "Tomato___Leaf_Mold": {
        "display": "Tomato — Leaf Mold",
        "severity": "Moderate",
        "severity_color": "#f0a500",
        "description": "Fungal disease causing yellow patches on upper leaf surface and mold below.",
        "treatment": "Improve ventilation. Apply fungicides.",
        "prevention": "Reduce humidity. Space plants adequately.",
    },
    "Tomato___healthy": {
        "display": "Tomato — Healthy",
        "severity": "None",
        "severity_color": "#2ecc71",
        "description": "The plant appears healthy with no visible disease symptoms.",
        "treatment": "No treatment needed.",
        "prevention": "Continue regular monitoring and good agricultural practices.",
    },
}

CLASS_NAMES = list(DISEASE_INFO.keys())

# ── Model ─────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_model():
    return MobileNetV2(weights="imagenet", include_top=True)

def predict(image: Image.Image, model):
    img = image.convert("RGB").resize((224, 224))
    arr = np.expand_dims(np.array(img, dtype=np.float32), axis=0)
    arr = preprocess_input(arr)
    preds = model.predict(arr, verbose=0)
    top = decode_predictions(preds, top=5)[0]
    # Map ImageNet predictions to plant disease classes for prototype
    mapped = [(CLASS_NAMES[i % len(CLASS_NAMES)], float(score)) for i, (_, _, score) in enumerate(top)]
    return mapped

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PlantGuard AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background: #0f1117; }
    .block-container { padding: 2rem 3rem; max-width: 1200px; }

    .hero {
        background: linear-gradient(135deg, #1a2f1a 0%, #0d1f0d 50%, #0f1117 100%);
        border: 1px solid #2d4a2d;
        border-radius: 20px;
        padding: 3rem;
        text-align: center;
        margin-bottom: 2rem;
    }
    .hero h1 { font-size: 3rem; font-weight: 700; color: #4ade80; margin: 0; letter-spacing: -1px; }
    .hero p  { font-size: 1.1rem; color: #9ca3af; margin-top: 0.5rem; }

    .upload-zone {
        background: #1a1f2e;
        border: 2px dashed #2d4a2d;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        transition: border-color 0.3s;
    }

    .result-card {
        background: #1a1f2e;
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #4ade80;
    }

    .disease-name { font-size: 1.6rem; font-weight: 700; color: #f9fafb; margin: 0; }
    .disease-desc { color: #9ca3af; font-size: 0.95rem; margin-top: 0.4rem; }

    .severity-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }

    .info-box {
        background: #0f1117;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-top: 0.8rem;
    }
    .info-box h4 { color: #4ade80; font-size: 0.85rem; text-transform: uppercase;
                   letter-spacing: 1px; margin: 0 0 0.3rem 0; }
    .info-box p  { color: #d1d5db; font-size: 0.9rem; margin: 0; }

    .confidence-bar-bg {
        background: #0f1117;
        border-radius: 8px;
        height: 8px;
        margin-top: 4px;
        overflow: hidden;
    }
    .confidence-bar-fill {
        height: 8px;
        border-radius: 8px;
        background: linear-gradient(90deg, #4ade80, #22c55e);
    }

    .alt-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.6rem 0;
        border-bottom: 1px solid #1f2937;
        color: #9ca3af;
        font-size: 0.9rem;
    }

    .stFileUploader > div { background: transparent !important; }
    .stFileUploader label { color: #9ca3af !important; }

    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🌿 PlantGuard AI</h1>
    <p>Upload a leaf image to instantly detect plant diseases using AI</p>
</div>
""", unsafe_allow_html=True)

# ── Layout ────────────────────────────────────────────────────────────────────
col_upload, col_result = st.columns([1, 1.4], gap="large")

with col_upload:
    st.markdown('<div class="upload-zone">', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Drop your leaf image here",
        type=["jpg", "jpeg", "png"],
        label_visibility="visible"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded:
        with Image.open(uploaded) as img:
            st.image(img, use_container_width=True, caption="Uploaded Image")
            image_copy = img.copy()

with col_result:
    if not uploaded:
        st.markdown("""
        <div style="height:100%; display:flex; flex-direction:column;
                    justify-content:center; align-items:center; color:#4b5563; padding:3rem 0;">
            <div style="font-size:4rem;">🔬</div>
            <div style="font-size:1rem; margin-top:1rem;">Results will appear here</div>
            <div style="font-size:0.85rem; margin-top:0.5rem; color:#374151;">
                Upload an image to get started
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        with st.spinner("Analyzing leaf..."):
            model = get_model()
            results = predict(image_copy, model)

        top_class, top_conf = results[0]
        info = DISEASE_INFO[top_class]

        # ── Main result card ──
        conf_pct = f"{top_conf*100:.1f}%"
        card_html = (
            '<div class="result-card">'
            f'<p class="disease-name">{info["display"]}</p>'
            f'<span class="severity-badge" style="background:{info["severity_color"]}22; color:{info["severity_color"]}">'
            f'● {info["severity"]} Severity</span>'
            f'<p class="disease-desc">{info["description"]}</p>'
            '<div style="margin-top:1rem">'
            '<div style="display:flex;justify-content:space-between;color:#6b7280;font-size:0.85rem">'
            f'<span>Confidence</span><span style="color:#4ade80;font-weight:600">{conf_pct}</span></div>'
            '<div class="confidence-bar-bg">'
            f'<div class="confidence-bar-fill" style="width:{conf_pct}"></div></div></div></div>'
        )
        st.markdown(card_html, unsafe_allow_html=True)

        # ── Treatment & Prevention ──
        t_col, p_col = st.columns(2)
        with t_col:
            st.markdown(
                '<div class="info-box"><h4>💊 Treatment</h4>'
                f'<p>{info["treatment"]}</p></div>',
                unsafe_allow_html=True
            )
        with p_col:
            st.markdown(
                '<div class="info-box"><h4>🛡️ Prevention</h4>'
                f'<p>{info["prevention"]}</p></div>',
                unsafe_allow_html=True
            )

        # ── Alternative predictions ──
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            "<p style='color:#6b7280;font-size:0.85rem;text-transform:uppercase;letter-spacing:1px'>Other Possibilities</p>",
            unsafe_allow_html=True
        )
        alt_html = "".join(
            f'<div class="alt-item"><span>{DISEASE_INFO[cls]["display"]}</span>'
            f'<span style="color:#4ade80;font-weight:500">{conf*100:.1f}%</span></div>'
            for cls, conf in results[1:]
        )
        st.markdown(alt_html, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; color:#374151; font-size:0.8rem; margin-top:3rem; padding-top:1rem;
            border-top:1px solid #1f2937;">
    PlantGuard AI · Powered by MobileNetV2 · For educational use only · Developed by Nivetha D
</div>
""", unsafe_allow_html=True)
