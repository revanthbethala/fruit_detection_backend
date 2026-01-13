import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import json
from PIL import Image
import io

# ============================================================
# CONFIG
# ============================================================
IMG_SIZE = 224
MODEL_PATH = "fruit_veg_model.keras"
CLASS_NAMES_PATH = "class_names.json"
NUTRITION_CSV_PATH = "fruit_vegetable_nutrition.csv"
HEALTH_JSON_PATH = "fruit_guide_adv.json"

st.set_page_config(
    page_title="Fruit & Vegetable Health Analyzer",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# THEME-AWARE CSS (Responsive + Light/Dark Mode)
# ============================================================
st.markdown(
    """
<style>
/* Root variables for professional theme */
:root {
    --card-bg: #ffffff;
    --text-primary: #111827;
    --text-secondary: #4b5563;
    --badge-bg: #dbeafe;
    --badge-text: #1e40af;
    --shadow: rgba(0,0,0,0.08);
    --border-color: #e5e7eb;
    --accent: #2563eb;
    --accent-light: #3b82f6;
    --success: #059669;
    --warning: #d97706;
}

/* Dark mode overrides */
@media (prefers-color-scheme: dark) {
    :root {
        --card-bg: #1e293b;
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
        --badge-bg: #1e3a5f;
        --badge-text: #60a5fa;
        --shadow: rgba(0,0,0,0.4);
        --border-color: #334155;
        --accent: #3b82f6;
        --accent-light: #60a5fa;
    }
}

/* Streamlit dark theme detection */
[data-testid="stAppViewContainer"][data-theme="dark"] {
    --card-bg: #1e293b;
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --badge-bg: #1e3a5f;
    --badge-text: #60a5fa;
    --shadow: rgba(0,0,0,0.4);
    --border-color: #334155;
    --accent: #3b82f6;
    --accent-light: #60a5fa;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}

h1 {
    color: var(--text-primary);
    font-weight: 700;
    margin-bottom: 0.5rem;
}

h2, h3 {
    color: var(--text-primary);
    font-weight: 600;
}

/* Professional Card Design */
.card {
    background: var(--card-bg);
    border-radius: 16px;
    padding: 2rem;
    box-shadow: 0 1px 3px var(--shadow), 0 1px 2px rgba(0,0,0,0.06);
    margin-bottom: 1.5rem;
    border: 1px solid var(--border-color);
    transition: all 0.3s ease;
}

.card:hover {
    box-shadow: 0 4px 6px -1px var(--shadow), 0 2px 4px -1px rgba(0,0,0,0.06);
    transform: translateY(-2px);
}

/* Refined Badges */
.badge {
    display: inline-block;
    padding: 10px 18px;
    border-radius: 24px;
    background: var(--badge-bg);
    color: var(--badge-text);
    font-size: 13px;
    font-weight: 600;
    margin: 6px 6px 6px 0;
    transition: all 0.2s ease;
    letter-spacing: 0.3px;
}

.badge:hover {
    transform: translateY(-1px);
    box-shadow: 0 2px 4px var(--shadow);
}

/* Info Grid - Professional Layout */
.info-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 3rem;
    margin-top: 1.5rem;
}

.info-item {
    background: linear-gradient(135deg, var(--card-bg) 0%, var(--card-bg) 100%);
    padding: 1.5rem;
    border-radius: 12px;
    border-left: 4px solid var(--accent);
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    transition: all 0.3s ease;
}

.info-item:hover {
    transform: translateX(4px);
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}

.info-label {
    font-weight: 700;
    color: var(--text-primary);
    font-size: 13px;
    margin-bottom: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.info-value {
    color: var(--text-secondary);
    font-size: 15px;
    line-height: 1.7;
    font-weight: 400;
}

/* Mobile Responsive */
@media (max-width: 768px) {
    .block-container {
        padding: 1rem;
    }
    
    .card {
        padding: 1.25rem;
        border-radius: 12px;
    }
    
    .badge {
        font-size: 12px;
        padding: 8px 14px;
    }
    
    .info-grid {
        grid-template-columns: 1fr;
        gap: 1rem;
    }
    
    .info-item {
        padding: 1.25rem;
    }
}

/* Enhanced Table styling */
.stDataFrame {
    border-radius: 8px;
    overflow: hidden;
}

/* Metric styling enhancement */
[data-testid="stMetricValue"] {
    font-size: 1.75rem;
    font-weight: 700;
    color: var(--accent);
}

[data-testid="stMetricDelta"] {
    font-size: 0.9rem;
    font-weight: 500;
}

/* Upload widget enhancement */
[data-testid="stFileUploader"] {
    border-radius: 12px;
}

/* Divider styling */
hr {
    margin: 2rem 0;
    border: none;
    border-top: 2px solid var(--border-color);
    opacity: 0.6;
}

/* Section headers */
.stMarkdown h2 {
    padding-bottom: 0.5rem;
    border-bottom: 3px solid var(--accent);
    display: inline-block;
    margin-bottom: 1.5rem;
}

/* Footer */
.footer {
    text-align: center;
    color: var(--text-secondary);
    padding: 2rem 0 1rem 0;
    font-size: 14px;
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# CACHED LOADERS
# ============================================================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_metadata():
    with open(CLASS_NAMES_PATH) as f:
        class_names = json.load(f)

    nutrition_df = pd.read_csv(NUTRITION_CSV_PATH)
    nutrition_df["Item"] = nutrition_df["Item"].str.lower()

    with open(HEALTH_JSON_PATH) as f:
        health_data = json.load(f)

    # Handle both array of objects and single object
    if isinstance(health_data, list):
        health_map = {item["name"].lower(): item for item in health_data}
    else:
        health_map = {health_data["name"].lower(): health_data}

    return class_names, nutrition_df, health_map


model = load_model()
class_names, nutrition_df, health_map = load_metadata()


# ============================================================
# IMAGE PREPROCESSING
# ============================================================
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


# ============================================================
# UI COMPONENTS
# ============================================================
def render_badges(items, label):
    """Render badge section with proper handling"""
    if items and len(items) > 0:
        st.markdown(f"<div class='info-label'>{label}</div>", unsafe_allow_html=True)
        badges_html = "".join(f"<span class='badge'>{item}</span>" for item in items)
        st.markdown(f"<div>{badges_html}</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)


def render_info_grid(health_info):
    """Render health information in a responsive grid"""
    st.markdown("<div class='info-grid'>", unsafe_allow_html=True)

    # Health Benefits
    if health_info.get("health_benefits"):
        benefits = ", ".join(health_info["health_benefits"])
        st.markdown(
            f"""
        <div class='info-item'>
            <div class='info-label'>🌟 Health Benefits</div>
            <div class='info-value'>{benefits}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Key Nutrients
    if health_info.get("key_nutrients"):
        nutrients = ", ".join(health_info["key_nutrients"])
        st.markdown(
            f"""
        <div class='info-item'>
            <div class='info-label'>💊 Key Nutrients</div>
            <div class='info-value'>{nutrients}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Glycemic Index
    if health_info.get("glycemic_index"):
        st.markdown(
            f"""
        <div class='info-item'>
            <div class='info-label'>📊 Glycemic Index</div>
            <div class='info-value'>{health_info["glycemic_index"]}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    # Origin
    if health_info.get("origin"):
        st.markdown(
            f"""
        <div class='info-item'>
            <div class='info-label'>🌍 Origin</div>
            <div class='info-value'>{health_info["origin"]}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    # Famous In
    if health_info.get("famous_in"):
        famous = ", ".join(health_info["famous_in"])
        st.markdown(
            f"""
        <div class='info-item'>
            <div class='info-label'>🗺️ Popular In</div>
            <div class='info-value'>{famous}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    # Preparation Tip
    if health_info.get("prep_tip"):
        st.markdown(
            f"""
        <div class='info-item'>
            <div class='info-label'>👨‍🍳 Preparation Tip</div>
            <div class='info-value'>{health_info["prep_tip"]}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    # Pairs Well With
    if health_info.get("pairs_well_with"):
        pairs = ", ".join(health_info["pairs_well_with"])
        st.markdown(
            f"""
        <div class='info-item'>
            <div class='info-label'>🍽️ Pairs Well With</div>
            <div class='info-value'>{pairs}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# MAIN UI
# ============================================================
st.title("🥗 Fruit & Vegetable Health Analyzer")
st.markdown(
    "<p style='font-size: 1.1rem; color: var(--text-secondary); margin-bottom: 2rem;'>Professional nutrition analysis and health insights powered by AI</p>",
    unsafe_allow_html=True,
)

# Responsive columns - info left, image right on desktop
col1, col2 = st.columns([1.2, 1], gap="large")

# ---------------- LEFT: UPLOAD & PREDICTION ----------------
with col1:
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Select a fruit or vegetable image",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

    if uploaded_file:
        image_bytes = uploaded_file.read()
        image = Image.open(io.BytesIO(image_bytes))
        with st.spinner("🔍 Analyzing image..."):
            img_array = preprocess_image(image_bytes)
            preds = model.predict(img_array, verbose=0)[0]
            idx = int(np.argmax(preds))
            confidence = float(preds[idx])
            predicted_name = class_names[idx].lower()

        # st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🎯 Analysis Result")


# ---------------- RIGHT: IMAGE DISPLAY ----------------
with col2:
    if uploaded_file:
        st.image(image, caption="Uploaded Image", use_container_width=True)
    else:
        # st.markdown("<div class='card'>", unsafe_allow_html=True)
        # st.info("👆 Upload an image to begin analysis")
        st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# DETAILS SECTION
# ============================================================
if uploaded_file:
    nutrition_row = nutrition_df[nutrition_df["Item"] == predicted_name]
    nutrition_info = (
        nutrition_row.to_dict(orient="records")[0] if not nutrition_row.empty else {}
    )

    health_info = health_map.get(predicted_name, {})

    st.markdown("---")
    st.markdown("## 📋 Detailed Information")

    # Responsive columns for nutrition and health guidance
    colA, colB = st.columns([1, 1], gap="large")

    # ---------------- NUTRITION ----------------
    with colA:
        st.subheader("📊 Nutrition Facts")

        if nutrition_info:
            nutrition_table = pd.DataFrame(
                nutrition_info.items(), columns=["Nutrient", "Value"]
            )
            nutrition_table = nutrition_table[nutrition_table["Nutrient"] != "Item"]
            st.dataframe(nutrition_table, use_container_width=True, hide_index=True)
        else:
            st.info("Nutrition data not available for this item.")

    # ---------------- HEALTH GUIDANCE ----------------
    with colB:
        st.subheader("💚 Health Guidance")

        render_badges(health_info.get("best_for", []), "✅ Best For")
        render_badges(health_info.get("avoid_if", []), "⚠️ Avoid If")
        render_badges(health_info.get("season", []), "📅 Season")

    # ---------------- COMPREHENSIVE DETAILS ----------------
    st.subheader("📖 Complete Health Profile")
    render_info_grid(health_info)
