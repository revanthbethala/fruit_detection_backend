from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import json
import pandas as pd

# ============================================================
# CONFIG
# ============================================================
IMG_SIZE = 224
MODEL_PATH = "fruit_veg_model.keras"
CLASS_NAMES_PATH = "class_names (1).json"
NUTRITION_CSV_PATH = "fruit_vegetable_nutrition.csv"
HEALTH_JSON_PATH = "fruit_vegetable_health_guide.json"

# ============================================================
# APP INIT
# ============================================================
app = Flask(__name__)
CORS(app)

# ============================================================
# LOAD MODEL & METADATA
# ============================================================
model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

nutrition_df = pd.read_csv(NUTRITION_CSV_PATH)
nutrition_df["Item"] = nutrition_df["Item"].str.lower()

with open(HEALTH_JSON_PATH, "r") as f:
    health_data = json.load(f)

# map: name -> full health object
health_map = {item["name"].lower(): item for item in health_data}

# ============================================================
# IMAGE PREPROCESSING
# ============================================================
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

# ============================================================
# ROUTES
# ============================================================
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        file = request.files["file"]
        img_array = preprocess_image(file.read())

        preds = model.predict(img_array, verbose=0)[0]
        idx = int(np.argmax(preds))
        confidence = float(preds[idx])

        predicted_name = class_names[idx].lower()

        # ---------------- Nutrition Lookup ----------------
        nutrition_row = nutrition_df[nutrition_df["Item"] == predicted_name]
        nutrition_info = (
            nutrition_row.to_dict(orient="records")[0]
            if not nutrition_row.empty else {}
        )

        # ---------------- Health Lookup (NEW FORMAT) ----------------
        health_info = health_map.get(predicted_name, {})

        # ---------------- Final Response ----------------
        return jsonify({
            "prediction": {
                "name": predicted_name,
                "confidence": round(confidence, 4)
            },
            "nutrition": nutrition_info,
            "health_guidance": {
                "best_for": health_info.get("best_for", []),
                "avoid_if": health_info.get("avoid_if", []),
                "season": health_info.get("season", []),
                "health_benefits": health_info.get("health_benefits", []),
                "origin": health_info.get("origin", ""),
                "famous_in": health_info.get("famous_in", []),
                "glycemic_index": health_info.get("glycemic_index", ""),
                "key_nutrients": health_info.get("key_nutrients", []),
                "prep_tip": health_info.get("prep_tip", ""),
                "pairs_well_with": health_info.get("pairs_well_with", [])
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ============================================================
# RUN SERVER
# ============================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
