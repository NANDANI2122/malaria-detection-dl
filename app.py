from flask import Flask, render_template, request, jsonify
import numpy as np
import os
from datetime import datetime
import tensorflow as tf
from PIL import Image
import time

# ----------------------------
# Performance & Log Settings
# ----------------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ----------------------------
# App Configuration
# ----------------------------
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

MODEL_PATH = "malaria_model.h5"
CLASS_NAMES = ['Parasitized', 'Uninfected']
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

model = None
model_error = None


# ----------------------------
# Utility: Check File Extension
# ----------------------------
def allowed_file(filename):
    return "." in filename and \
           filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ----------------------------
# Load Model (Inference Mode)
# ----------------------------
try:
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)

        print("✅ Model Loaded Successfully")
        print("Input Shape:", model.input_shape)
        print("Output Shape:", model.output_shape)

    else:
        model_error = "Model file not found!"
        print("❌ Model file not found!")

except Exception as e:
    model_error = str(e)
    print("❌ Model Loading Error:", model_error)


# ----------------------------
# Image Preprocessing
# ----------------------------
# ----------------------------
# Image Preprocessing (Safe Version)
# ----------------------------
def prepare_image(image):
    try:
        img = Image.open(image)
    except Exception:
        raise ValueError("Invalid or corrupted image file")

    try:
        if img.mode != "RGB":
            img = img.convert("RGB")

        height = model.input_shape[1]
        width = model.input_shape[2]

        img = img.resize((width, height))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        return img_array

    except Exception:
        raise ValueError("Image preprocessing failed")

# ----------------------------
# Routes
# ----------------------------

@app.route("/")
def home():
    return render_template(
        "index.html",
        model_loaded=model is not None,
        model_error=model_error
    )


@app.route("/predict", methods=["POST"])
def predict():

    if model is None:
        return jsonify({"success": False, "error": "Model not loaded"}), 500

    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"success": False, "error": "Empty filename"}), 400

    if not allowed_file(file.filename):
        return jsonify({"success": False, "error": "Invalid file type"}), 400

    try:
        start_time = time.time()

        processed_image = prepare_image(file)
        prediction = model.predict(processed_image, verbose=0)

        # Handle Binary OR Softmax Model Automatically
        if prediction.shape[1] == 1:
            probability = float(prediction[0][0])
            predicted_class = 1 if probability > 0.5 else 0
            confidence = probability if predicted_class == 1 else 1 - probability
        else:
            predicted_class = int(np.argmax(prediction))
            confidence = float(np.max(prediction))

        processing_time = round(time.time() - start_time, 2)

        return jsonify({
            "success": True,
            "prediction": CLASS_NAMES[predicted_class],
            "confidence": round(confidence * 100, 2),
            "processing_time": processing_time,
            "timestamp": datetime.now().strftime("%d/%m/%Y, %H:%M:%S")
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({
        "status": "running",
        "model_loaded": model is not None
    })
from flask import request

@app.route("/report")
def report():
    prediction = request.args.get("prediction")
    confidence = request.args.get("confidence")
    time_taken = request.args.get("time")
    timestamp = request.args.get("timestamp")

    return render_template(
        "report.html",
        prediction=prediction,
        confidence=confidence,
        time=time_taken,
        timestamp=timestamp
    )

# ----------------------------
# Run Server
# ----------------------------
if __name__ == "__main__":
    print("🚀 Server Starting...")
    print("Model Loaded:", model is not None)

    app.run(debug=True, port=5000, use_reloader=False)