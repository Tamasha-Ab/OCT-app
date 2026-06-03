import sys
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.append(str(SRC_DIR))

from config import BEST_MODEL_PATH, CLASS_NAMES
from preprocess import preprocess_oct_image

app = Flask(__name__)
model = None


def get_model():
    global model
    if model is None:
        model = tf.keras.models.load_model(BEST_MODEL_PATH)
    return model


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None

    if request.method == "POST":
        file = request.files.get("image")
        if not file:
            error = "Please upload an OCT image."
        elif not BEST_MODEL_PATH.exists():
            error = f"Model not found at {BEST_MODEL_PATH}. Train the model first."
        else:
            file_bytes = np.frombuffer(file.read(), np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            processed = preprocess_oct_image(image)
            predictions = get_model().predict(np.expand_dims(processed, axis=0), verbose=0)[0]
            class_index = int(np.argmax(predictions))
            top_indices = predictions.argsort()[-3:][::-1]
            result = {
                "label": CLASS_NAMES[class_index],
                "confidence": f"{predictions[class_index] * 100:.2f}%",
                "confidence_value": round(float(predictions[class_index]) * 100, 2),
                "top_predictions": [
                    {
                        "label": CLASS_NAMES[index],
                        "confidence": f"{predictions[index] * 100:.2f}%",
                        "confidence_value": round(float(predictions[index]) * 100, 2),
                    }
                    for index in top_indices
                ],
            }

    return render_template("index.html", result=result, error=error)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
