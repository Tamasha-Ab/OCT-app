import argparse
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.append(str(Path(__file__).resolve().parent))

from config import CLASS_NAMES
from preprocess import load_and_preprocess


def predict_image(model, image_path):
    image = load_and_preprocess(image_path)
    batch = np.expand_dims(image, axis=0)
    predictions = model.predict(batch, verbose=0)[0]
    class_index = int(np.argmax(predictions))
    return CLASS_NAMES[class_index], float(predictions[class_index])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model_path)
    label, confidence = predict_image(model, args.image)
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.2%}")


if __name__ == "__main__":
    main()
