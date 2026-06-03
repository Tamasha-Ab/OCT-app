import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

from config import CLASS_NAMES, REPORT_DIR
from dataset import make_oct_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Folder containing train, val, and test folders.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    test_dir = Path(args.data_dir) / "test"
    test_ds = make_oct_dataset(test_dir, args.batch_size, shuffle=False)
    model = tf.keras.models.load_model(args.model_path)

    y_true = []
    y_pred = []

    for images, labels in test_ds:
        predictions = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(predictions, axis=1))

    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)
    report_path = REPORT_DIR / "classification_report.txt"
    report_path.write_text(report, encoding="utf-8")

    cm = confusion_matrix(y_true, y_pred)
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    display.plot(cmap="Blues", xticks_rotation=45)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "confusion_matrix.png", dpi=200)

    print(report)
    print(f"Saved report files to {REPORT_DIR}")


if __name__ == "__main__":
    main()
