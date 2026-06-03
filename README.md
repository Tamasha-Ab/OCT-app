# OCT Eye Disease Prediction System

This project classifies retinal OCT images into eight Retinal OCT C8 classes:

- AMD
- CNV
- CSR
- DME
- DR
- DRUSEN
- MH
- NORMAL

It uses OpenCV preprocessing, TensorFlow/Keras augmentation, MobileNetV3-Large transfer learning, evaluation metrics and a Flask web app for image upload predictions.

## Project Structure

````text
oct-eye-disease/
  app/
    app.py
    templates/
      index.html
    static/
      styles.css
  src/
    config.py
    preprocess.py
    train.py
    evaluate.py
    predict.py
  requirements.txt
  README.md
``

## Dataset Layout

For the Retinal OCT C8 dataset, use class folders like this if you train locally:

```text
dataset/
  train/
    AMD/
    CNV/
    CSR/
    DME/
    DR/
    DRUSEN/
    MH/
    NORMAL/
  val/
    AMD/
    CNV/
    CSR/
    DME/
    DR/
    DRUSEN/
    MH/
    NORMAL/
  test/
    AMD/
    CNV/
    CSR/
    DME/
    DR/
    DRUSEN/
    MH/
    NORMAL/
````

If you trained in Google Colab and already saved `best_c8_model.keras`, place it in the `models/` folder and run the web app.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Train

```bash
python src/train.py --data-dir path\to\dataset --epochs 10 --batch-size 32
```

The best model is expected at:

```text
models/best_c8_model.keras
```

## Evaluate

```bash
python src/evaluate.py --data-dir path\to\dataset --model-path models/best_c8_model.keras
```

Evaluation outputs are saved in `reports/`.

## Predict One Image

```bash
python src/predict.py --model-path models/best_c8_model.keras --image path\to\image.jpeg
```

## Run Web App

```bash
python app/app.py
```

Then open:

```text
http://127.0.0.1:5000
```

## Recommended Workflow

1. Download dataset.
2. Train the frozen MobileNetV3 model.
3. Fine-tune with a lower learning rate.
4. Evaluate on the test set.
5. Use the saved model in the web app.
6. Include the generated graphs and confusion matrix in the report.
