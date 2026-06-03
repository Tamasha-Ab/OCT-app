from pathlib import Path

IMAGE_SIZE = (224, 224)
CLASS_NAMES = ["AMD", "CNV", "CSR", "DME", "DR", "DRUSEN", "MH", "NORMAL"]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"
BEST_MODEL_PATH = MODEL_DIR / "best_c8_model.keras"
