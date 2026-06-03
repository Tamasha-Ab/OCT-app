import cv2
import numpy as np

from config import IMAGE_SIZE


def preprocess_oct_image(image_bgr_or_gray, image_size=IMAGE_SIZE):
    """Apply grayscale, denoising, CLAHE, resize, and MobileNet-ready RGB conversion."""
    if image_bgr_or_gray is None:
        raise ValueError("Image could not be loaded.")

    if len(image_bgr_or_gray.shape) == 3:
        gray = cv2.cvtColor(image_bgr_or_gray, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_bgr_or_gray

    denoised = cv2.medianBlur(gray, 3)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    resized = cv2.resize(enhanced, image_size, interpolation=cv2.INTER_AREA)

    rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    return rgb.astype(np.float32)


def load_and_preprocess(path, image_size=IMAGE_SIZE):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return preprocess_oct_image(image, image_size=image_size)
