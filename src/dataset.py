from pathlib import Path

import numpy as np
import tensorflow as tf

from config import CLASS_NAMES, IMAGE_SIZE
from preprocess import load_and_preprocess

IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.bmp")


def collect_image_paths(split_dir):
    split_dir = Path(split_dir)
    paths = []
    labels = []

    for class_index, class_name in enumerate(CLASS_NAMES):
        class_dir = split_dir / class_name
        for pattern in IMAGE_EXTENSIONS:
            for image_path in class_dir.glob(pattern):
                paths.append(str(image_path))
                labels.append(class_index)

    if not paths:
        raise ValueError(f"No images found in {split_dir}")

    return paths, labels


def _load_image_numpy(path):
    path = path.decode("utf-8")
    image = load_and_preprocess(path, image_size=IMAGE_SIZE)
    return image.astype(np.float32)


def make_oct_dataset(split_dir, batch_size, shuffle):
    paths, labels = collect_image_paths(split_dir)
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(len(paths), 4096), reshuffle_each_iteration=True)

    def load_example(path, label):
        image = tf.numpy_function(_load_image_numpy, [path], Tout=tf.float32)
        image.set_shape((*IMAGE_SIZE, 3))
        label = tf.one_hot(label, depth=len(CLASS_NAMES))
        return image, label

    return (
        dataset
        .map(load_example, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
