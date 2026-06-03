import argparse
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers

from config import BEST_MODEL_PATH, CLASS_NAMES, IMAGE_SIZE, MODEL_DIR
from dataset import make_oct_dataset


def build_model(num_classes, learning_rate, fine_tune_at=None):
    data_augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
        ],
        name="augmentation",
    )

    base_model = tf.keras.applications.MobileNetV3Large(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
        include_preprocessing=True,
    )

    base_model.trainable = fine_tune_at is not None
    if fine_tune_at is not None:
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

    inputs = layers.Input(shape=(*IMAGE_SIZE, 3))
    x = data_augmentation(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Folder containing train, val, and test folders.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--fine-tune-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--fine-tune-learning-rate", type=float, default=1e-5)
    parser.add_argument("--fine-tune-at", type=int, default=200)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train_ds = make_oct_dataset(train_dir, args.batch_size, shuffle=True)
    val_ds = make_oct_dataset(val_dir, args.batch_size, shuffle=False)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            BEST_MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
        ),
    ]

    model = build_model(len(CLASS_NAMES), args.learning_rate)
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)

    fine_tune_model = build_model(
        len(CLASS_NAMES),
        args.fine_tune_learning_rate,
        fine_tune_at=args.fine_tune_at,
    )
    fine_tune_model.set_weights(model.get_weights())
    fine_tune_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.fine_tune_epochs,
        callbacks=callbacks,
    )

    fine_tune_model.save(BEST_MODEL_PATH)
    print(f"Saved model to {BEST_MODEL_PATH}")


if __name__ == "__main__":
    main()
