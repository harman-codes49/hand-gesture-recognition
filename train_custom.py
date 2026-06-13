"""
Train MobileNetV2 on the custom 20-class hand gesture dataset.
Dataset path: ~/Downloads/test/  (folders 0-19, each with ~300 grayscale 50x50 JPGs)

Usage:
    python train_custom.py
"""

import json, os, sys, random
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR   = os.path.expanduser("~/Downloads/test")
MODEL_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "gesture_model.keras")
MAP_PATH   = os.path.join(MODEL_DIR, "label_map.json")
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Label map (visually identified from each folder) ─────────────────────────
LABEL_MAP = {
    0:  "C",
    1:  "Index",
    2:  "L",
    3:  "Three",
    4:  "Four",
    5:  "Palm",
    6:  "OK",
    7:  "Rock",
    8:  "ILoveYou",
    9:  "Spock",
    10: "Down",
    11: "PalmSide",
    12: "Point",
    13: "One",
    14: "Fist",
    15: "Grip",
    16: "Thumb",
    17: "Hang",
    18: "PointDown",
    19: "ThumbUp",
}

IMG_SIZE    = 96          # smaller size saves memory, still plenty for MobileNetV2
BATCH_SIZE  = 16
NUM_CLASSES = len(LABEL_MAP)

# ── Build file/label lists ────────────────────────────────────────────────────
def collect_paths():
    paths, labels = [], []
    for cls_id in range(NUM_CLASSES):
        folder = os.path.join(DATA_DIR, str(cls_id))
        if not os.path.isdir(folder):
            print(f"  WARNING: {folder} not found, skipping")
            continue
        files = [os.path.join(folder, f) for f in os.listdir(folder)
                 if f.lower().endswith((".jpg", ".png"))]
        paths.extend(files)
        labels.extend([cls_id] * len(files))
        print(f"  class {cls_id:2d} ({LABEL_MAP[cls_id]:12s}): {len(files)} images")
    return paths, labels


# ── tf.data pipeline (reads from disk, no RAM blowup) ────────────────────────
def parse_image(path, label):
    raw = tf.io.read_file(path)
    img = tf.image.decode_jpeg(raw, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32)          # keep [0, 255] — preprocess_input handles scaling
    return img, label

def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.15 * 255)
    image = tf.image.random_contrast(image, 0.85, 1.15)
    image = tf.clip_by_value(image, 0.0, 255.0)
    return image, label

def make_datasets(paths, labels):
    combined = list(zip(paths, labels))
    random.seed(42)
    random.shuffle(combined)
    split = int(len(combined) * 0.8)
    train_data, val_data = combined[:split], combined[split:]

    def make_ds(data, augment_fn=None):
        p, l = zip(*data)
        ds = tf.data.Dataset.from_tensor_slices((list(p), list(l)))
        ds = ds.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
        if augment_fn:
            ds = ds.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
        return ds.shuffle(512).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return make_ds(train_data, augment), make_ds(val_data)


# ── Model ────────────────────────────────────────────────────────────────────
def build_model():
    base = keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    inputs  = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x       = keras.applications.mobilenet_v2.preprocess_input(inputs)  # expects [0,255]
    x       = base(x, training=False)
    x       = layers.GlobalAveragePooling2D()(x)
    x       = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    return keras.Model(inputs, outputs), base


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*55}")
    print("  Hand Gesture Recognition — Training")
    print(f"  Classes: {NUM_CLASSES}   IMG: {IMG_SIZE}x{IMG_SIZE}")
    print(f"{'='*55}\n")

    print("Collecting image paths …")
    paths, labels = collect_paths()
    print(f"\nTotal: {len(paths)} images, {NUM_CLASSES} classes\n")

    train_ds, val_ds = make_datasets(paths, labels)

    model, base = build_model()
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary(line_length=60, print_fn=lambda s: None)

    callbacks = [
        keras.callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True, verbose=1),
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, verbose=1),
    ]

    # Phase 1 — frozen base
    print("── Phase 1: training head (base frozen) ──")
    model.fit(train_ds, validation_data=val_ds, epochs=15, callbacks=callbacks)

    # Phase 2 — unfreeze top 30 layers
    print("\n── Phase 2: fine-tuning top 30 layers ──")
    for layer in base.layers[-30:]:
        layer.trainable = True
    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(train_ds, validation_data=val_ds, epochs=20, callbacks=callbacks)

    # Save label map
    with open(MAP_PATH, "w") as f:
        json.dump({str(k): v for k, v in LABEL_MAP.items()}, f, indent=2)
    print(f"\nLabel map saved → {MAP_PATH}")
    print(f"Model saved     → {MODEL_PATH}")

    # Quick eval
    loss, acc = model.evaluate(val_ds, verbose=0)
    print(f"\nVal accuracy: {acc:.2%}  |  Val loss: {loss:.4f}")
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
