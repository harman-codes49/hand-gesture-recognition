"""
Training pipeline for hand gesture recognition.

Usage:
    python src/train.py

Dataset expected at data/leapGestRecog/ (download with download_dataset.py).
Trained model saved to models/gesture_model.h5.
"""

import os
import sys
import json
import time

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    DATA_DIR, MODEL_PATH, LABEL_MAP_PATH, HISTORY_PATH,
    IMG_SIZE, BATCH_SIZE, EPOCHS_FROZEN, EPOCHS_FINETUNE,
    LEARNING_RATE, FINETUNE_LR, UNFREEZE_LAYERS, NUM_CLASSES,
)
from src.model import build_gesture_model, unfreeze_top_layers, compile_model
from src.utils import load_dataset, split_dataset, make_tf_datasets, save_label_map


def callbacks_for(phase: str, model_path: str):
    return [
        tf.keras.callbacks.ModelCheckpoint(
            model_path, monitor="val_accuracy", save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=7, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=f"logs/{phase}_{int(time.time())}", histogram_freq=1
        ),
    ]


def plot_history(h1, h2, save_path: str = "models/training_curves.png"):
    acc  = h1.history["accuracy"]       + h2.history["accuracy"]
    val  = h1.history["val_accuracy"]   + h2.history["val_accuracy"]
    loss = h1.history["loss"]           + h2.history["loss"]
    vloss= h1.history["val_loss"]       + h2.history["val_loss"]
    ep   = range(1, len(acc) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(ep, acc, "b-",  label="Train Accuracy")
    ax1.plot(ep, val, "r--", label="Val Accuracy")
    ax1.axvline(len(h1.history["accuracy"]), color="grey", linestyle=":", label="Fine-tune start")
    ax1.set_title("Accuracy"); ax1.legend(); ax1.set_xlabel("Epoch")

    ax2.plot(ep, loss,  "b-",  label="Train Loss")
    ax2.plot(ep, vloss, "r--", label="Val Loss")
    ax2.axvline(len(h1.history["loss"]), color="grey", linestyle=":", label="Fine-tune start")
    ax2.set_title("Loss"); ax2.legend(); ax2.set_xlabel("Epoch")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training curves saved to {save_path}")


def save_history(h1, h2, path: str):
    combined = {}
    for key in h1.history:
        combined[key] = h1.history[key] + h2.history[key]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(combined, f, indent=2)


def main():
    # --- GPU memory growth ---
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    print("=" * 60)
    print("Loading dataset …")
    images, labels, label_map, idx_to_name = load_dataset(DATA_DIR)
    num_classes = len(idx_to_name)
    print(f"  Loaded {len(images)} images across {num_classes} classes")
    print(f"  Classes: {list(idx_to_name.values())}")

    save_label_map(idx_to_name, LABEL_MAP_PATH)
    print(f"  Label map saved → {LABEL_MAP_PATH}")

    # --- Split ---
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_dataset(images, labels)
    print(f"  Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

    train_ds, val_ds = make_tf_datasets(X_train, y_train, X_val, y_val, num_classes, BATCH_SIZE)

    # --- Phase 1: Train top layers ---
    print("\n" + "=" * 60)
    print("Phase 1 – Training classification head (base frozen) …")
    model = build_gesture_model(num_classes=num_classes, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    model = compile_model(model, LEARNING_RATE, num_classes)
    model.summary(line_length=90)

    history1 = model.fit(
        train_ds, validation_data=val_ds,
        epochs=EPOCHS_FROZEN,
        callbacks=callbacks_for("frozen", MODEL_PATH),
    )

    # --- Phase 2: Fine-tune ---
    print("\n" + "=" * 60)
    print(f"Phase 2 – Fine-tuning top {UNFREEZE_LAYERS} base layers …")
    model = tf.keras.models.load_model(MODEL_PATH)        # best weights from phase 1
    model = unfreeze_top_layers(model, UNFREEZE_LAYERS)
    model = compile_model(model, FINETUNE_LR, num_classes)

    history2 = model.fit(
        train_ds, validation_data=val_ds,
        epochs=EPOCHS_FINETUNE,
        callbacks=callbacks_for("finetune", MODEL_PATH),
    )

    # --- Save artifacts ---
    model.save(MODEL_PATH)
    print(f"\nModel saved → {MODEL_PATH}")
    plot_history(history1, history2)
    save_history(history1, history2, HISTORY_PATH)

    # --- Quick test-set evaluation ---
    print("\n" + "=" * 60)
    print("Evaluating on held-out test set …")
    X_test_f = X_test.astype(np.float32) / 255.0
    y_test_oh = tf.keras.utils.to_categorical(y_test, num_classes)
    loss, acc = model.evaluate(X_test_f, y_test_oh, batch_size=BATCH_SIZE, verbose=0)
    print(f"  Test accuracy : {acc * 100:.2f}%")
    print(f"  Test loss     : {loss:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
