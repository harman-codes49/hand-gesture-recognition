"""
Model evaluation: confusion matrix, classification report, per-class accuracy.

Usage:
    python src/evaluate.py
"""

import os
import sys
import json

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DATA_DIR, MODEL_PATH, LABEL_MAP_PATH, BATCH_SIZE
from src.utils import load_dataset, split_dataset, load_label_map


def plot_confusion_matrix(cm: np.ndarray, class_names: list, save_path: str):
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True",      fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved → {save_path}")


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}. Run src/train.py first.")
        return

    idx_to_name = load_label_map(LABEL_MAP_PATH)
    num_classes  = len(idx_to_name)
    class_names  = [idx_to_name[i] for i in range(num_classes)]

    print("Loading dataset …")
    images, labels, _, _ = load_dataset(DATA_DIR)
    _, _, (X_test, y_test) = split_dataset(images, labels)
    X_test_f = X_test.astype(np.float32) / 255.0

    print("Loading model …")
    model = tf.keras.models.load_model(MODEL_PATH)

    print("Running predictions …")
    preds = model.predict(X_test_f, batch_size=BATCH_SIZE, verbose=1)
    y_pred = np.argmax(preds, axis=1)

    # --- Metrics ---
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, class_names, "models/confusion_matrix.png")

    # Per-class accuracy
    print("\nPer-class Accuracy:")
    for i, name in enumerate(class_names):
        mask = y_test == i
        if mask.sum() > 0:
            acc = (y_pred[mask] == i).mean()
            print(f"  {name:15s}: {acc * 100:.1f}%")


if __name__ == "__main__":
    main()
