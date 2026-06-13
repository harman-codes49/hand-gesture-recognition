"""Data loading and preprocessing utilities for LeapGestRecog dataset."""

import os
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from config import IMG_SIZE, GESTURE_FOLDER_MAP, VAL_SPLIT, TEST_SPLIT


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def discover_classes(data_dir: str) -> dict:
    """
    Walk the LeapGestRecog directory and return an ordered {folder_name: class_idx} map.
    Supports the original hierarchy: data_dir/subject_id/gesture_folder/images.
    Falls back to flat structure: data_dir/gesture_folder/images.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    subdirs = sorted([d for d in data_path.iterdir() if d.is_dir()])
    if not subdirs:
        raise ValueError(f"No subdirectories found in {data_dir}")

    # Check if this looks like subject-level hierarchy
    first_sub = subdirs[0]
    children = sorted([d for d in first_sub.iterdir() if d.is_dir()])

    if children:
        # children of first subject are gesture folders
        gesture_folders = [c.name for c in children]
    else:
        # flat structure
        gesture_folders = [d.name for d in subdirs]

    label_map = {folder: idx for idx, folder in enumerate(gesture_folders)}
    return label_map


def load_dataset(data_dir: str, img_size: int = IMG_SIZE, label_map: dict = None):
    """
    Load all images and labels from LeapGestRecog (or any similarly structured dataset).

    Returns:
        images  : np.ndarray (N, H, W, 3) uint8
        labels  : np.ndarray (N,) int
        label_map: dict  folder_name -> class_index
        idx_to_name: dict  class_index -> display_name
    """
    data_path = Path(data_dir)
    subdirs   = sorted([d for d in data_path.iterdir() if d.is_dir()])

    # Decide depth
    first_sub     = subdirs[0]
    first_children = sorted([d for d in first_sub.iterdir() if d.is_dir()])
    is_hierarchical = bool(first_children)

    if label_map is None:
        if is_hierarchical:
            gesture_folders = [c.name for c in first_children]
        else:
            gesture_folders = [d.name for d in subdirs]
        label_map = {folder: idx for idx, folder in enumerate(gesture_folders)}

    # Build human-readable name from folder (use GESTURE_FOLDER_MAP if match found)
    idx_to_name = {}
    for folder, idx in label_map.items():
        display = GESTURE_FOLDER_MAP.get(folder, folder.split("_", 1)[-1].replace("_", " ").title())
        idx_to_name[idx] = display

    images, labels = [], []

    subject_dirs = subdirs if is_hierarchical else [data_path]
    for subject_dir in tqdm(subject_dirs, desc="Loading subjects"):
        gesture_dirs = sorted([d for d in subject_dir.iterdir() if d.is_dir()])
        for gesture_dir in gesture_dirs:
            cls_idx = label_map.get(gesture_dir.name)
            if cls_idx is None:
                continue
            for img_path in gesture_dir.glob("*.png"):
                img = _load_image(str(img_path), img_size)
                if img is not None:
                    images.append(img)
                    labels.append(cls_idx)
            for img_path in gesture_dir.glob("*.jpg"):
                img = _load_image(str(img_path), img_size)
                if img is not None:
                    images.append(img)
                    labels.append(cls_idx)

    return np.array(images, dtype=np.uint8), np.array(labels, dtype=np.int32), label_map, idx_to_name


def _load_image(path: str, size: int) -> np.ndarray | None:
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size))
    return img


# ---------------------------------------------------------------------------
# Splitting and tf.data pipeline
# ---------------------------------------------------------------------------

def split_dataset(images: np.ndarray, labels: np.ndarray):
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        images, labels, test_size=TEST_SPLIT, random_state=42, stratify=labels
    )
    adjusted_val = VAL_SPLIT / (1.0 - TEST_SPLIT)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=adjusted_val, random_state=42, stratify=y_tmp
    )
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def make_tf_datasets(X_train, y_train, X_val, y_val, num_classes: int, batch_size: int):
    def augment(image, label):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, 0.2)
        image = tf.image.random_contrast(image, 0.8, 1.2)
        image = tf.image.random_saturation(image, 0.8, 1.2)
        image = tf.image.random_hue(image, 0.05)
        # Random rotation via crop-and-pad trick
        image = tf.image.resize_with_crop_or_pad(image, IMG_SIZE + 20, IMG_SIZE + 20)
        image = tf.image.random_crop(image, [IMG_SIZE, IMG_SIZE, 3])
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image, label

    y_train_oh = tf.keras.utils.to_categorical(y_train, num_classes).astype(np.float32)
    y_val_oh   = tf.keras.utils.to_categorical(y_val,   num_classes).astype(np.float32)

    X_train_f = X_train.astype(np.float32) / 255.0
    X_val_f   = X_val.astype(np.float32)   / 255.0

    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_train_f, y_train_oh))
        .shuffle(buffer_size=min(5000, len(X_train)))
        .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((X_val_f, y_val_oh))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    return train_ds, val_ds


# ---------------------------------------------------------------------------
# Inference preprocessing
# ---------------------------------------------------------------------------

def preprocess_frame(frame: np.ndarray, img_size: int = IMG_SIZE) -> np.ndarray:
    """
    Prepare a BGR/RGB webcam frame or uploaded image for model inference.
    Returns a float32 array of shape (1, img_size, img_size, 3).
    """
    img = cv2.resize(frame, (img_size, img_size))
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_label_map(idx_to_name: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    serializable = {str(k): v for k, v in idx_to_name.items()}
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)


def load_label_map(path: str) -> dict:
    with open(path) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}
