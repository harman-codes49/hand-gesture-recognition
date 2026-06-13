"""
Train a HOG + MLP gesture classifier.

Training images are already binary (white hand on black background).
At inference we apply skin-colour detection to webcam frames, extract the
hand contour, then compute the same HOG descriptor — giving features
that are directly comparable to the training distribution.

Usage:
    python train_hog.py
"""

import os
import numpy as np
import cv2
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

DATA_DIR  = os.path.expanduser("~/Downloads/test")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
OUT_MODEL = os.path.join(MODEL_DIR, "hog_model.pkl")

LABEL_MAP = {
    0: "C",        1: "Index",     2: "L",         3: "Three",    4: "Four",
    5: "Palm",     6: "OK",        7: "Rock",       8: "ILoveYou", 9: "Spock",
    10: "Down",   11: "PalmSide", 12: "Point",     13: "One",     14: "Fist",
    15: "Grip",   16: "Thumb",    17: "Hang",      18: "PointDown",19: "ThumbUp",
}

WIN = 64   # HOG window size

HOG = cv2.HOGDescriptor(
    (WIN, WIN),   # winSize
    (16, 16),     # blockSize
    (8, 8),       # blockStride
    (8, 8),       # cellSize
    9,            # nbins
)


def crop_hand(binary: np.ndarray, pad: float = 0.15) -> np.ndarray | None:
    """Crop tightly to the hand region and return WIN×WIN patch."""
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 50:
        return None
    x, y, w, h = cv2.boundingRect(c)
    # Square bounding box with padding
    side = max(w, h)
    px   = int(side * pad)
    cx, cy = x + w // 2, y + h // 2
    half = side // 2 + px
    x1 = max(cx - half, 0)
    y1 = max(cy - half, 0)
    x2 = min(cx + half, binary.shape[1])
    y2 = min(cy + half, binary.shape[0])
    crop = binary[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (WIN, WIN))


def hog_features(patch: np.ndarray) -> np.ndarray:
    return HOG.compute(patch).flatten()


# ── Training ───────────────────────────────────────────────────────────────────

def extract_training(img_bgr: np.ndarray) -> np.ndarray | None:
    """Training images are already binary — just crop and compute HOG."""
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # Threshold to clean binary (images may have mid values from JPEG compression)
    _, bw = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    patch = crop_hand(bw)
    if patch is None:
        return None
    return hog_features(patch)


def collect():
    X, y = [], []
    for cls_id, cls_name in LABEL_MAP.items():
        folder = os.path.join(DATA_DIR, str(cls_id))
        if not os.path.isdir(folder):
            continue
        files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".png"))]
        found = 0
        for fname in files:
            img = cv2.imread(os.path.join(folder, fname))
            if img is None:
                continue
            feat = extract_training(img)
            if feat is not None:
                X.append(feat)
                y.append(cls_id)
                found += 1
        pct = found / max(len(files), 1) * 100
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"  {cls_name:12s} [{bar}] {found}/{len(files)} ({pct:.0f}%)")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def main():
    print("\n" + "=" * 55)
    print("  HOG + MLP Gesture Classifier — Training")
    print("=" * 55 + "\n")

    print("Extracting HOG features from training images …")
    X, y = collect()
    print(f"\nTotal samples: {len(X)}")

    if len(X) < 200:
        print("ERROR: Too few samples extracted.")
        return

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(512, 256, 128),
            activation="relu",
            max_iter=500,
            early_stopping=True,
            n_iter_no_change=20,
            random_state=42,
        )),
    ])

    print("\nTraining … ", end="", flush=True)
    clf.fit(X_tr, y_tr)
    print("done.")
    print(f"Validation accuracy: {clf.score(X_val, y_val):.2%}")

    joblib.dump(clf, OUT_MODEL)
    print(f"\nSaved → {OUT_MODEL}")
    print("Commit models/hog_model.pkl to GitHub.")


if __name__ == "__main__":
    main()
