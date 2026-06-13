"""
Train a landmark-based gesture classifier using MediaPipe 0.10 Tasks API.
Extracts 21 hand keypoints from each training image, trains a small MLP.

Usage:
    python train_landmark.py
"""

import os, urllib.request
import numpy as np
import cv2
import joblib
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

DATA_DIR    = os.path.expanduser("~/Downloads/test")
MODEL_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
LM_MODEL    = os.path.join(MODEL_DIR, "hand_landmarker.task")
OUT_MODEL   = os.path.join(MODEL_DIR, "landmark_model.pkl")

LABEL_MAP = {
    0: "C",        1: "Index",     2: "L",         3: "Three",    4: "Four",
    5: "Palm",     6: "OK",        7: "Rock",       8: "ILoveYou", 9: "Spock",
    10: "Down",   11: "PalmSide", 12: "Point",     13: "One",     14: "Fist",
    15: "Grip",   16: "Thumb",    17: "Hang",      18: "PointDown",19: "ThumbUp",
}

_LM_URL = ("https://storage.googleapis.com/mediapipe-models/"
           "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")


def download_model():
    if os.path.exists(LM_MODEL):
        return
    print(f"Downloading hand_landmarker.task …")
    urllib.request.urlretrieve(_LM_URL, LM_MODEL)
    print("  done.\n")


def make_detector():
    opts = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=LM_MODEL),
        num_hands=1,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    return mp_vision.HandLandmarker.create_from_options(opts)


def normalize(lm_list) -> np.ndarray:
    pts = np.array([[l.x, l.y, l.z] for l in lm_list], dtype=np.float32)
    pts -= pts[0]
    scale = np.linalg.norm(pts[9])
    if scale > 1e-6:
        pts /= scale
    return pts.flatten()


def extract(img_bgr: np.ndarray, detector) -> np.ndarray | None:
    h, w = img_bgr.shape[:2]
    if h < 128 or w < 128:
        img_bgr = cv2.resize(img_bgr, (224, 224), interpolation=cv2.INTER_LINEAR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mp_img  = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    result  = detector.detect(mp_img)
    if not result.hand_landmarks:
        return None
    return normalize(result.hand_landmarks[0])


def collect(detector):
    X, y = [], []
    for cls_id, cls_name in LABEL_MAP.items():
        folder = os.path.join(DATA_DIR, str(cls_id))
        if not os.path.isdir(folder):
            print(f"  MISSING: {folder}")
            continue
        files = [f for f in os.listdir(folder)
                 if f.lower().endswith((".jpg", ".png"))]
        found = 0
        for fname in files:
            img = cv2.imread(os.path.join(folder, fname))
            if img is None:
                continue
            lm = extract(img, detector)
            if lm is not None:
                X.append(lm)
                y.append(cls_id)
                found += 1
        pct = found / max(len(files), 1) * 100
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"  {cls_name:12s} [{bar}] {found}/{len(files)} ({pct:.0f}%)")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def main():
    print("\n" + "=" * 55)
    print("  Landmark Gesture Classifier — Training")
    print("=" * 55 + "\n")

    download_model()
    detector = make_detector()

    print("Extracting landmarks …")
    X, y = collect(detector)
    detector.close()

    print(f"\nTotal samples: {len(X)}")
    if len(X) < 100:
        print("ERROR: Too few samples. Check DATA_DIR path.")
        return

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu",
            max_iter=500,
            early_stopping=True,
            n_iter_no_change=20,
            random_state=42,
        )),
    ])

    print("Training MLP …", end=" ", flush=True)
    clf.fit(X_tr, y_tr)
    print("done.")
    print(f"Validation accuracy: {clf.score(X_val, y_val):.2%}")

    joblib.dump(clf, OUT_MODEL)
    print(f"\nSaved → {OUT_MODEL}")
    print("Now commit models/landmark_model.pkl + models/hand_landmarker.task to GitHub.")


if __name__ == "__main__":
    main()
