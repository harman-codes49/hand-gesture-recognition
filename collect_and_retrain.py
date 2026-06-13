"""
Collect webcam gesture photos and retrain the HOG model.

Usage:
    python collect_and_retrain.py

Controls:
    SPACE  — capture current frame as a sample
    S      — skip to next gesture (if gesture is hard to do)
    Q      — quit early
"""

import os, sys, time
import cv2
import numpy as np
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

MODEL_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
DATA_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "webcam_data")
OUT_MODEL   = os.path.join(MODEL_DIR, "hog_model.pkl")
SAMPLES_PER = 20      # photos to collect per gesture
WIN         = 64

LABEL_MAP = {
    0: "C",        1: "Index",     2: "L",         3: "Three",    4: "Four",
    5: "Palm",     6: "OK",        7: "Rock",       8: "ILoveYou", 9: "Spock",
    10: "Down",   11: "PalmSide", 12: "Point",     13: "One",     14: "Fist",
    15: "Grip",   16: "Thumb",    17: "Hang",      18: "PointDown",19: "ThumbUp",
}

HOG = cv2.HOGDescriptor((WIN, WIN), (16, 16), (8, 8), (8, 8), 9)
_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def skin_mask(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    ycrcb   = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
    mask    = cv2.inRange(ycrcb,
                          np.array([0, 133, 77],   dtype=np.uint8),
                          np.array([255, 173, 127], dtype=np.uint8))
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces   = _face_cascade.detectMultiScale(gray, 1.1, 4)
    for (fx, fy, fw, fh) in faces:
        mask[max(0, fy-20):fy+fh+80, max(0, fx-20):fx+fw+20] = 0
    k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
    return mask


def crop_hand(binary, pad=0.15):
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 200:
        return None
    x, y, w, h = cv2.boundingRect(c)
    side = max(w, h);  px = int(side * pad)
    cx, cy = x + w // 2, y + h // 2;  half = side // 2 + px
    x1 = max(cx - half, 0);  y1 = max(cy - half, 0)
    x2 = min(cx + half, binary.shape[1]);  y2 = min(cy + half, binary.shape[0])
    crop = binary[y1:y2, x1:x2]
    return cv2.resize(crop, (WIN, WIN)) if crop.size > 0 else None


def collect():
    os.makedirs(DATA_DIR, exist_ok=True)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    for cls_id, cls_name in LABEL_MAP.items():
        cls_dir = os.path.join(DATA_DIR, str(cls_id))
        os.makedirs(cls_dir, exist_ok=True)
        existing = len([f for f in os.listdir(cls_dir) if f.endswith(".jpg")])
        if existing >= SAMPLES_PER:
            print(f"  {cls_name:12s} — already have {existing} samples, skipping")
            continue

        count = existing
        print(f"\n  ► Gesture: {cls_name}  ({SAMPLES_PER - count} more needed)")
        print("    SPACE = capture  |  S = skip gesture  |  Q = quit")

        while count < SAMPLES_PER:
            ret, frame = cap.read()
            if not ret:
                break
            mask  = skin_mask(frame)
            patch = crop_hand(mask)

            display = frame.copy()
            # Draw mask overlay in green
            overlay      = display.copy()
            overlay[mask > 0] = [0, 255, 100]
            cv2.addWeighted(overlay, 0.3, display, 0.7, 0, display)

            # Status bar
            cv2.rectangle(display, (0, 0), (display.shape[1], 70), (0, 0, 0), -1)
            status = "HAND FOUND" if patch is not None else "NO HAND DETECTED"
            colour = (0, 255, 0)  if patch is not None else (0, 0, 255)
            cv2.putText(display, f"{cls_name}  [{count}/{SAMPLES_PER}]  {status}",
                        (10, 45), cv2.FONT_HERSHEY_DUPLEX, 1.1, colour, 2)

            if patch is not None:
                thumb = cv2.resize(patch, (80, 80))
                display[80:160, display.shape[1]-100:display.shape[1]-20] = \
                    cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)

            cv2.imshow("Gesture Data Collection", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                cap.release(); cv2.destroyAllWindows(); return
            elif key == ord('s'):
                print(f"    Skipped {cls_name}")
                break
            elif key == ord(' ') and patch is not None:
                fname = os.path.join(cls_dir, f"{count:04d}.jpg")
                cv2.imwrite(fname, mask)          # save binary mask (matches training format)
                count += 1
                print(f"    Captured {count}/{SAMPLES_PER}")
                time.sleep(0.1)

    cap.release()
    cv2.destroyAllWindows()
    print("\nCollection complete.")


def extract_hog(img_bgr):
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    patch = crop_hand(bw)
    if patch is None:
        return None
    return HOG.compute(patch).flatten()


def retrain():
    print("\nExtracting HOG features from collected data …")
    X, y = [], []
    for cls_id in LABEL_MAP:
        folder = os.path.join(DATA_DIR, str(cls_id))
        if not os.path.isdir(folder):
            continue
        files  = [f for f in os.listdir(folder) if f.endswith(".jpg")]
        for fname in files:
            img  = cv2.imread(os.path.join(folder, fname))
            feat = extract_hog(img)
            if feat is not None:
                X.append(feat); y.append(cls_id)
        print(f"  {LABEL_MAP[cls_id]:12s}: {len(files)} samples")

    print(f"\nTotal: {len(X)} samples")
    X_tr, X_val, y_tr, y_val = train_test_split(
        np.array(X), np.array(y), test_size=0.2, stratify=y, random_state=42
    )
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(hidden_layer_sizes=(256, 128, 64),
                               max_iter=500, early_stopping=True, random_state=42)),
    ])
    print("Training … ", end="", flush=True)
    clf.fit(X_tr, y_tr)
    print("done.")
    print(f"Val accuracy: {clf.score(X_val, y_val):.2%}")
    joblib.dump(clf, OUT_MODEL)
    print(f"\nNew model saved → {OUT_MODEL}")
    print("Now run:  git add models/hog_model.pkl && git commit -m 'retrain on webcam data' && git push")


if __name__ == "__main__":
    print("=" * 55)
    print("  Webcam Gesture Data Collection")
    print(f"  Collecting {SAMPLES_PER} samples × {len(LABEL_MAP)} gestures")
    print("=" * 55)
    collect()
    retrain()
