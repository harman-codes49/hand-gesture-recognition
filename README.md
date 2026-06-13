# ✋ Hand Gesture Recognition

Real-time hand gesture recognition using **MobileNetV2 transfer learning** and **MediaPipe Hands**, deployed as a **Streamlit** web application.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-orange)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)](https://streamlit.io)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10%2B-green)](https://mediapipe.dev)

---

## 🎯 Features

| Feature | Description |
|---|---|
| 🎥 **Live Camera** | Real-time gesture recognition via WebRTC (streamlit-webrtc) |
| 📸 **Camera Snapshot** | Take a single photo and classify |
| 🖼️ **Image Upload** | Upload any JPG/PNG and get predictions |
| 🦴 **Hand Landmarks** | MediaPipe overlays 21 hand keypoints |
| 📊 **Confidence Chart** | Plotly bar chart of all gesture probabilities |
| ⚙️ **Adjustable Threshold** | Filter low-confidence predictions via sidebar |

---

## 🗂️ Project Structure

```
hand gesture recognition/
├── app.py                  # Streamlit application (main entry point)
├── config.py               # All project-wide constants and paths
├── download_dataset.py     # Kaggle dataset downloader
├── setup.sh                # One-shot environment setup
├── requirements.txt        # Python dependencies
│
├── src/
│   ├── model.py            # MobileNetV2 model definition
│   ├── train.py            # Two-phase training pipeline
│   ├── utils.py            # Data loading, preprocessing, tf.data pipeline
│   └── evaluate.py         # Confusion matrix, classification report
│
├── models/
│   ├── gesture_model.h5    # Trained model (generated after training)
│   └── label_map.json      # Class index → gesture name mapping
│
└── data/
    └── leapGestRecog/      # Dataset (downloaded, not committed to git)
```

---

## 🚀 Quick Start

### 1 · Clone the repository

```bash
git clone https://github.com/<your-username>/hand-gesture-recognition.git
cd hand-gesture-recognition
```

### 2 · One-shot setup

```bash
bash setup.sh
source .venv/bin/activate
```

Or manually:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3 · Download the dataset

**Set up Kaggle credentials first:**

1. Go to [kaggle.com/settings](https://www.kaggle.com/settings) → **API** → **Create New Token**
2. This downloads `kaggle.json`. Place it at `~/.kaggle/kaggle.json`
3. `chmod 600 ~/.kaggle/kaggle.json`  *(Linux/Mac)*

Then download:

```bash
python download_dataset.py
```

This fetches the **LeapGestRecog** dataset (~20,000 infrared hand images, 10 gesture classes).

### 4 · Train the model

```bash
python src/train.py
```

Training runs in two phases:
- **Phase 1** (20 epochs): only the classification head is trained; MobileNetV2 base frozen
- **Phase 2** (30 epochs): top 30 base layers are unfrozen for fine-tuning

Saved to `models/gesture_model.h5`. Expected accuracy: **~97–99%** on this dataset.

> ⏱️ Time: ~10 min on GPU, ~45 min on CPU.

### 5 · (Optional) Evaluate

```bash
python src/evaluate.py
```

Prints per-class accuracy, classification report, and saves `models/confusion_matrix.png`.

### 6 · Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🖐️ Gesture Classes

| # | Folder | Gesture | Emoji |
|---|--------|---------|-------|
| 0 | `01_palm` | Palm | ✋ |
| 1 | `02_l` | L | 🤙 |
| 2 | `03_fist` | Fist | ✊ |
| 3 | `04_fist_moved` | Fist Moved | ✊ |
| 4 | `05_thumb` | Thumb | 👍 |
| 5 | `06_index` | Index | ☝️ |
| 6 | `07_ok` | OK | 👌 |
| 7 | `08_palm_moved` | Palm Moved | ✋ |
| 8 | `09_c` | C | 🤏 |
| 9 | `10_down` | Down | 👇 |

---

## 🏗️ Model Architecture

```
Input (224×224×3)
      │
MobileNetV2 (ImageNet pretrained)  ← Phase 1: frozen; Phase 2: top 30 layers unfrozen
      │
GlobalAveragePooling2D
BatchNormalization
Dense(512, relu) + Dropout(0.4)
Dense(256, relu) + Dropout(0.3)
Dense(10, softmax)
```

---

## ☁️ Deployment on Streamlit Cloud

1. Push your repo to GitHub (the trained `models/gesture_model.h5` should be committed — it's ~14 MB).  
   If it exceeds GitHub's 100 MB limit, use [Git LFS](https://git-lfs.github.com/).

2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** → select your repo → `app.py`.

3. Add a `packages.txt` for system-level dependencies (if needed):
   ```
   libgl1-mesa-glx
   libglib2.0-0
   ```

4. The app will auto-deploy. For live camera (WebRTC) to work through Streamlit Cloud, a TURN server may be needed for some networks — configure via Twilio or another provider and pass credentials via Streamlit Secrets.

---

## 📝 Important Notes

- **Domain gap**: LeapGestRecog images are infrared (high-contrast, dark background). For best webcam accuracy, ensure good lighting and a plain background.
- **Model file**: `models/gesture_model.h5` is in `.gitignore` by default. After training, you can either commit it (if <100 MB) or host it externally and add a download step.
- To use a different Kaggle dataset, update `DATA_DIR` in `config.py` and adjust `GESTURE_FOLDER_MAP` to match your folder names.

---

## 📦 Key Dependencies

| Package | Purpose |
|---|---|
| `tensorflow` | Model training & inference |
| `mediapipe` | Real-time hand landmark detection |
| `streamlit` | Web application framework |
| `streamlit-webrtc` | Live camera via WebRTC |
| `opencv-python-headless` | Image processing |
| `plotly` | Interactive confidence charts |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
