import copy, itertools, json, os, sys, urllib.request
from threading import Lock

import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

try:
    import av
    from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, webrtc_streamer
    _WEBRTC_OK = True
    _WEBRTC_ERR = ""
except Exception as e:
    av = None  # type: ignore
    _WEBRTC_OK = False
    _WEBRTC_ERR = str(e)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import LABEL_MAP_PATH, GESTURE_EMOJIS

_BASE      = os.path.dirname(os.path.abspath(__file__))
_LM_MODEL  = os.path.join(_BASE, "models", "hand_landmarker.task")
_CLF_PATH  = os.path.join(_BASE, "models", "keypoint_model.pkl")
_LM_URL    = ("https://storage.googleapis.com/mediapipe-models/"
              "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")

GESTURE_LABELS = {0: "Open", 1: "Close", 2: "Pointer", 3: "OK"}
GESTURE_EMOJIS_LOCAL = {
    "Open": "✋", "Close": "✊", "Pointer": "☝️", "OK": "👌"
}

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hand Gesture Recognition",
    page_icon="✋",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.html("""
<style>
.pred-box {
    background: linear-gradient(135deg,#1e3a5f,#0d2137);
    border: 1px solid #2e6da4; border-radius:12px;
    padding:20px; text-align:center; margin:10px 0;
}
.g-emoji { font-size:64px; }
.g-name  { font-size:28px; font-weight:bold; color:#4fc3f7; }
.g-conf  { font-size:16px; color:#90caf9; margin-top:4px; }
</style>
""")


# ── MediaPipe detector ─────────────────────────────────────────────────────────

def _ensure_lm_model():
    if not os.path.exists(_LM_MODEL):
        st.info("Downloading hand landmark model (7 MB) …")
        urllib.request.urlretrieve(_LM_URL, _LM_MODEL)


@st.cache_resource(show_spinner="Loading MediaPipe hand detector …")
def load_detector():
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
        _ensure_lm_model()
        opts = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=_LM_MODEL),
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        return mp_vision.HandLandmarker.create_from_options(opts), mp
    except Exception as e:
        st.error(f"MediaPipe load failed: {e}")
        return None, None


@st.cache_resource(show_spinner="Loading gesture classifier …")
def load_classifier():
    try:
        import joblib
        return joblib.load(_CLF_PATH)
    except Exception as e:
        st.error(f"Classifier load failed: {e}")
        return None


# ── Landmark normalization ────────────────────────────────────────────────────

def pre_process_landmark(landmark_list):
    temp = copy.deepcopy(landmark_list)
    base_x, base_y = temp[0][0], temp[0][1]
    for i in range(len(temp)):
        temp[i][0] -= base_x
        temp[i][1] -= base_y
    flat = list(itertools.chain.from_iterable(temp))
    max_val = max(map(abs, flat))
    return [v / max_val for v in flat] if max_val > 0 else flat


def get_landmarks(img_rgb, detector, mp_mod):
    """Returns (landmark_list [[x,y]×21], hand_rect, handedness_str) or (None, None, None)."""
    mp_img = mp_mod.Image(image_format=mp_mod.ImageFormat.SRGB, data=img_rgb)
    result = detector.detect(mp_img)
    if not result.hand_landmarks:
        return None, None, None
    h, w = img_rgb.shape[:2]
    lm  = result.hand_landmarks[0]
    pts = [[int(l.x * w), int(l.y * h)] for l in lm]
    arr = np.array(pts)
    x, y, bw, bh = cv2.boundingRect(arr)
    # Handedness: "Left" or "Right"
    handedness = "Right"
    if result.handedness and result.handedness[0]:
        handedness = result.handedness[0][0].display_name
    return pts, (x, y, x + bw, y + bh), handedness


def predict_from_landmarks(pts, clf):
    feat = pre_process_landmark(copy.deepcopy(pts))
    probs = clf.predict_proba([feat])[0]
    idx = int(np.argmax(probs))
    return GESTURE_LABELS.get(idx, str(idx)), float(probs[idx]), probs


# ── Drawing helpers ───────────────────────────────────────────────────────────

_CONNECTIONS = [
    (2,3),(3,4),          # thumb
    (5,6),(6,7),(7,8),    # index
    (9,10),(10,11),(11,12),  # middle
    (13,14),(14,15),(15,16), # ring
    (17,18),(18,19),(19,20), # pinky
    (0,1),(1,2),(2,5),(5,9),(9,13),(13,17),(17,0),  # palm
]

def draw_landmarks(frame, pts):
    for a, b in _CONNECTIONS:
        cv2.line(frame, tuple(pts[a]), tuple(pts[b]), (0,0,0), 5)
        cv2.line(frame, tuple(pts[a]), tuple(pts[b]), (255,255,255), 2)
    for i, p in enumerate(pts):
        r = 8 if i in (4,8,12,16,20) else 5
        cv2.circle(frame, tuple(p), r, (255,255,255), -1)
        cv2.circle(frame, tuple(p), r, (0,0,0), 1)
    return frame


def draw_label(frame, rect, name, conf, handedness="Right"):
    x1, y1, x2, y2 = rect
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,0), 1)
    cv2.rectangle(frame, (x1, y1-26), (x2, y1), (0,0,0), -1)
    side = handedness[0]  # "L" or "R"
    cv2.putText(frame, f"{side}: {name}  {conf:.0%}",
                (x1+5, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (255,255,255), 1, cv2.LINE_AA)
    return frame


def process_frame(img_rgb, detector, mp_mod, clf):
    pts, rect, handedness = get_landmarks(img_rgb, detector, mp_mod)
    if pts is None:
        return None, 0.0, None, None, None
    name, conf, probs = predict_from_landmarks(pts, clf)
    return name, conf, probs, (pts, rect), handedness


# ── UI helpers ─────────────────────────────────────────────────────────────────

def conf_chart(probs):
    names  = list(GESTURE_LABELS.values())
    hi     = int(np.argmax(probs))
    colors = ["#FF4B4B" if i == hi else "#4B8BFF" for i in range(len(probs))]
    fig = go.Figure(go.Bar(
        x=names, y=(probs * 100).tolist(), marker_color=colors,
        text=[f"{p:.1f}%" for p in probs * 100], textposition="outside",
    ))
    fig.update_layout(
        title="Confidence per Gesture",
        yaxis=dict(title="Confidence (%)", range=[0, 115]),
        height=300, margin=dict(l=20, r=20, t=40, b=40),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#ffffff",
    )
    return fig


def prediction_card(name, confidence, probs, threshold=0.5):
    if name is None:
        st.warning("No hand detected — make sure your hand is clearly visible.")
        return
    emoji = GESTURE_EMOJIS_LOCAL.get(name, "✋")
    if confidence < threshold:
        st.warning(f"Low confidence — best guess: **{name}** ({confidence:.1%})")
        return
    st.html(
        f'<div class="pred-box">'
        f'<div class="g-emoji">{emoji}</div>'
        f'<div class="g-name">{name}</div>'
        f'<div class="g-conf">{confidence:.1%} confidence</div>'
        f'</div>'
    )
    st.plotly_chart(conf_chart(probs), use_container_width=True)


# ── WebRTC video processor ─────────────────────────────────────────────────────

class GestureProcessor(VideoProcessorBase if _WEBRTC_OK else object):
    def __init__(self):
        self._lock      = Lock()
        self._detector, self._mp = load_detector()
        self._clf       = load_classifier()

    def recv(self, frame):
        img_bgr = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        if self._detector and self._clf:
            name, conf, _, extra, handedness = process_frame(
                img_rgb, self._detector, self._mp, self._clf)
            if extra:
                pts, rect = extra
                draw_landmarks(img_bgr, pts)
                draw_label(img_bgr, rect, name, conf, handedness or "Right")
            else:
                cv2.putText(img_bgr, "Show your hand",
                            (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                            (200,200,200), 2)
        return av.VideoFrame.from_ndarray(img_bgr, format="bgr24")


# ── Sidebar ────────────────────────────────────────────────────────────────────

def sidebar():
    with st.sidebar:
        st.title("✋ Hand Gesture Recognition")
        st.divider()
        st.subheader("🖐️ Gesture Classes")
        for name, emoji in GESTURE_EMOJIS_LOCAL.items():
            st.write(f"{emoji} {name}")


# ── Tabs ───────────────────────────────────────────────────────────────────────

def tab_live():
    st.subheader("🎥 Real-time Hand Gesture Recognition")
    if not _WEBRTC_OK:
        st.warning(f"Live camera unavailable: {_WEBRTC_ERR}")
        st.info("Use the **📸 Camera Snapshot** tab instead.")
        return
    st.info("Click **START**, allow camera access, then show your hand.")
    webrtc_streamer(
        key="gesture-live",
        video_processor_factory=GestureProcessor,
        rtc_configuration=RTCConfiguration({"iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
        ]}),
        media_stream_constraints={
            "video": {"width": {"ideal": 1280}, "height": {"ideal": 720},
                      "frameRate": {"ideal": 30}, "facingMode": "user"},
            "audio": False,
        },
        async_processing=True,
    )
    st.caption("💡 Good lighting, plain background, hand centred in frame.")


def _predict_and_show(img_rgb, detector, mp_mod, clf):
    name, conf, probs, extra, handedness = process_frame(img_rgb, detector, mp_mod, clf)
    if extra:
        pts, rect = extra
        annotated = cv2.cvtColor(img_rgb.copy(), cv2.COLOR_RGB2BGR)
        draw_landmarks(annotated, pts)
        draw_label(annotated, rect, name, conf, handedness or "Right")
        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        st.image(annotated, use_container_width=True)
    if handedness:
        st.caption(f"🖐️ **{handedness} hand** detected")
    prediction_card(name, conf, probs)


def tab_snapshot(detector, mp_mod, clf):
    st.subheader("📸 Camera Snapshot")
    st.caption("💡 Hold hand clearly in centre of frame.")
    img_file = st.camera_input("Take a photo of your hand gesture")
    if img_file:
        img_rgb = np.array(Image.open(img_file).convert("RGB"))
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_rgb, use_container_width=True)
        with col2:
            if detector and clf:
                _predict_and_show(img_rgb, detector, mp_mod, clf)
            else:
                st.error("Model not loaded.")


def tab_upload(detector, mp_mod, clf):
    st.subheader("🖼️ Upload an Image")
    uploaded = st.file_uploader("Choose an image", type=["jpg","jpeg","png","bmp","webp"])
    if uploaded:
        img_rgb = np.array(Image.open(uploaded).convert("RGB"))
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_rgb, use_container_width=True)
        with col2:
            if detector and clf:
                _predict_and_show(img_rgb, detector, mp_mod, clf)
            else:
                st.error("Model not loaded.")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    detector, mp_mod = load_detector()
    clf = load_classifier()

    sidebar()
    st.title("✋ Hand Gesture Recognition")
    st.divider()

    tab1, tab2, tab3 = st.tabs(["🎥 Live Camera", "📸 Camera Snapshot", "🖼️ Upload Image"])
    with tab1: tab_live()
    with tab2: tab_snapshot(detector, mp_mod, clf)
    with tab3: tab_upload(detector, mp_mod, clf)


if __name__ == "__main__":
    main()
