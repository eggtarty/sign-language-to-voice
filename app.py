import os
import cv2
import av
import time
import numpy as np
import streamlit as st
import mediapipe as mp

from collections import deque
from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

# ==================
# CONFIG & PATHS
# ==================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SEQ_LEN = 30
MIN_DYNAMIC_FRAMES = 20

STATIC_MODEL_PATH = os.path.join(BASE_DIR, "models", "static_model.keras")
DYNAMIC_MODEL_PATH = os.path.join(BASE_DIR, "models", "dynamic_model.keras")
STATIC_LABELS_PATH = os.path.join(BASE_DIR, "models", "labels_static.npy")
DYNAMIC_LABELS_PATH = os.path.join(BASE_DIR, "models", "labels_dynamic.npy")

st.set_page_config(page_title="🤘BWS33203 Artificial Intelligence Group 9", layout="wide")

# ==========================================
# UI STYLE (UNCHANGED)
# ==========================================
st.markdown("""<style>
.stApp { background-color: #D6EAF8; color: #1B2631; }
.prediction-card { background: linear-gradient(135deg,#1a1a2e,#16213e);
color:#f8f9fa;padding:40px;border-radius:20px;border:2px solid #3498DB;
text-align:center;box-shadow:0 12px 24px rgba(0,0,0,.15);margin:20px 0;}
.mode-tag { color:#4fc3f7;font-size:1.3rem;font-weight:600; }
.gesture-output { font-size:4.5rem;font-weight:850;
background:linear-gradient(45deg,#00e676,#00c853);
-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.timer-bar { font-family:Courier New;color:#bbdefb; }
.static-mode { border-color:#FFC107!important; }
.dynamic-mode { border-color:#2196F3!important; }
</style>""", unsafe_allow_html=True)

# ==========================================
# LOAD MODELS
# ==========================================
@st.cache_resource
def load_assets():
    return (
        load_model(STATIC_MODEL_PATH, compile=False),
        load_model(DYNAMIC_MODEL_PATH, compile=False),
        np.load(STATIC_LABELS_PATH, allow_pickle=True),
        np.load(DYNAMIC_LABELS_PATH, allow_pickle=True),
    )

m_static, m_dynamic, le_static, le_dynamic = load_assets()

# ==========================================
# MEDIAPIPE
# ==========================================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ==========================================
# FEATURES
# ==========================================
def get_120_features(coords_seq):
    feats = []
    for frame in coords_seq:
        wrist = frame[0]
        norm = frame - wrist
        scale = np.max(np.linalg.norm(norm, axis=1))
        if scale > 0:
            norm /= scale
        bones = np.array([norm[i] - norm[0] for i in range(1, 21)]).flatten()
        feats.append(bones)

    feats = np.array(feats)
    if len(feats) > 1:
        vel = np.diff(feats, axis=0)
        vel = np.vstack([vel, np.zeros((1, 60))])
    else:
        vel = np.zeros_like(feats)

    return np.concatenate([feats, vel], axis=1)

def calculate_motion_intensity(features, window=3):
    if len(features) < window:
        return 0
    v = features[-window:, 60:]
    return np.mean(np.linalg.norm(v, axis=1))

def pad_sequence(seq, target):
    return seq[-target:] if len(seq) >= target else seq + [seq[-1]] * (target - len(seq))

# ==========================================
# STATE MACHINE (UNCHANGED)
# ==========================================
class GestureDetector:
    def __init__(self):
        self.coord_buffer = deque(maxlen=SEQ_LEN)
        self.last_move = time.time()
        self.last_pred = 0
        self.mode = "Init"
        self.dynamic = False

    def update(self, coords, motion, thresh):
        t = time.time()
        if coords is not None:
            self.coord_buffer.append(coords)

        if motion > thresh:
            self.last_move = t
            if len(self.coord_buffer) >= MIN_DYNAMIC_FRAMES:
                self.dynamic = True

        still = t - self.last_move
        if self.dynamic:
            if still > 0.5:
                self.dynamic = False
                self.mode = "Static"
            else:
                self.mode = "Dynamic"
        elif still > 2:
            self.mode = "Static"
        else:
            self.mode = "Transition"

        return self.mode, still

# ==========================================
# VIDEO PROCESSOR
# ==========================================
class Processor(VideoProcessorBase):
    def __init__(self):
        self.detector = GestureDetector()
        self.last_label = ""

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        label = "Ready"
        motion = 0

        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0]
            coords = np.array([[p.x, p.y, p.z] for p in lm.landmark])
            mp_draw.draw_landmarks(img, lm, mp_hands.HAND_CONNECTIONS)

            buf = list(self.detector.coord_buffer) + [coords]
            feats = get_120_features(buf)
            motion = calculate_motion_intensity(feats)

            mode, _ = self.detector.update(coords, motion, 0.15)

            if mode == "Dynamic" and len(buf) >= MIN_DYNAMIC_FRAMES:
                seq = pad_sequence(buf, SEQ_LEN)
                x = get_120_features(seq).reshape(1, SEQ_LEN, 120)
                p = m_dynamic.predict(x, verbose=0)
                label = le_dynamic[np.argmax(p)]

            elif mode == "Static":
                x = feats[-1].reshape(1, 120)
                p = m_static.predict(x, verbose=0)
                label = le_static[np.argmax(p)]

        cv2.rectangle(img, (0, 0), (640, 50), (0, 0, 0), -1)
        cv2.putText(img, label, (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ==========================================
# UI
# ==========================================
st.title("Communication Friend - AI-BASED Sign Language Translation System")

webrtc_streamer(
    key="sign",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=Processor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

