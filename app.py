import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import pyttsx3
import threading
import tempfile
import time

# ================================
# CONFIG
# ================================
SEQ_LEN = 30
FEATURE_DIM = 120
PRED_FREQ = 3          # predict every 3 frames
COOLDOWN = 1.5         # seconds before TTS repeats
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# ================================
# LOAD MODELS
# ================================
# One-hand gestures
one_static_model = load_model(r"C:\Users\minya\Documents\U\Year 3\Year 3 Sem 1\Artificial Intelligence\project 2\Training one hand\static_model.keras")
one_dynamic_model = load_model(r"C:\Users\minya\Documents\U\Year 3\Year 3 Sem 1\Artificial Intelligence\project 2\Training one hand\dynamic_model.keras")
one_labels_static = np.load(r"C:\Users\minya\Documents\U\Year 3\Year 3 Sem 1\Artificial Intelligence\project 2\Training one hand\labels_static.npy", allow_pickle=True)
one_labels_dynamic = np.load(r"C:\Users\minya\Documents\U\Year 3\Year 3 Sem 1\Artificial Intelligence\project 2\Training one hand\labels_dynamic.npy", allow_pickle=True)

# Two-hand gestures (dynamic only)
two_dynamic_model = load_model(r"C:\Users\minya\Documents\U\Year 3\Year 3 Sem 1\Artificial Intelligence\project 2\two_hand_gesture_model.h5")
two_labels_dynamic = ["how are you","im fine"]

# ================================
# TTS SETUP
# ================================
engine = pyttsx3.init()
engine.setProperty('rate', 150)
last_spoken = ""
last_time = 0

def speak(text):
    global last_spoken, last_time
    now = time.time()
    if text != last_spoken or (now - last_time) > COOLDOWN:
        last_spoken = text
        last_time = now
        threading.Thread(target=lambda: engine.say(text) or engine.runAndWait(), daemon=True).start()

# ================================
# FEATURE FUNCTIONS
# ================================
def extract_features(sequence):
    features = []
    for t in range(sequence.shape[0]):
        frame = sequence[t]
        wrist = frame[0]
        frame = frame - wrist
        scale = np.max(np.linalg.norm(frame, axis=1))
        if scale > 0:
            frame /= scale
        bones = np.array([frame[i]-frame[0] for i in range(1,21)]).flatten()
        features.append(bones)
    features = np.array(features)
    if features.shape[0] < 2:
        velocity = np.zeros_like(features)
    else:
        velocity = np.diff(features, axis=0)
        velocity = np.vstack([velocity, np.zeros((1, velocity.shape[1]))])
    return np.concatenate([features, velocity], axis=1)

def pad_or_trim(seq):
    if len(seq) < SEQ_LEN:
        pad = np.repeat(seq[-1][None,:], SEQ_LEN-len(seq), axis=0)
        seq = np.vstack([seq,pad])
    return seq[:SEQ_LEN]

# ================================
# STREAMLIT UI
# ================================
st.title("Sign Language to Voice (One-hand & Two-hand)")
mode = st.radio("Choose mode", ["Upload video", "Use webcam"])

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# ================================
# PREDICTION FUNCTION
# ================================
def predict_gesture(landmarks, frame_buffer, hand_count):
    feat = extract_features(landmarks)
    frame_buffer.append(feat)
    if len(frame_buffer) < SEQ_LEN:
        return None, None  # wait for full sequence

    seq = pad_or_trim(np.array(frame_buffer))
    seq_input = seq.reshape(1, SEQ_LEN, FEATURE_DIM)

    # 1-hand → predict static and dynamic
    if hand_count == 1:
        # Static: use last frame only
        pred_static_idx = one_static_model.predict(feat.reshape(1,-1), verbose=0).argmax()
        label_static = one_labels_static[pred_static_idx]

        # Dynamic: use sequence
        pred_dynamic_idx = one_dynamic_model.predict(seq_input, verbose=0).argmax()
        label_dynamic = one_labels_dynamic[pred_dynamic_idx]

        frame_buffer[:] = frame_buffer[-SEQ_LEN:]
        return label_static, label_dynamic

    # 2-hand → dynamic only
    else:
        pred_idx = two_dynamic_model.predict(seq_input, verbose=0).argmax()
        label_dynamic = two_labels_dynamic[pred_idx]
        frame_buffer[:] = frame_buffer[-SEQ_LEN:]
        return None, label_dynamic

# ================================
# VIDEO PROCESSING
# ================================
def process_video(cap):
    stframe = st.image([])
    frame_buffer = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        landmarks = []
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                coords = [[lm.x, lm.y, lm.z] for lm in handLms.landmark]
                landmarks.append(coords)
                mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

        if len(landmarks) > 0:
            landmarks_np = np.array(landmarks)
            frame_count += 1
            if frame_count % PRED_FREQ == 0:
                label_static, label_dynamic = predict_gesture(landmarks_np, frame_buffer, len(landmarks_np))

                if label_static:
                    cv2.putText(img, f"Static: {label_static}", (10,50), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
                    speak(label_static)
                if label_dynamic:
                    cv2.putText(img, f"Dynamic: {label_dynamic}", (10,100), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
                    speak(label_dynamic)

        stframe.image(img, channels="BGR")

# ================================
# STREAMLIT MODES
# ================================
if mode == "Upload video":
    uploaded_file = st.file_uploader("Upload your video", type=["mp4","avi"])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)
        process_video(cap)
        cap.release()
elif mode == "Use webcam":
    cap = cv2.VideoCapture(0)
    process_video(cap)
    cap.release()