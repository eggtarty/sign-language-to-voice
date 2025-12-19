import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from collections import deque, Counter
import time
import pyttsx3

# ================================
# CONFIG
# ================================
MODEL_PATH = "two_hand_gesture_model.h5"
MAX_LEN = 78
CONF_THRESHOLD = 0.6
SMOOTHING_WINDOW = 5
SPEAK_COOLDOWN = 2.0

inv_label_map = {0: "how_are_you", 1: "im_fine"}

gesture_text = {
    "how_are_you": "How are you?",
    "im_fine": "I am fine."
}

# ================================
# INIT TTS (LOCAL ONLY)
# ================================
@st.cache_resource
def init_tts():
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    return engine

tts_engine = init_tts()

# ================================
# UI
# ================================
st.set_page_config(page_title="Sign Language to Voice", layout="centered")
st.title("🤟 Two-Hand Sign Language to Voice")
st.write("Real-time two-hand gesture recognition with speech output")

start = st.button("▶ Start Camera")
stop = st.button("⏹ Stop Camera")

frame_window = st.image([])
status = st.empty()

# ================================
# LOAD MODEL
# ================================
@st.cache_resource
def load_gesture_model():
    return load_model(MODEL_PATH)

model = load_gesture_model()

# ================================
# MEDIAPIPE
# ================================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# ================================
# STATE
# ================================
sequence = []
pred_queue = deque(maxlen=SMOOTHING_WINDOW)
last_spoken_gesture = None
last_spoken_time = 0

# ================================
# CAMERA LOOP
# ================================
if start:
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if not results.multi_hand_landmarks or len(results.multi_hand_landmarks) != 2:
            sequence.clear()
            pred_queue.clear()
            last_spoken_gesture = None
            status.warning("⚠ Two hands required")
        else:
            landmarks = []
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(rgb, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks.extend([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])

            landmarks = np.array(landmarks).flatten()
            landmarks = np.pad(landmarks, (0, max(0, 126 - len(landmarks))))[:126]

            sequence.append(landmarks)
            if len(sequence) > MAX_LEN:
                sequence.pop(0)

            if len(sequence) >= 10:
                seq_input = np.expand_dims(np.array(sequence, dtype=np.float32), axis=0)
                pred = model.predict(seq_input, verbose=0)[0]
                idx = np.argmax(pred)
                conf = np.max(pred)

                if conf >= CONF_THRESHOLD:
                    pred_queue.append(idx)
                    gesture = inv_label_map[Counter(pred_queue).most_common(1)[0][0]]

                    status.success(f"{gesture_text[gesture]} ({conf*100:.1f}%)")

                    now = time.time()
                    if gesture != last_spoken_gesture or now - last_spoken_time > SPEAK_COOLDOWN:
                        tts_engine.say(gesture_text[gesture])
                        tts_engine.runAndWait()
                        last_spoken_gesture = gesture
                        last_spoken_time = now

        frame_window.image(rgb)

        if stop:
            break

    cap.release()
