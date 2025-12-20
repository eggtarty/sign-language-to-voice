import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque, Counter
import mediapipe as mp
from PIL import Image

# ================================
# OPTIONAL AUDIO (SAFE IMPORT)
# ================================
AUDIO_ENABLED = True
try:
    import pyttsx3
except Exception:
    AUDIO_ENABLED = False

# ================================
# CONFIG
# ================================
MODEL_PATH = "your_twohand_model.h5"   # <-- change to your model filename
GESTURES = ["how are you", "im fine"]  # must match training order
SEQUENCE_LENGTH = 30

# ================================
# STREAMLIT UI
# ================================
st.set_page_config(page_title="Two-Hand Gesture to Voice", layout="centered")
st.title("🤟 Two-Hand Gesture to Voice")
st.write("Use your webcam to perform **two-hand gestures**.")

if not AUDIO_ENABLED:
    st.info("🔇 Audio is disabled on web deployment. Text output only.")

# ================================
# LOAD MODEL (cached)
# ================================
@st.cache_resource
def load_gesture_model():
    return load_model(MODEL_PATH)

model = load_gesture_model()

# ================================
# INIT MEDIAPIPE
# ================================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# ================================
# INIT TTS (LOCAL ONLY)
# ================================
if AUDIO_ENABLED:
    @st.cache_resource
    def init_tts():
        engine = pyttsx3.init()
        engine.setProperty("rate", 150)
        return engine

    tts_engine = init_tts()
    last_spoken = None

# ================================
# BUFFERS
# ================================
seq = deque(maxlen=SEQUENCE_LENGTH)
predictions = deque(maxlen=5)

# ================================
# CAMERA INPUT (BROWSER FRIENDLY)
# ================================
camera_input = st.camera_input("📷 Show both hands clearly")

if camera_input is not None:
    # Convert image to OpenCV format
    img = np.array(Image.open(camera_input))
    frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    keypoints = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )
            for lm in hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])

        # ================================
        # REQUIRE EXACTLY TWO HANDS
        # ================================
        if len(results.multi_hand_landmarks) == 2:
            seq.append(keypoints)

            if len(seq) == SEQUENCE_LENGTH:
                input_data = np.expand_dims(seq, axis=0)
                prediction = model.predict(input_data, verbose=0)[0]
                gesture_id = np.argmax(prediction)
                confidence = np.max(prediction)

                predictions.append(gesture_id)
                stable_id = Counter(predictions).most_common(1)[0][0]
                gesture_name = GESTURES[stable_id]

                st.success(f"✅ Detected: **{gesture_name}** ({confidence*100:.1f}%)")

                # ================================
                # AUDIO OUTPUT (LOCAL ONLY)
                # ================================
                if AUDIO_ENABLED:
                    if last_spoken != gesture_name:
                        tts_engine.say(gesture_name)
                        tts_engine.runAndWait()
                        last_spoken = gesture_name
        else:
            st.warning("⚠ Please show **BOTH hands**.")
            seq.clear()
            predictions.clear()
    else:
        st.warning("⚠ No hands detected.")

    # Display frame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    st.image(frame_rgb, caption="Live Camera Feed")
