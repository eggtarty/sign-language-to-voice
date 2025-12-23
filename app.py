import streamlit as st
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from collections import deque, Counter
from PIL import Image
from gtts import gTTS
import tempfile
import os

# ================================
# CONFIG
# ================================
MODEL_PATH = "your_twohand_model.h5"
GESTURES = ["how are you", "im fine"]
SEQUENCE_LENGTH = 30

# ================================
# INIT
# ================================
st.title("🤟 Two-Hand Gesture to Voice")
st.write("Show **BOTH hands** to the camera")

model = load_model(MODEL_PATH)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

seq = deque(maxlen=SEQUENCE_LENGTH)
predictions = deque(maxlen=5)
last_spoken = None

# ================================
# CAMERA INPUT
# ================================
image_file = st.camera_input("Capture gesture")

if image_file:
    image = Image.open(image_file)
    img = np.array(image)

    results = hands.process(img)
    keypoints = []

    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])

        seq.append(keypoints)

        if len(seq) == SEQUENCE_LENGTH:
            input_data = np.expand_dims(seq, axis=0)
            preds = model.predict(input_data, verbose=0)[0]
            gesture = GESTURES[np.argmax(preds)]
            predictions.append(gesture)

            final_gesture = Counter(predictions).most_common(1)[0][0]
            st.success(f"Detected Gesture: **{final_gesture}**")

            # 🔊 TEXT TO SPEECH (Browser-safe)
            if final_gesture != last_spoken:
                tts = gTTS(final_gesture)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                    tts.save(fp.name)
                    st.audio(fp.name)
                last_spoken = final_gesture

    elif results.multi_hand_landmarks:
        st.warning("⚠ Please show **BOTH hands**")

    else:
        st.warning("❌ No hands detected")
