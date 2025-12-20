import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque, Counter
import mediapipe as mp
import pyttsx3
from PIL import Image

# ================================
# CONFIG
# ================================
MODEL_PATH = "your_twohand_model.h5"  # <- replace with your trained model path
GESTURES = ['how are you', 'im fine']  # <- replace with your gestures
SEQUENCE_LENGTH = 30  # sequence length for prediction

# ================================
# INIT
# ================================
st.title("Two-Hand Gesture to Voice (Browser-Friendly)")
st.write("Use your webcam to perform two-hand gestures.")

# Load model
model = load_model(MODEL_PATH)

# MediaPipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Smoothing buffers
seq = deque(maxlen=SEQUENCE_LENGTH)
predictions = deque(maxlen=5)

# ================================
# STREAMLIT CAMERA INPUT
# ================================
camera_input = st.camera_input("Show your hands to the camera")

if camera_input is not None:
    # Convert uploaded image to OpenCV format
    img = np.array(Image.open(camera_input))
    frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Flip for selfie view
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # MediaPipe processing
    result = hands.process(rgb_frame)
    keypoints = []
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            for lm in hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
        
        # Only predict if two hands detected
        if len(result.multi_hand_landmarks) == 2:
            seq.append(keypoints)
            
            if len(seq) == SEQUENCE_LENGTH:
                input_data = np.expand_dims(seq, axis=0)
                res = model.predict(input_data, verbose=0)[0]
                gesture_id = np.argmax(res)
                gesture = GESTURES[gesture_id]
                predictions.append(gesture)
                
                # Majority voting for stability
                most_common = Counter(predictions).most_common(1)[0][0]
                
                # Display and speak
                st.write(f"**Detected Gesture:** {most_common}")
                if not hasattr(engine, 'last_spoken') or engine.last_spoken != most_common:
                    engine.say(most_common)
                    engine.runAndWait()
                    engine.last_spoken = most_common
        else:
            st.warning("Please show **BOTH hands**.")
    else:
        st.warning("No hands detected.")
    
    # Show annotated frame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    st.image(frame_rgb)