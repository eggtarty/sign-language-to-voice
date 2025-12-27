import cv2
import streamlit as st
import mediapipe as mp
import numpy as np
import pyttsx3
import threading
import time
import queue
from collections import deque
from tensorflow.keras.models import load_model

# ==================
# CONFIG & PATHS 
# ==================
import os

# Get the directory where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SEQ_LEN = 30
MIN_DYNAMIC_FRAMES = 20
STATIC_MODEL_PATH = os.path.join(BASE_DIR, "models", "static_model.keras")
DYNAMIC_MODEL_PATH = os.path.join(BASE_DIR, "models", "dynamic_model.keras")
STATIC_LABELS_PATH = os.path.join(BASE_DIR, "models", "labels_static.npy")
DYNAMIC_LABELS_PATH = os.path.join(BASE_DIR, "models", "labels_dynamic.npy")

st.set_page_config(page_title="SignAL Academic Pro", layout="wide")

# ==========================================
# UI THEME: IMPROVED COLOR SCHEME
# ==========================================
st.markdown("""
<style>
    .stApp { 
        background-color: #D6EAF8;
        color: #1B2631;
    }
    
    /* Light theme for dark cards */
    .prediction-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        color: #f8f9fa;
        padding: 40px;
        border-radius: 20px;
        border: 2px solid #3498DB;
        text-align: center;
        box-shadow: 0 12px 24px rgba(0, 0, 0, 0.15);
        margin: 20px 0;
    }
    
    /* Dark theme text elements for light background */
    .info-text {
        color: #2c3e50;
        font-weight: 500;
    }
    
    .title-text {
        color: #1a237e;
        font-weight: 700;
    }
    
    /* Light theme text for dark cards */
    .mode-tag { 
        color: #4fc3f7;
        font-size: 1.3rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        background: rgba(79, 195, 247, 0.1);
        padding: 8px 16px;
        border-radius: 10px;
        display: inline-block;
        margin-bottom: 20px;
    }
    
    .gesture-output { 
        font-size: 4.5rem;
        color: #00e676;
        font-weight: 850;
        margin: 25px 0;
        text-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
        background: linear-gradient(45deg, #00e676, #00c853);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .timer-bar { 
        color: #bbdefb;
        font-size: 1.1rem;
        font-family: 'Courier New', monospace;
        background: rgba(30, 30, 46, 0.7);
        padding: 10px 15px;
        border-radius: 8px;
        display: inline-block;
    }
    
    /* Sidebar styling */
    .css-1d391kg, .css-12oz5g7 {
        color: #d0eff7;
    }
    
    /* Button styling */
    .stButton button {
        background: linear-gradient(45deg, #2196F3, #21CBF3);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 8px;
        font-weight: 600;
    }
    
    .stButton button:hover {
        background: linear-gradient(45deg, #1976D2, #0D47A1);
        color: white;
    }
    
    /* Status indicators */
    .static-mode { 
        border-color: #FFC107 !important;
        background: linear-gradient(135deg, #1a1a2e 0%, #1a120e 100%);
    }
    .dynamic-mode { 
        border-color: #2196F3 !important;
        background: linear-gradient(135deg, #1a1a2e 0%, #0d1a2e 100%);
    }
    
    .buffer-status {
        color: #ff9800;
        font-size: 1.2rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# CORE ASSETS & UTILS
# ==========================================
@st.cache_resource
def load_assets():
    try:
        m_s = load_model(STATIC_MODEL_PATH)
        m_d = load_model(DYNAMIC_MODEL_PATH)
        l_s = np.load(STATIC_LABELS_PATH, allow_pickle=True)
        l_d = np.load(DYNAMIC_LABELS_PATH, allow_pickle=True)
        return m_s, m_d, l_s, l_d
    except Exception as e:
        st.error(f"Error loading models/labels: {e}")
        return None, None, None, None

def speak_text(text):
    """Simple thread-safe speech function"""
    def shell():
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 140)
            engine.say(text)
            engine.runAndWait()
        except:
            pass
    threading.Thread(target=shell).start()

def get_120_features(coords_seq):
    """Extract 120 features from coordinate sequence"""
    all_frame_feats = []
    for frame in coords_seq:
        # Bone normalization
        wrist = frame[0]
        norm_frame = frame - wrist
        scale = np.max(np.linalg.norm(norm_frame, axis=1))
        if scale > 0: 
            norm_frame /= scale
        
        # 60 bones
        bones = np.array([norm_frame[i] - norm_frame[0] for i in range(1, 21)]).flatten()
        all_frame_feats.append(bones)
    
    all_frame_feats = np.array(all_frame_feats)
    
    # Velocity (60 dims)
    if all_frame_feats.shape[0] < 2:
        velocity = np.zeros_like(all_frame_feats)
    else:
        velocity = np.diff(all_frame_feats, axis=0)
        velocity = np.vstack([velocity, np.zeros((1, 60))])
    
    return np.concatenate([all_frame_feats, velocity], axis=1) # (T, 120)

def calculate_motion_intensity(features_120, window_size=5):
    """Calculate motion intensity over a window of frames"""
    if len(features_120) < window_size:
        return 0
    
    # Get velocity features (last 60 dimensions)
    velocities = features_120[-window_size:, 60:]
    
    # Calculate magnitude of velocity for each frame
    motion_magnitudes = np.linalg.norm(velocities, axis=1)
    
    # Return average motion intensity
    return np.mean(motion_magnitudes)

def pad_sequence_to_length(sequence, target_length):
    """Pad sequence to required length for dynamic model"""
    if len(sequence) >= target_length:
        return sequence[-target_length:]
    else:
        # Repeat the last frame to pad the sequence
        padding_needed = target_length - len(sequence)
        padding = [sequence[-1]] * padding_needed
        return sequence + padding

# ==========================================
# GESTURE DETECTION STATE MACHINE
# ==========================================
class GestureDetector:
    def __init__(self, seq_len=30, min_dynamic_frames=20):
        self.seq_len = seq_len
        self.min_dynamic_frames = min_dynamic_frames
        self.coord_buffer = deque(maxlen=seq_len)
        self.last_move_time = time.time()
        self.last_prediction_time = 0
        self.current_mode = "Initializing"
        self.prediction_cooldown = 1.0  # seconds between predictions
        self.last_spoken = ""
        self.motion_history = deque(maxlen=10)
        self.is_dynamic_active = False
        self.dynamic_start_time = 0
        
    def update(self, coords, motion_intensity, motion_threshold=0.15):
        """Update detector state and return current mode"""
        current_time = time.time()
        
        # Update buffer
        if coords is not None:
            self.coord_buffer.append(coords)
        
        # Motion detection
        if motion_intensity > motion_threshold:
            self.last_move_time = current_time
            self.motion_history.append(True)
            
            # Start dynamic mode if not already active
            if not self.is_dynamic_active and len(self.coord_buffer) >= self.min_dynamic_frames:
                self.is_dynamic_active = True
                self.dynamic_start_time = current_time
        else:
            self.motion_history.append(False)
        
        # Still duration
        still_duration = current_time - self.last_move_time
        
        # Mode decision
        if self.is_dynamic_active:
            # Check if dynamic gesture is complete (stopped moving for a bit)
            if still_duration > 0.5:  # 0.5 seconds of stillness ends dynamic
                self.is_dynamic_active = False
                self.current_mode = "Static"
            else:
                self.current_mode = "Dynamic"
        elif still_duration > 2.0:  # 2 seconds of stillness for static
            self.current_mode = "Static"
        else:
            self.current_mode = "Transition"
        
        # Reset buffer if mode changes from dynamic to static
        if self.current_mode == "Static" and len(self.motion_history) > 0 and any(self.motion_history):
            self.coord_buffer.clear()
        
        return self.current_mode, still_duration
    
    def can_predict_dynamic(self):
        """Check if we can make a dynamic prediction"""
        return (self.is_dynamic_active and 
                len(self.coord_buffer) >= self.min_dynamic_frames and
                (time.time() - self.last_prediction_time) > self.prediction_cooldown)
    
    def can_predict_static(self):
        """Check if we can make a static prediction"""
        still_duration = time.time() - self.last_move_time
        return (self.current_mode == "Static" and 
                still_duration > 2.0 and
                len(self.coord_buffer) > 0 and
                (time.time() - self.last_prediction_time) > self.prediction_cooldown)
    
    def record_prediction(self):
        """Record prediction time to prevent spamming"""
        self.last_prediction_time = time.time()

# ==========================================
# MAIN APP LOOP
# ==========================================
m_static, m_dynamic, le_static, le_dynamic = load_assets()

st.markdown('<h1 class="title-text">🎓 Sign Language AI - Academic System</h1>', unsafe_allow_html=True)

# Add mode explanation
with st.sidebar:
    st.markdown("### 📋 Mode Detection")
    st.markdown("""
    **Static Mode**: Hand remains still for 2+ seconds  
    **Dynamic Mode**: Continuous hand movement detected
    
    *Tip: Complete full gesture movements*
    """)
    
    # Add confidence threshold slider
    motion_threshold = st.slider(
        "Motion Detection Sensitivity",
        min_value=0.05,
        max_value=0.30,
        value=0.15,
        step=0.01,
        help="Adjust how sensitive the system is to hand movement"
    )
    
    static_confidence = st.slider(
        "Static Gesture Confidence",
        min_value=0.5,
        max_value=0.95,
        value=0.7,
        step=0.05,
        help="Minimum confidence for static gesture prediction"
    )
    
    dynamic_confidence = st.slider(
        "Dynamic Gesture Confidence",
        min_value=0.4,
        max_value=0.9,
        value=0.6,
        step=0.05,
        help="Minimum confidence for dynamic gesture prediction"
    )
    
    # Show current thresholds
    st.info(f"""
    **Current Settings:**
    - Motion Threshold: `{motion_threshold:.3f}`
    - Static Confidence: `{static_confidence:.2f}`
    - Dynamic Confidence: `{dynamic_confidence:.2f}`
    - Min Dynamic Frames: `{MIN_DYNAMIC_FRAMES}`
    """)

run_cam = st.sidebar.toggle("Switch Camera ON", value=False, 
                          help="Enable/Disable webcam feed")
col_cam, col_res = st.columns([2, 1])

if run_cam and m_static is not None:
    cap = cv2.VideoCapture(0)
    # MediaPipe setup
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=1, 
        min_detection_confidence=0.7, 
        min_tracking_confidence=0.7
    )
    
    # Initialize gesture detector
    detector = GestureDetector(seq_len=SEQ_LEN, min_dynamic_frames=MIN_DYNAMIC_FRAMES)
    
    cam_ui = col_cam.empty()
    res_ui = col_res.empty()
    
    # Add status placeholder
    status_placeholder = st.sidebar.empty()
    prediction_placeholder = st.sidebar.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: 
            break
        
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            h_lms = results.multi_hand_landmarks[0]
            curr_coords = np.array([[lm.x, lm.y, lm.z] for lm in h_lms.landmark])
            mp.solutions.drawing_utils.draw_landmarks(frame, h_lms, mp_hands.HAND_CONNECTIONS)
            
            # Calculate features for motion detection
            feats_120 = None
            if len(detector.coord_buffer) > 0:
                buffer_list = list(detector.coord_buffer)
                buffer_list.append(curr_coords)
                feats_120 = get_120_features(buffer_list)
                motion_intensity = calculate_motion_intensity(feats_120, window_size=3)
            else:
                motion_intensity = 0
            
            # Update detector state
            current_mode, still_duration = detector.update(
                curr_coords, 
                motion_intensity, 
                motion_threshold
            )
            
            # Default label
            label = "Ready..."
            mode_str = current_mode
            prediction_made = False
            
            # Make predictions based on mode
            if detector.can_predict_dynamic():
                # DYNAMIC PREDICTION
                mode_str = "Dynamic Gesture"
                if len(detector.coord_buffer) >= MIN_DYNAMIC_FRAMES:
                    # Prepare sequence for prediction
                    buffer_list = list(detector.coord_buffer)
                    
                    # Pad if needed (but we already have minimum frames)
                    if len(buffer_list) < SEQ_LEN:
                        buffer_list = pad_sequence_to_length(buffer_list, SEQ_LEN)
                    
                    # Get features and predict
                    feats_120_dynamic = get_120_features(buffer_list)
                    input_data = feats_120_dynamic.reshape(1, SEQ_LEN, 120)
                    
                    try:
                        pred = m_dynamic.predict(input_data, verbose=0)
                        label_idx = np.argmax(pred)
                        confidence = np.max(pred)
                        
                        if confidence > dynamic_confidence:
                            label = le_dynamic[label_idx]
                            prediction_made = True
                            detector.record_prediction()
                            
                            # Show confidence in status
                            prediction_placeholder.success(f"Dynamic: {label} ({confidence:.2%})")
                        else:
                            label = f"Low Confidence ({confidence:.2%})"
                            prediction_placeholder.warning(f"Dynamic confidence low: {confidence:.2%}")
                    except Exception as e:
                        label = "Prediction Error"
                        st.error(f"Dynamic prediction error: {e}")
                else:
                    label = f"Collecting: {len(detector.coord_buffer)}/{MIN_DYNAMIC_FRAMES}"
            
            elif detector.can_predict_static():
                # STATIC PREDICTION
                mode_str = "Static Gesture"
                if feats_120 is not None and len(feats_120) > 0:
                    input_data = feats_120[-1].reshape(1, 120)
                    pred = m_static.predict(input_data, verbose=0)
                    label_idx = np.argmax(pred)
                    confidence = np.max(pred)
                    
                    if confidence > static_confidence:
                        label = le_static[label_idx]
                        prediction_made = True
                        detector.record_prediction()
                        
                        # Show confidence in status
                        prediction_placeholder.success(f"Static: {label} ({confidence:.2%})")
                    else:
                        label = f"Uncertain ({confidence:.2%})"
                        prediction_placeholder.warning(f"Static confidence low: {confidence:.2%}")
                else:
                    label = "Analyzing..."
            
            else:
                # Transition or waiting state
                if current_mode == "Transition":
                    if len(detector.coord_buffer) < MIN_DYNAMIC_FRAMES:
                        label = f"Start Moving ({len(detector.coord_buffer)}/{MIN_DYNAMIC_FRAMES})"
                    else:
                        label = "Perform Gesture"
                else:
                    label = "Ready..."
            
            # Update status display
            buffer_size = len(detector.coord_buffer)
            status_text = f"""
            **Status Panel**
            - Mode: `{current_mode}`
            - Motion: `{motion_intensity:.3f}`
            - Still: `{still_duration:.1f}s`
            - Buffer: `{buffer_size}` frames
            - Last Prediction: `{time.time() - detector.last_prediction_time:.1f}s ago`
            """
            status_placeholder.markdown(status_text)
            
            # UI Update with conditional styling
            mode_class = "static-mode" if "Static" in mode_str else "dynamic-mode"
            
            # Special styling for buffering/collecting
            if "Collecting" in label or "Start Moving" in label:
                buffer_percent = min(100, (buffer_size / MIN_DYNAMIC_FRAMES) * 100)
                res_ui.markdown(f"""
                    <div class="prediction-card dynamic-mode">
                        <p class="mode-tag">Collecting Frames</p>
                        <p class="buffer-status">{label}</p>
                        <div style="background: #333; height: 20px; border-radius: 10px; margin: 20px 0;">
                            <div style="background: linear-gradient(90deg, #2196F3, #21CBF3); 
                                     height: 100%; width: {buffer_percent}%; 
                                     border-radius: 10px;"></div>
                        </div>
                        <p class="timer-bar">
                            Need {MIN_DYNAMIC_FRAMES} frames for dynamic gesture
                        </p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                res_ui.markdown(f"""
                    <div class="prediction-card {mode_class}">
                        <p class="mode-tag">{mode_str}</p>
                        <p class="gesture-output">{label}</p>
                        <p class="timer-bar">
                            Motion: {motion_intensity:.3f} | 
                            Buffer: {buffer_size} frames
                        </p>
                    </div>
                """, unsafe_allow_html=True)

            # Trigger Speech (only for valid predictions)
            if (prediction_made and label != detector.last_spoken and 
                label not in ["Ready...", "Analyzing...", "Uncertain", "Low Confidence", 
                             "Collecting", "Start Moving", "Perform Gesture"]):
                speak_text(label)
                detector.last_spoken = label
                
                # Add visual feedback for speech
                cv2.putText(frame, "✓ SPEAKING", (50, 50), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        else:
            # No hand detected
            detector.coord_buffer.clear()
            detector.is_dynamic_active = False
            current_mode = "No Hand"
            
            res_ui.markdown(f"""
                <div class="prediction-card">
                    <p class="mode-tag" style="color:#ff9800;">Waiting</p>
                    <p class="gesture-output" style="color:#ff9800;">Show Hand</p>
                    <p class="timer-bar">Position hand in frame</p>
                </div>
            """, unsafe_allow_html=True)
            
            status_placeholder.warning("⚠️ No hand detected")
            prediction_placeholder.empty()
        
        # Add motion visualization to camera feed
        if len(detector.coord_buffer) > 0:
            motion_text = f"Motion: {motion_intensity:.3f}"
            cv2.putText(frame, motion_text, (50, frame.shape[0] - 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            mode_text = f"Mode: {current_mode}"
            cv2.putText(frame, mode_text, (50, frame.shape[0] - 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            buffer_text = f"Buffer: {len(detector.coord_buffer)}/{SEQ_LEN}"
            cv2.putText(frame, buffer_text, (50, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)

        cam_ui.image(frame, channels="BGR", caption="Live Feed")
    
    cap.release()
    hands.close()
    st.sidebar.success("Camera released successfully")
else:
    if not run_cam:
        st.sidebar.info("Toggle the switch to start camera")
    else:
        st.error("Models failed to load. Please check the model paths.")