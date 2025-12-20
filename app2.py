import streamlit as st

st.set_page_config(page_title="AI Sign Language Translator", layout="centered")

st.title("🤟 AI-Based Sign Language Translation System")

st.warning("🧪 Prototype Version – Under Development")

st.markdown("""
### 📌 Project Overview
This application aims to translate Malaysian Sign Language (BIM) into spoken language 
using Artificial Intelligence, helping bridge communication between the hearing-impaired 
community and the public.

### ⚙️ How the System Works
1. Hand gestures are captured using a camera  
2. Hand landmarks are extracted using MediaPipe  
3. Sequential gestures are analyzed using LSTM/GRU models  
4. Recognized gestures are converted into speech output  

### 🎯 Current Prototype Capabilities
- Two-hand gesture recognition  
- Supported gestures:
  - *How are you*
  - *I am fine*
- Real-time speech output (local environment)

### 🚧 Development Status
This is an early-stage prototype prepared for demonstration purposes.
More gestures and improvements will be added in future versions.
""")

st.info("Live webcam demo is available on the local system.")
