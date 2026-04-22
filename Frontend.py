import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
import time

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Brain Tumor Detection System",
    page_icon="🧠",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------  ← EXACTLY YOUR ORIGINAL
st.markdown("""
<style>
body {
    background-color: #020617;
}
.block-container {
    padding-top: 2rem;
}
.title {
    font-size: 46px;
    font-weight: 800;
    color: #38bdf8;
}
.subtitle {
    color: #cbd5e1;
    font-size: 18px;
}
.card {
    background: linear-gradient(145deg, #020617, #020617);
    border-radius: 18px;
    padding: 24px;
    box-shadow: 0px 10px 30px rgba(56,189,248,0.15);
}
.section-title {
    color: #22d3ee;
    font-size: 26px;
    font-weight: 700;
}
.label {
    color: #94a3b8;
    font-size: 15px;
}
.result-text {
    font-size: 30px;
    font-weight: 800;
    color: #22c55e;
}
</style>
""", unsafe_allow_html=True)

# ---------------- MODEL LOADING ----------------
CLASSES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import os
import requests

MODEL_URL = "https://huggingface.co/Tarekkkkk12/brain_classifier/best_resnet50_brain.pth"
MODEL_PATH = "best_resnet50_brain.pth"
def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model from Hugging Face..."):
            r = requests.get(MODEL_URL)
            with open(MODEL_PATH, "wb") as f:
                f.write(r.content)

@st.cache_resource
def load_model():
    download_model()  # 👈 ADD THIS
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(2048, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, 4)
    )
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE)
    )
    model.to(DEVICE)
    model.eval()
    return model

def predict(image):
    model  = load_model()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).squeeze().cpu().tolist()
    pred_idx   = probs.index(max(probs))
    pred_class = CLASSES[pred_idx]
    confidence = round(probs[pred_idx] * 100, 2)
    return pred_class, confidence

# ---------------- SIDEBAR ---------------- 
st.sidebar.markdown("## 🧠 Brain Tumor Detection")
st.sidebar.markdown("---")

st.sidebar.markdown("### 📌 How to Use")
st.sidebar.markdown("""
1. Upload a **Brain MRI image**
2. Click **Analyze Scan**
3. View tumor type & confidence
""")

st.sidebar.markdown("### ⚙️ Model Information")
st.sidebar.markdown("""
- **Model Type:** CNN (ResNet-based)
- **Input Size:** 224 × 224
- **Classes:**
  - No Tumor  
  - Glioma  
  - Meningioma  
  - Pituitary  
""")

st.sidebar.markdown("### 🏥 Clinical Disclaimer")
st.sidebar.warning(
    "This system is for **educational purposes only** "
    "and must not be used for real medical diagnosis."
)

st.sidebar.markdown("### 📊 Performance (Demo)")
st.sidebar.progress(0.94)
st.sidebar.caption("Validation Accuracy: 94%")

# ---------------- HEADER ----------------  
st.markdown("<div class='title'>🧠 Brain Tumor Detection System</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI-Powered MRI Scan Analysis Dashboard</div>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ---------------- TABS ---------------- 
tab1, tab2, tab3 = st.tabs(["📤 Upload Scan", "📊 Results", "ℹ️ About System"])

# ---------------- TAB 1: UPLOAD ---------------- 
with tab1:
    st.markdown("<div class='section-title'>Upload Brain MRI Image</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    upload_col, info_col = st.columns([2, 1])

    with upload_col:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Select MRI Image",
            type=["png", "jpg", "jpeg"]
        )
        st.markdown("</div>", unsafe_allow_html=True)

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded MRI Scan", use_container_width=True)

    with info_col:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 📋 Upload Guidelines")
        st.markdown("""
- Use **clear MRI scans**
- Avoid blurred images
- Prefer **axial slices**
- JPG / PNG format only
""")
        st.markdown("</div>", unsafe_allow_html=True)

# ---------------- TAB 2: RESULTS ----------------  
with tab2:
    st.markdown("<div class='section-title'>Analysis Results</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    if uploaded_file:
        if st.button("🔍 Analyze Scan", use_container_width=True):
            with st.spinner("Running deep learning model..."):
                time.sleep(2)
                image      = Image.open(uploaded_file)
                # ── ONLY CHANGE: real model instead of random ──
                prediction, confidence = predict(image)

            col1, col2, col3 = st.columns(3)

            col1.metric("🧪 Tumor Type", prediction)
            col2.metric("📊 Confidence", f"{confidence}%")
            col3.metric("🧠 Model Accuracy", "95.94%")

            st.markdown("### 🔬 Confidence Level")
            st.progress(confidence / 100)

            st.info(
                "The prediction is generated using a **deep convolutional neural network**. "
                "Final medical decisions should be taken by professionals."
            )
    else:
        st.warning("Please upload an MRI image first.")

# ---------------- TAB 3: ABOUT ----------------  
with tab3:
    st.markdown("<div class='section-title'>About This System</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("""
### 🧠 Brain Tumor Detection Using Deep Learning

This project uses **Convolutional Neural Networks (CNNs)** 
to classify brain MRI scans into tumor categories.

#### 🔍 Key Features
- Automated MRI analysis
- Multi-class tumor classification
- High accuracy CNN model
- User-friendly medical dashboard

#### 🛠️ Technologies Used
- Python  
- Streamlit  
- TensorFlow / PyTorch  
- OpenCV  
- NumPy  

#### 🎓 Academic Use
Designed for **final year projects**, research demos, 
and AI healthcare presentations.
""")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- FOOTER ---------------- 
st.markdown("<br><br>", unsafe_allow_html=True)
st.caption("© 2026 | Brain Tumor Detection System | AI in Healthcare")
