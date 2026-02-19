🖐 Real-Time Hand Gesture Recognition
Graduation Project – AMIT AI Engineering Track
📌 Project Overview

This project implements a real-time hand gesture recognition system using two different AI approaches:

1️⃣ Option A – Deep Learning (YOLO Classifier)

2️⃣ Option B – Landmark-Based ML (MediaPipe + Scikit-Learn)

The system performs:

Dataset creation

Preprocessing (hand cropping)

Model training (GPU supported)

Evaluation & comparison

Real-time webcam inference

The goal is to compare deep learning vs feature-based ML in terms of:

Accuracy

Stability

Real-world performance

Inference speed

🧠 System Architecture
Webcam → Hand Detection → Feature Extraction → Model Prediction → Display Result
Option A – YOLO26n Classification

Model: YOLO (Ultralytics)

Input: Cropped hand images (224x224)

Training: GPU (RTX 3050 Ti)

Output: Gesture class (1–5)

Option B – Landmark-Based Classifier

Hand detection: MediaPipe

Features: 21 landmarks × (x,y,z) = 63 features

Model: Scikit-learn Pipeline (StandardScaler + Classifier)

Lightweight & CPU-friendly

📂 Project Structure

Hand-Gesture-AMIT
│
├── notebooks/
│   └── Real_Time_Hand_Gesture_Recognition_AMIT_FIXED.ipynb
│
├── src/
│   ├── webcam_compare.py
│   └── crop_hands.py
│
├── artifacts/
│   └── optionB_best_model.joblib
│
├── .vscode/
│   ├── launch.json
│   └── settings.json
│
└── README.md

⚙️ Installation

1️⃣ Create Conda Environment
conda create -n handai python=3.10 -y
conda activate handai
2️⃣ Install Dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics mediapipe opencv-python scikit-learn numpy matplotlib joblib
🚀 Training
YOLO Training
model = YOLO("yolo26n-cls.pt")
model.train(
    data="path/to/dataset",
    epochs=50,
    imgsz=224,
    device=0
)
🎥 Run Real-Time Demo
From VS Code

Press F5

OR

python src/webcam_compare.py

Press Q to exit.

📊 Results
Model	Validation Accuracy	Real-Time Stability
YOLO26n	~90%	High (after cropping)
Landmark ML	~88–95%	Very stable

Observation:

YOLO performs better numerically

Landmark model sometimes feels more stable in uncontrolled lighting

Combining both gives robust performance

🔥 GPU Info

Device: NVIDIA RTX 3050 Ti Laptop GPU

CUDA: Enabled

Torch Version: CUDA 12.1

🧪 Key Engineering Lessons

Dataset consistency is critical (train distribution = inference distribution)

Feature mismatch (42 vs 63) causes scaler failure

GPU acceleration significantly reduces training time

Real-time preprocessing matters more than validation metrics

👨‍💻 Author

Mohamed Walid
AI Engineering – AMIT
Graduation Project 2026

📜 License


Academic use only – Graduation Project

