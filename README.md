# 🛡️ AthleteGuard AI — Biomechanical Movement Screening & Injury Risk Engine

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Pose%20Landmarker-orange.svg)](https://developers.google.com/mediapipe)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> **Clinical-grade video analytics for athletic jump landing and takeoff kinematics, powered by Google MediaPipe pose tracking, the Landing Error Scoring System (LESS) protocol (Padua et al., 2009), and AI-driven coaching intelligence.**

---

## 🌟 Key Capabilities

* 🔬 **Phase-Aware Movement Parsing**: Automatically identifies and tracks **Approach**, **Takeoff**, **Flight**, and **Landing** phases via vertical ankle kinematics and velocity profiling.
* 🦵 **Frontal Plane Knee Valgus (FPPA)**: Real-time quantification of dynamic knee collapse—the primary non-contact mechanism for ACL ruptures.
* 🧠 **Plain-Language Explainability Layer**: Translates raw angular measurements into structured biomechanical risk drivers grounded in LESS literature.
* 🎯 **Coach Comparison Overlay**: Ghost-overlaid or side-by-side synchronized view comparing the athlete's attempt against an elite benchmark or their personal best, featuring frame-by-frame **Takeoff Joint Deltas**.
* 📊 **Multi-Rep Consistency & Fatigue Degradation**: Computes motor variance across repeated attempts and tracks neuromuscular fatigue drift via linear regression slopes.
* 📄 **Clinical PDF Report Generation**: One-click export of an executive 1-page PDF screening summary suitable for physicians, coaches, and physiotherapists.
* 🤖 **Clinical Physio AI Advisor**: Generates periodized 4-week corrective training regimens targeting specific joint deficit vectors.

---

## 🛠️ Architecture & Tech Stack

* **Frontend**: [Streamlit](https://streamlit.io/)
* **Pose Estimation**: [Google MediaPipe](https://developers.google.com/mediapipe)
* **Computer Vision**: OpenCV (`opencv-python-headless`)
* **Kinematic Analytics**: Pandas, NumPy
* **PDF Engine**: ReportLab
* **Local Persistence**: SQLite3

---

## 🚀 Quick Start (Local Setup)

### 1. Clone the repository
```bash
git clone https://github.com/Abinav2107/Athleteguard-ai.git
cd Athleteguard-ai
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the dashboard
```bash
streamlit run dashboard.py
```
Open `http://localhost:8501` in your browser.

---

## ☁️ Deploying to Streamlit Community Cloud

This repository is pre-configured with `requirements.txt`, `packages.txt`, and `.streamlit/config.toml` for 1-click cloud deployment:

1. Sign in to **[share.streamlit.io](https://share.streamlit.io/)** with GitHub.
2. Click **"New app"**.
3. Select repository: `Abinav2107/Athleteguard-ai`.
4. Set Main file path: `dashboard.py`.
5. Click **Deploy!**

---

## 🏥 Clinical & Ethical Disclaimer
*AthleteGuard AI is a biomechanical movement screening and technique analysis tool intended for educational, athletic training, and injury prevention purposes. It does not provide medical diagnoses or replace clinical evaluations by qualified healthcare professionals.*
