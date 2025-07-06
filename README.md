# 🏏 IPL Score Predictor

**A machine learning-powered web app that predicts the final score of an IPL innings based on real-time match conditions, trained on detailed historical IPL data (2008–2025).**

<p align="center">
  <img src="https://img.shields.io/badge/Machine%20Learning-Regression-blue" />
  <img src="https://img.shields.io/badge/Framework-TensorFlow-orange" />
  <img src="https://img.shields.io/badge/Deployment-Streamlit-green" />
  <img src="https://img.shields.io/badge/Data-IPL%202008--2025-blueviolet" />
</p>

[![Streamlit App](https://img.shields.io/badge/Launch%20App-Try%20Now-1f6f9d?style=for-the-badge&logo=streamlit)](https://ipl-score-predictor-mqvm3ei6iyeeq2tquuqxjd.streamlit.app/)

---

## 🚀 Project Overview

This project predicts the **final score** of a team during an IPL match using historical ball-by-ball data. It includes a clean **Streamlit UI**, trained deep learning model (`.keras`), and uses match context like:

- Batting team
- Bowling team
- Venue
- Overs completed
- Runs scored
- Wickets fallen

---

## 🧠 Machine Learning Pipeline

### 📊 Dataset
- Source: [`ritesh-ojha/IPL-DATASET`](https://github.com/ritesh-ojha/IPL-DATASET)
- Files used:
  - `Ball_By_Ball_Match_Data.csv`
  - `Match_Info.csv`
- Covers matches up to **IPL 2025**

## 🎯 Model Details

- **Framework**: TensorFlow/Keras
- **Architecture**: `Dense(128) → Dense(64) → Dense(1)`
- **Loss**: Huber Loss for robustness against outliers
- **Scaler**: `MinMaxScaler` (stored as `scaler.pkl`)
- **Dataset**: Ritesh-Ojha IPL Dataset ([GitHub Link](https://github.com/ritesh-ojha/IPL-DATASET))

---

## 🖼️ Screenshot

![App Screenshot](assets/screenshot.png)

---

## 🎥 Demo

https://github.com/meghanadh05/ipl-score-predictor/assets/demo-video.mp4

---
## 🛠️ Run Locally (ML Inference)

```bash
# 1. Clone the repository
git clone https://github.com/meghanadh05/ipl-score-predictor.git
cd ipl-score-predictor

# 2. (Optional) Create and activate a virtual environment
python -m venv venv

# macOS/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Streamlit app
streamlit run app.py
