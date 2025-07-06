# 🏏 IPL Score Predictor (Streamlit App)

A machine learning-powered web app that predicts the **final score** of an IPL match based on live match conditions like overs, runs, wickets, venue, and teams.

![App Screenshot](assets/demo_screenshot.png)

## 🚀 Features

- Predicts **final score** based on current match situation.
- Real-time **Streamlit web interface**.
- Trained using **ball-by-ball IPL data up to 2025**.
- Encoded features: team, venue, runs, wickets, overs, etc.
- Deployed online via Streamlit Cloud.

## 📊 Tech Stack

- Python, Pandas, NumPy
- TensorFlow / Keras
- Scikit-learn
- Streamlit
- Git + GitHub

## 📦 Dataset

- Based on [Ritesh Ojha's IPL Dataset (2025)](https://github.com/ritesh-ojha/IPL-DATASET)
- Includes detailed ball-by-ball and match info.

## 📈 Model Info

- Neural Network (Keras Sequential)
- Optimized with Huber loss + Adam optimizer
- Trained on engineered features with MinMaxScaler

## 🧪 Live Demo

🔗 [Click here to open the app](https://ipl-score-predictor-mqvm3ei6iyeeq2tquuqxjd.streamlit.app/)

## 🛠️ How to Run Locally

```bash
git clone https://github.com/meghanadh05/ipl-score-predictor.git
cd ipl-score-predictor
pip install -r requirements.txt
streamlit run app.py
