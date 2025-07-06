🏏 IPL Score Predictor (Streamlit App)

A machine learning-powered web app that predicts the final score of an IPL innings based on match conditions.

<div align="center">
  <img src="https://img.shields.io/badge/Machine%20Learning-Regression-blue">
  <img src="https://img.shields.io/badge/Framework-TensorFlow-orange">
  <img src="https://img.shields.io/badge/Deployment-Streamlit-green">
  <img src="https://img.shields.io/badge/Data-IPL%202008--2025-blueviolet">
</div>



⸻

🚀 Project Overview

This project predicts the final score of a team in an IPL match in real time using historical ball-by-ball data. It includes a clean Streamlit UI, trained deep learning model (.keras), and uses team and match context like:
	•	Batting team
	•	Bowling team
	•	Venue
	•	Overs completed
	•	Runs scored
	•	Wickets fallen

⸻

🧠 Machine Learning Pipeline

📊 Dataset
	•	Source: ritesh-ojha/IPL-DATASET
	•	Files used:
	•	Ball_By_Ball_Match_Data.csv
	•	Match_Info.csv
	•	Covers matches up to IPL 2025

🔍 Feature Engineering
	•	over_number: Over + Ball/10 for precision
	•	runs_till_now: Cumulative runs till current ball
	•	wickets_till_now: Cumulative wickets till current ball
	•	Label encoding for categorical features: teams, venue

🎯 Model
	•	Framework: TensorFlow (Keras)
	•	Architecture:

Input → Dense(128, ReLU) → Dense(64, ReLU) → Dense(1, Linear)


	•	Loss: Huber (handles outliers better than MSE)
	•	Metrics: MAE, RMSE
	•	Scaler: MinMaxScaler for numeric feature normalization

🏁 Performance
	•	📉 MAE: ~10–12 runs
	•	📉 RMSE: ~15–20 runs
	•	✅ Realistic predictions in second innings scenarios

⸻

📲 Streamlit Web App

Features:
	•	Dropdowns to select teams and venues
	•	Inputs for overs, runs, and wickets
	•	Displays predicted final score instantly


🌐 aclike here To : [streamlit.app link here]

⸻

🧰 Tech Stack

Layer	Technology
ML Model	TensorFlow + Keras
Preprocessing	Scikit-learn, Pandas
Deployment	Streamlit
Versioning	Git, GitHub


⸻

📁 Folder Structure

ipl-score-predictor/
├── app.py                  # Streamlit app
├── ipl_score_predictor.keras   # Trained model
├── scaler.pkl             # Scaler used for features
├── requirements.txt
├── README.md


⸻

📌 How to Run Locally

# Clone repo
git clone https://github.com/yourusername/ipl-score-predictor
cd ipl-score-predictor

# (Optional) Create virtual env
conda create -n ipl-env python=3.10
conda activate ipl-env

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py


⸻

📜 Future Improvements
	•	Include ball-by-ball prediction
	•	Add player-level context (strike rate, economy)
	•	Expand venue and pitch condition features
	•	Improve deployment scalability

⸻

📧 Contact

Meghanadh Borra
📫 GitHub | LinkedIn

⸻

Let me know if you want this version pushed to your GitHub repo, or want help adding badges/screenshots.
