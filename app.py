import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# 🧠 Load Keras model
model = load_model("ipl_score_predictor.keras")

# 🧪 Load scaler
scaler = joblib.load("scaler.pkl")

# ✅ Label encoders (same as training)
team_mapping = {
    0: "CSK", 1: "MI", 2: "RCB", 3: "GT", 4: "DC",
    5: "RR", 6: "SRH", 7: "KKR", 8: "LSG", 9: "PBKS"
}

venue_mapping = {
    0: "Wankhede Stadium",
    1: "Eden Gardens",
    2: "Narendra Modi Stadium",
    3: "M. Chinnaswamy Stadium",
    4: "Arun Jaitley Stadium",
    5: "Sawai Mansingh Stadium",
    6: "Rajiv Gandhi International Stadium",
    7: "MA Chidambaram Stadium",
    8: "Himachal Pradesh Cricket Association Stadium",
    9: "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium",
    10: "Punjab Cricket Association IS Bindra Stadium",
    11: "Dr DY Patil Sports Academy",
    12: "Brabourne Stadium",
    13: "Maharashtra Cricket Association Stadium",
    14: "Holkar Cricket Stadium"
}

# Streamlit UI
st.set_page_config(page_title="IPL Score Predictor", layout="centered")
st.title("🏏 IPL Final Score Predictor")

# Select Inputs
batting_team = st.selectbox("Select Batting Team", list(team_mapping.values()))
bowling_team = st.selectbox("Select Bowling Team", list(team_mapping.values()))
venue = st.selectbox("Select Venue", list(venue_mapping.values()))

# Numeric Inputs
overs = st.slider("Overs Completed", min_value=5.0, max_value=20.0, value=10.0, step=0.1)
runs = st.number_input("Runs Scored", min_value=0)
wickets = st.number_input("Wickets Lost", min_value=0, max_value=10, value=2)

# Encode inputs
batting_encoded = list(team_mapping.keys())[list(team_mapping.values()).index(batting_team)]
bowling_encoded = list(team_mapping.keys())[list(team_mapping.values()).index(bowling_team)]
venue_encoded = list(venue_mapping.keys())[list(venue_mapping.values()).index(venue)]

# Prediction
if st.button("🎯 Predict Final Score"):
    input_data = np.array([[batting_encoded, bowling_encoded, venue_encoded, overs, runs, wickets]])
    scaled_input = scaler.transform(input_data)
    predicted_score = model.predict(scaled_input)[0][0]
    st.success(f"🏁 Predicted Final Score: {int(predicted_score)}")
