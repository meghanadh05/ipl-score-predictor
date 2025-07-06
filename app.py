import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# 🧠 Load Keras model
model = load_model("ipl_score_predictor.keras")

# 🧪 Load scaler
scaler = joblib.load("scaler.pkl")

# 🧾 Hardcoded label encoders (you can use joblib if needed)
team_mapping = {
    0: "CSK", 1: "MI", 2: "RCB", 3: "GT", 4: "DC", 5: "RR", 6: "SRH", 7: "KKR", 8: "LSG", 9: "PBKS"
}
venue_mapping = {
    0: "Wankhede", 1: "Eden Gardens", 2: "Narendra Modi Stadium", 3: "Chinnaswamy"
}

# UI
st.title("🏏 IPL Final Score Predictor")

batting_team = st.selectbox("Batting Team", list(team_mapping.values()))
bowling_team = st.selectbox("Bowling Team", list(team_mapping.values()))
venue = st.selectbox("Venue", list(venue_mapping.values()))
overs = st.slider("Overs Completed", 5.0, 20.0, 10.0, 0.1)
runs = st.number_input("Runs Scored", min_value=0)
wickets = st.slider("Wickets Lost", 0, 10, 2)

# Encode
batting_encoded = list(team_mapping.keys())[list(team_mapping.values()).index(batting_team)]
bowling_encoded = list(team_mapping.keys())[list(team_mapping.values()).index(bowling_team)]
venue_encoded = list(venue_mapping.keys())[list(venue_mapping.values()).index(venue)]

# 🔍 Predict
input_array = np.array([[batting_encoded, bowling_encoded, venue_encoded, overs, runs, wickets]])
scaled_input = scaler.transform(input_array)

if st.button("Predict Score"):
    prediction = model.predict(scaled_input)
    st.success(f"🎯 Predicted Final Score: {int(prediction[0][0])}")
