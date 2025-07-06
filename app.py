import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# Load model and scaler
model = load_model("ipl_score_predictor.keras")
scaler = joblib.load("scaler.pkl")

# Team & venue mappings
team_mapping = {
    0: "CSK", 1: "MI", 2: "RCB", 3: "GT", 4: "DC", 5: "RR",
    6: "SRH", 7: "KKR", 8: "LSG", 9: "PBKS"
}
venue_mapping = {
    0: "Wankhede Stadium",
    1: "Eden Gardens",
    2: "Narendra Modi Stadium",
    3: "M. Chinnaswamy Stadium",
    4: "Arun Jaitley Stadium",
    5: "Sawai Mansingh Stadium",
    6: "Rajiv Gandhi Intl. Stadium",
    7: "MA Chidambaram Stadium",
    8: "Punjab Cricket Association IS Bindra Stadium",
    9: "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium",
    10: "Barabati Stadium",
    11: "Himachal Pradesh Cricket Association Stadium",
    12: "Dr. DY Patil Sports Academy",
    13: "Brabourne Stadium",
    14: "Shaheed Veer Narayan Singh Stadium",
    15: "Holkar Cricket Stadium",
    16: "Greenfield International Stadium",
    17: "Maharashtra Cricket Association Stadium",
    18: "Saurashtra Cricket Association Stadium",
    19: "Green Park",
    20: "JSCA International Stadium Complex",
    21: "Sheikh Zayed Stadium, Abu Dhabi",
    22: "Dubai International Stadium",
    23: "Sharjah Cricket Stadium"
}

# UI
st.title("🏏 IPL Final Score Predictor")

# Dropdowns with placeholder option
batting_team = st.selectbox("Batting Team", ["Select"] + list(team_mapping.values()))
bowling_team = st.selectbox("Bowling Team", ["Select"] + list(team_mapping.values()))
venue = st.selectbox("Venue", ["Select"] + list(venue_mapping.values()))

overs = st.number_input("Overs Completed", min_value=5.0, max_value=20.0, step=0.1)
runs = st.number_input("Runs Scored", min_value=0)
wickets = st.number_input("Wickets Lost", min_value=0, max_value=10)

if st.button("Predict Score"):
    if "Select" in (batting_team, bowling_team, venue):
        st.warning("⚠️ Please select all fields (team and venue) before predicting.")
    elif batting_team == bowling_team:
        st.error("⚠️ Batting and bowling teams must be different.")
    else:
        # Encoding
        batting_encoded = list(team_mapping.keys())[list(team_mapping.values()).index(batting_team)]
        bowling_encoded = list(team_mapping.keys())[list(team_mapping.values()).index(bowling_team)]
        venue_encoded = list(venue_mapping.keys())[list(venue_mapping.values()).index(venue)]

        # Prepare input
        input_array = np.array([[batting_encoded, bowling_encoded, venue_encoded, overs, runs, wickets]])
        scaled_input = scaler.transform(input_array)

        prediction = model.predict(scaled_input)
        st.success(f"🎯 Predicted Final Score: {int(prediction[0][0])}")
