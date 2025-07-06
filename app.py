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
    0: "Wankhede Stadium, Mumbai",
    1: "Eden Gardens, Kolkata",
    2: "M. Chinnaswamy Stadium, Bengaluru",
    3: "Arun Jaitley Stadium, Delhi",
    4: "MA Chidambaram Stadium, Chennai",
    5: "Sawai Mansingh Stadium, Jaipur",
    6: "Rajiv Gandhi Intl. Stadium, Hyderabad",
    7: "Narendra Modi Stadium, Ahmedabad",
    8: "Punjab Cricket Association Stadium, Mohali",
    9: "Ekana Cricket Stadium, Lucknow",
    10: "Dr. Y.S. Rajasekhara Reddy ACA–VDCA Stadium, Visakhapatnam",
    11: "Holkar Cricket Stadium, Indore",
    12: "JSCA International Stadium Complex, Ranchi",
    13: "Shaheed Veer Narayan Singh Stadium, Raipur",
    14: "Green Park Stadium, Kanpur",
    15: "Barabati Stadium, Cuttack",
    16: "Brabourne Stadium, Mumbai",
    17: "Dr DY Patil Sports Academy, Navi Mumbai",
    18: "Maharashtra Cricket Association Stadium, Pune",
    19: "Saurashtra Cricket Association Stadium, Rajkot",
    20: "Greenfield International Stadium, Thiruvananthapuram",
    21: "Nehru Stadium, Kochi",
    22: "Guwahati Barsapara Stadium, Guwahati",
    
    # UAE (used in 2020, 2021)
    23: "Dubai International Stadium, Dubai",
    24: "Sharjah Cricket Stadium, Sharjah",
    25: "Sheikh Zayed Stadium, Abu Dhabi"
}
# UI
st.title("🏏 IPL Final Score Predictor")

# Dropdowns with placeholder option
batting_team = st.selectbox("Batting Team", ["Select"] + list(team_mapping.values()))
bowling_team = st.selectbox("Bowling Team", ["Select"] + list(team_mapping.values()))
venue = st.selectbox("Venue", ["Select"] + list(venue_mapping.values()))

# ⏱ Overs changed to slider
overs = st.slider("Overs Completed", min_value=5.0, max_value=20.0, value=10.0, step=0.1)
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
