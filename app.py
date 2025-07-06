import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load model and scaler
model = load_model("ipl_score_predictor.keras")
scaler = joblib.load("scaler.pkl")

# Title
st.title("🏏 IPL Final Score Predictor")
st.markdown("Predict the final score of an IPL innings based on current match situation.")

# Teams and venues
teams = [
    'Chennai Super Kings', 'Delhi Capitals', 'Gujarat Titans',
    'Kolkata Knight Riders', 'Lucknow Super Giants', 'Mumbai Indians',
    'Punjab Kings', 'Rajasthan Royals', 'Royal Challengers Bangalore', 'Sunrisers Hyderabad'
]

venues = [
    'Arun Jaitley Stadium', 'Barabati Stadium', 'Brabourne Stadium', 'Buffalo Park',
    'De Beers Diamond Oval', 'Dr DY Patil Sports Academy',
    'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium',
    'Dubai International Cricket Stadium', 'Eden Gardens', 'Feroz Shah Kotla',
    'Himachal Pradesh Cricket Association Stadium', 'Holkar Cricket Stadium',
    'JSCA International Stadium Complex', 'Kingsmead', 'M Chinnaswamy Stadium',
    'M. A. Chidambaram Stadium', 'M. A. Chidambaram Stadium, Chepauk',
    'Maharashtra Cricket Association Stadium', 'Narendra Modi Stadium',
    'New Wanderers Stadium', 'OUTsurance Oval',
    'Punjab Cricket Association IS Bindra Stadium', 'Punjab Cricket Association Stadium',
    'Rajiv Gandhi International Stadium', 'Sardar Patel Stadium, Motera',
    'Sawai Mansingh Stadium', 'Shaheed Veer Narayan Singh International Stadium',
    'Sharjah Cricket Stadium', 'Sheikh Zayed Stadium', 'St George’s Park',
    'Subrata Roy Sahara Stadium', 'SuperSport Park', 'Vidarbha Cricket Association Stadium',
    'Wankhede Stadium'
]

# UI Inputs
batting_team = st.selectbox("🏏 Batting Team", teams)
bowling_team = st.selectbox("🎯 Bowling Team", [team for team in teams if team != batting_team])
venue = st.selectbox("🏟️ Venue", venues)

overs = st.number_input("⏱ Overs Completed", min_value=5.0, max_value=20.0, step=0.1)
runs = st.number_input(" Runs Scored So Far", min_value=0, step=1)
wickets = st.number_input(" Wickets Lost So Far", min_value=0, max_value=10, step=1)

# Encode inputs manually using fixed dictionary (must match training order)
team_encoder = {name: idx for idx, name in enumerate(teams)}
venue_encoder = {name: idx for idx, name in enumerate(venues)}

# Predict button
if st.button("📊 Predict Final Score"):
    input_data = np.array([[
        team_encoder[batting_team],
        team_encoder[bowling_team],
        venue_encoder[venue],
        overs,
        runs,
        wickets
    ]])

    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)

    st.success(f"🏆 Predicted Final Score: {int(prediction[0][0])}")
