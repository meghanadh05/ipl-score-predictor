import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# 🧠 Load trained model
model = load_model("ipl_score_predictor.keras")

# 👥 Hardcoded options (based on training data)
teams = [
    'Chennai Super Kings', 'Delhi Capitals', 'Kings XI Punjab', 'Kolkata Knight Riders',
    'Mumbai Indians', 'Rajasthan Royals', 'Royal Challengers Bangalore', 'Sunrisers Hyderabad'
]
venues = [
    'Eden Gardens', 'Wankhede Stadium', 'M Chinnaswamy Stadium', 'Feroz Shah Kotla',
    'MA Chidambaram Stadium', 'Punjab Cricket Association Stadium', 'Rajiv Gandhi International Stadium'
]

# 🚀 Streamlit UI
st.title("🏏 IPL Score Predictor")
st.markdown("Predict the final score of an IPL innings based on current match context.")

# 🧾 User Inputs
venue = st.selectbox("Venue", venues)
batting_team = st.selectbox("Batting Team", teams)
bowling_team = st.selectbox("Bowling Team", [team for team in teams if team != batting_team])
over_number = st.slider("Overs completed", 5.0, 20.0, step=0.1)
runs_till_now = st.number_input("Runs scored so far", min_value=0)
wickets_till_now = st.slider("Wickets fallen", 0, 10)

# 📦 Dummy label encoders (replace with real encoders if saved)
def dummy_encode(value, options):
    return options.index(value) if value in options else 0

# 🧮 Prediction button
if st.button("Predict Score"):
    # Encode categorical values
    input_array = np.array([
        dummy_encode(batting_team, teams),
        dummy_encode(bowling_team, teams),
        dummy_encode(venue, venues),
        over_number,
        runs_till_now,
        wickets_till_now
    ]).reshape(1, -1)

    # Apply MinMax scaling manually (same as used in training)
    scaler = MinMaxScaler()
    scaler.fit([[0, 0, 0, 5.0, 0, 0], [7, 7, 6, 20.0, 250, 10]])  # fit on training-like bounds
    input_scaled = scaler.transform(input_array)

    # Predict
    predicted_score = model.predict(input_scaled)
    predicted_score = int(predicted_score[0][0])

    st.success(f"🎯 Predicted Final Score: {predicted_score} runs")
