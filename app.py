import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# 🧠 Load model and scaler
model = load_model("ipl_score_predictor.keras")
scaler = joblib.load("scaler.pkl")

# ✅ Team and Venue mappings
team_mapping = {
    "CSK": 0, "MI": 1, "RCB": 2, "GT": 3, "DC": 4,
    "RR": 5, "SRH": 6, "KKR": 7, "LSG": 8, "PBKS": 9
}

venue_mapping = {
    "Wankhede Stadium": 0,
    "Eden Gardens": 1,
    "Narendra Modi Stadium": 2,
    "M. Chinnaswamy Stadium": 3,
    "Arun Jaitley Stadium": 4,
    "Sawai Mansingh Stadium": 5,
    "Rajiv Gandhi International Stadium": 6,
    "MA Chidambaram Stadium": 7,
    "HPCA Stadium": 8,
    "Ekana Cricket Stadium": 9,
    "PCA IS Bindra Stadium": 10,
    "Dr DY Patil Stadium": 11,
    "Brabourne Stadium": 12,
    "Maharashtra Cricket Stadium": 13,
    "Holkar Cricket Stadium": 14
}

# 🖥️ Streamlit UI
st.set_page_config(page_title="IPL Score Predictor", layout="centered")
st.title("🏏 IPL Final Score Predictor")

batting_team = st.selectbox("Batting Team", list(team_mapping.keys()))
bowling_team = st.selectbox("Bowling Team", list(team_mapping.keys()))
venue = st.selectbox("Match Venue", list(venue_mapping.keys()))

# 🎯 Match state inputs
overs = st.slider("Overs Completed", min_value=5.0, max_value=20.0, value=10.0, step=0.1)
runs = st.number_input("Runs Scored", min_value=0, step=1)
wickets = st.number_input("Wickets Lost", min_value=0, max_value=10, value=2, step=1)

# 🔄 Encode
batting_encoded = team_mapping[batting_team]
bowling_encoded = team_mapping[bowling_team]
venue_encoded = venue_mapping[venue]

# 🧠 Predict
if st.button("🎯 Predict Final Score"):
    input_features = np.array([[batting_encoded, bowling_encoded, venue_encoded, overs, runs, wickets]])
    scaled_input = scaler.transform(input_features)
    predicted_score = model.predict(scaled_input)[0][0]
    st.success(f"📢 Predicted Final Score: {int(predicted_score)} runs")
