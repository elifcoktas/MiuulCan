import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import time

# Load Pre-Trained Model and Feature Names
model = joblib.load(r"C:\Users\okank\Downloads\xgb_model.pkl")
feature_names = joblib.load(r"C:\Users\okank\Downloads\feature_names.pkl")

# Page Configuration
st.set_page_config(page_title="Promotion Prediction with MiuulCan", layout="centered")

# Add Custom CSS for Compact Layout, Alignment, and Animation
st.markdown("""
    <style>
    body {
        background-color: #2b2d42;
        color: #edf2f4;
    }
    h1 {
        color: #FF6347;
        text-align: center;
        font-size: 36px;
        border-bottom: 3px solid #FF6347;
        padding-bottom: 10px;
    }
    .header {
        font-size: 20px;
        font-weight: bold;
        color: #FF6347;
        margin-top: 15px;
    }
    .stRadio, .stSelectbox, .stSlider, .stNumberInput {
        margin-bottom: 10px !important;  /* Reduce spacing */
    }
    .stColumn > div {
        padding-top: 0 !important;  /* Fix alignment issue for dropdown */
    }
    .happy-emoji {
        text-align: center;
        font-size: 100px;
        animation: dance 2s infinite;
    }
    .sad-emoji {
        text-align: center;
        font-size: 100px;
        animation: bounce 2s infinite;
    }
    .crying-emoji {
        text-align: center;
        font-size: 100px;
        animation: tears 2s infinite;
    }
    @keyframes dance {
        0%, 100% { transform: rotate(0deg); }
        25% { transform: rotate(-10deg); }
        50% { transform: rotate(10deg); }
        75% { transform: rotate(-10deg); }
    }
    @keyframes bounce {
        0%, 20%, 50%, 80%, 100% { transform: translateY(0); }
        40% { transform: translateY(-15px); }
        60% { transform: translateY(-7px); }
    }
    @keyframes tears {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(5px); }
    }
    </style>
""", unsafe_allow_html=True)

# Page Title
st.markdown("""
    <h1>Promotion Prediction with MiuulCan</h1>
    <p style="font-size: 20px; color: #4682B4; text-align: center;">Analyze promotion likelihood with intelligent insights.</p>
""", unsafe_allow_html=True)

# Add Mascot Image
st.image("MiuulCan.png", width=150, caption="Meet MiuulCan!")

# Input Fields
st.markdown('<div class="header">Personal Information</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    gender = st.radio("Select the gender:", ["Male", "Female"], help="Choose his/her gender.")
    age = st.number_input("Enter the age (18-80):", min_value=18, max_value=80, step=1, value=25)

with col2:
    education = st.selectbox("Education Level:", ["Bachelor's", "Master's & Above", "Other"], help="Select his/her highest education level.")
    recruitment_channel = st.selectbox("Recruitment Channel:", ["Sourcing", "Referral", "Others"], help="Select how he/she was recruited.")

st.markdown('<div class="header">Professional Details</div>', unsafe_allow_html=True)
tenure = st.number_input("Tenure in Years:", min_value=0.0, max_value=40.0, step=0.5, value=2.0, help="Enter his/her years of experience.")
department = st.selectbox("Department:", [
    "Sales & Marketing", "Operations", "Technology", "Analytics", "R&D",
    "Procurement", "Finance", "HR", "Legal"
], help="Select his/her current department.")
previous_year_rating = st.slider("Previous Year Rating (1-5):", 1, 5, 3, help="Select his/her performance score from last year.")
kpi_met = st.radio("KPI Met >80%?", ["Yes", "No"], help="Did he/she meet his/her KPIs this year?")
no_of_trainings = st.slider("Number of Trainings (1-10):", 1, 10, 3, help="Enter the number of trainings he/she attended.")
training_score = st.slider("Training Score (0-100):", 0, 100, 75, help="Enter his/her average training score.")

# Submit Button
if st.button("Predict"):
    # Prepare Input for the Model
    input_data = pd.DataFrame({
        'education_Bachelor\'s': [1 if education == "Bachelor's" else 0],
        'education_Master\'s & Above': [1 if education == "Master's & Above" else 0],
        'education_Other': [1 if education == "Other" else 0],
        'previous_year_rating': [previous_year_rating],
        'no_of_trainings': [no_of_trainings],
        'KPIs_met >80%': [1 if kpi_met == "Yes" else 0],
        'gender_m': [1 if gender == "Male" else 0],
        'recruitment_channel_Sourcing': [1 if recruitment_channel == "Sourcing" else 0],
        'recruitment_channel_Referral': [1 if recruitment_channel == "Referral" else 0],
        f'department_{department}': [1]
    })

    # Ensure All Features are Present and in Correct Order
    for col in feature_names:
        if col not in input_data.columns:
            input_data[col] = 0
    input_data = input_data[feature_names]

    # Make Prediction
    prediction_prob = model.predict_proba(input_data)[:, 1][0]  # Probability of promotion
    prediction = "Yes, probably!" if prediction_prob > 0.75 else "Unfortunately, no..."

    # Display Results
    if prediction == "Yes, probably!":
        st.balloons()
        st.success(f"🎉 Congratulations! Promotion is likely with {prediction_prob * 100:.2f}% probability!")
        st.markdown("""
            <div class='happy-emoji'>😊🕺💃🎉</div>
        """, unsafe_allow_html=True)
        time.sleep(5)  # Keep the dancing emojis for 5 seconds
    else:
        st.error(f"😢 Sorry! Promotion is unlikely with {prediction_prob * 100:.2f}% probability.")
        st.markdown("""
            <div class='sad-emoji'>😢😭</div>
        """, unsafe_allow_html=True)
        time.sleep(5)  # Keep the sad and crying emojis for 5 seconds
