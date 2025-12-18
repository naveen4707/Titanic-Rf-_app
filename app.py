import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="centered")

# --- CUSTOM CSS FOR NEAT UI ---
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #0e1117;
        color: white;
    }
    .result-text {
        font-size: 24px;
        font-weight: bold;
        text-align: center;
        padding: 20px;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    model = pickle.load(open('titanic_model.pkl', 'rb'))
    features = pickle.load(open('features.pkl', 'rb'))
    return model, features

try:
    model, model_features = load_assets()
except:
    st.error("Error: Model files not found. Please ensure 'titanic_model.pkl' and 'features.pkl' are in the directory.")
    st.stop()

# --- HEADER ---
st.title("🚢 RMS Titanic Survival Predictor")
st.markdown("Enter the passenger's details below to predict their likelihood of survival.")
st.divider()

# --- INPUT FORM ---
with st.container():
    col1, col2 = st.columns(2)
    
    with col1:
        pclass = st.selectbox("Ticket Class (Pclass)", [1, 2, 3], help="1 = First Class, 3 = Third Class")
        sex = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", min_value=0, max_value=100, value=30)
        fare = st.number_input("Fare (Ticket Price)", min_value=0.0, value=32.0)

    with col2:
        sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0, value=0)
        parch = st.number_input("Parents/Children Aboard", min_value=0, value=0)
        embarked = st.selectbox("Port of Embarkation", ["Cherbourg", "Queenstown", "Southampton"])

# --- PREDICTION LOGIC ---
if st.button("Predict Survival Probability"):
    # 1. Map Inputs
    sex_numeric = 0 if sex == "Male" else 1
    
    # 2. Handle One-Hot Encoded Embarked (Matching pd.get_dummies logic)
    # Embarked_Q and Embarked_S (C is dropped in training via drop_first=True)
    emb_q = 1 if embarked == "Queenstown" else 0
    emb_s = 1 if embarked == "Southampton" else 0
    
    # 3. Create Feature Vector
    # Order must be: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked_Q, Embarked_S
    input_dict = {
        'Pclass': pclass,
        'Sex': sex_numeric,
        'Age': age,
        'SibSp': sibsp,
        'Parch': parch,
        'Fare': fare,
        'Embarked_Q': emb_q,
        'Embarked_S': emb_s
    }
    
    input_df = pd.DataFrame([input_dict])
    
    # Ensure column order matches training exactly
    input_df = input_df[model_features]
    
    # 4. Make Prediction
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0]

    # --- DISPLAY RESULTS ---
    st.divider()
    if prediction == 1:
        st.balloons()
        st.markdown(f'<div class="result-text" style="background-color: #d4edda; color: #155724;">✅ Likely Survived<br><small>Probability: {prob[1]:.2%}</small></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="result-text" style="background-color: #f8d7da; color: #721c24;">❌ Unlikely to Have Survived<br><small>Probability: {prob[0]:.2%}</small></div>', unsafe_allow_html=True)

st.divider()
st.info("Note: This model is based on historical Titanic data and uses a Random Forest Classifier with a max depth of 4.")
