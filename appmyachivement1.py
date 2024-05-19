import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sklearn

# Set page configuration
st.set_page_config(page_title="Heart Disease Prediction", layout="wide")

# Load the model and feature names
model_path = 'rfc_model.pkl'
with open(model_path, 'rb') as file:
    loaded_model, feature_names = pickle.load(file)

# Streamlit app title
st.title('Heart Disease Prediction')

st.write("""
## Enter the patient's information:
""")

# Input fields for all features
def user_input_features():
    age = st.number_input('Age', min_value=0, max_value=120, value=50)
    sex = st.selectbox('Sex', options=[0, 1])  # Assuming 0 = Female, 1 = Male
    chest_pain_type = st.selectbox('Chest Pain Type', options=[0, 1, 2, 3])  # Assuming encoded values
    resting_bp_s = st.number_input('Resting Blood Pressure (systolic)', min_value=0, max_value=300, value=120)
    cholesterol = st.number_input('Cholesterol', min_value=0, max_value=600, value=200)
    fasting_blood_sugar = st.selectbox('Fasting Blood Sugar > 120 mg/dl', options=[0, 1])  # Assuming 0 = False, 1 = True
    resting_ecg = st.selectbox('Resting ECG', options=[0, 1, 2])  # Assuming encoded values
    max_heart_rate = st.number_input('Max Heart Rate Achieved', min_value=0, max_value=250, value=150)
    exercise_angina = st.selectbox('Exercise Induced Angina', options=[0, 1])  # Assuming 0 = No, 1 = Yes
    oldpeak = st.number_input('Oldpeak', min_value=0.0, max_value=10.0, value=1.0)
    st_slope = st.selectbox('ST Slope', options=[0, 1, 2])  # Assuming encoded values
    
    data = {
        'age': age,
        'sex': sex,
        'chest pain type': chest_pain_type,
        'resting bp s': resting_bp_s,
        'cholesterol': cholesterol,
        'fasting blood sugar': fasting_blood_sugar,
        'resting ecg': resting_ecg,
        'max heart rate': max_heart_rate,
        'exercise angina': exercise_angina,
        'oldpeak': oldpeak,
        'ST slope': st_slope
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

# Get user input
input_df = user_input_features()

# Display the user input
st.write("### Patient's information:")
st.write(input_df)

# Prediction button
if st.button('Predict'):
    # Make predictions
    prediction = loaded_model.predict(input_df)
    prediction_proba = loaded_model.predict_proba(input_df)

    # Display the prediction
    st.write("### Prediction")
    if prediction[0] == 1:
        st.write("Patient has heart disease.")
    else:
        st.write("Patient is healthy.")

    # Display the prediction probabilities
    st.write("### Prediction Probability")
    st.write(f"Probability of having heart disease: {prediction_proba[0][1]:.2f}")
    st.write(f"Probability of not having heart disease: {prediction_proba[0][0]:.2f}")
