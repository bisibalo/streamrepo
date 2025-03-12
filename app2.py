import streamlit as st
import pickle
import numpy as np

# Load the trained model
filename = "stream_model.pkl"
with open(filename, "rb") as file:
    model = pickle.load(file)

# Streamlit UI
st.title("Bank Account Prediction App")

# Input fields for features
household_size = st.number_input("Household Size", min_value=1, max_value=20, step=1)
age_of_respondent = st.number_input("Age of Respondent", min_value=18, max_value=100, step=1)
country_rwanda = st.checkbox("Country: Rwanda")
country_tanzania = st.checkbox("Country: Tanzania")
education_vocational = st.checkbox("Vocational/Specialised Training")
job_farming = st.checkbox("Job: Farming and Fishing")
job_government = st.checkbox("Job: Formally Employed Government")
job_private = st.checkbox("Job: Formally Employed Private")
job_self_employed = st.checkbox("Job: Self Employed")

# Convert checkbox inputs to 0/1
features = np.array([
    household_size, 
    age_of_respondent, 
    int(country_rwanda), 
    int(country_tanzania), 
    int(education_vocational), 
    int(job_farming), 
    int(job_government), 
    int(job_private), 
    int(job_self_employed)
]).reshape(1, -1)

# Prediction Button
if st.button("Predict"):
    prediction = model.predict(features)
    result = "Has a Bank Account" if prediction[0] == 1 else "No Bank Account"
    st.success(f"Prediction: {result}")
