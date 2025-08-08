import streamlit as st
from joblib import load
import numpy as np

# Load the model
rf_best = load('D:\GITHUB\my_streamlit_app\rf_best_model.joblib')

st.title("My ML Model Prediction")

# Example: Input fields depending on your model features
# Let's say your model expects two features: feature1 and feature2
id = st.number_input("id")
group = st.number_input("group")
age = st.number_input("age")
sex = st.number_input("sex")
uric_se1 = st.number_input("uric_se1")
uremia = st.number_input("uremia")
outcome = st.number_input("outcome")

if st.button("Predict"):
    input_data = np.array([[id, group, age, sex, uric_se1, uremia, outcome]])
    prediction = rf_best.predict(input_data)
    st.write(f"Prediction: {prediction[0]}")
