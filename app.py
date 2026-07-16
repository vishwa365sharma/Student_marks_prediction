import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("model (1).pkl")
scaler = joblib.load("scaler (1).pkl")
le = joblib.load("label_encoder (1).pkl")

st.title("🎓 Student Marks Prediction")

age = st.number_input("Enter Age", min_value=1, max_value=100)
number_courses = st.number_input("Number of Courses", min_value=1)
time_study = st.number_input("Time Study (Hours)", min_value=0.0)

if st.button("Predict Marks"):
    data = np.array([[age, number_courses, time_study]])
    data = scaler.transform(data)
    prediction = model.predict(data)

    st.success(f"Predicted Marks: {prediction[0]:.2f}")
