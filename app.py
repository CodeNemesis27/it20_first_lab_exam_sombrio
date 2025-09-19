import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib

# Load the trained model and the scaler
# Ensure these files are in the same directory as this script
try:
    xgb_model = joblib.load('xgb_model.joblib')
    scaler = joblib.load('scaler.joblib')
except FileNotFoundError:
    st.error("Model or scaler files not found. Please train the model and save them first.")
    st.stop()

# --- Streamlit App UI ---
st.set_page_config(page_title="Red Wine Quality Predictor")

st.title("🍷 Red Wine Quality Predictor")
st.write("Enter the chemical attributes of the wine to predict if it is of 'good' or 'not good' quality.")

st.markdown("---")

# Input fields for the chemical attributes
st.header("Input Wine Attributes")

col1, col2, col3 = st.columns(3)
with col1:
    fixed_acidity = st.number_input("Fixed Acidity", value=7.4, format="%.2f")
    volatile_acidity = st.number_input("Volatile Acidity", value=0.70, format="%.2f")
    citric_acid = st.number_input("Citric Acid", value=0.00, format="%.2f")
with col2:
    residual_sugar = st.number_input("Residual Sugar", value=1.9, format="%.2f")
    chlorides = st.number_input("Chlorides", value=0.076, format="%.3f")
    free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", value=11.0, format="%.1f")
with col3:
    total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", value=34.0, format="%.1f")
    density = st.number_input("Density", value=0.9978, format="%.4f")
    pH = st.number_input("pH", value=3.51, format="%.2f")
    sulphates = st.number_input("Sulphates", value=0.56, format="%.2f")
    alcohol = st.number_input("Alcohol", value=9.4, format="%.1f")


# Create a DataFrame from user inputs
data = {
    'fixed acidity': fixed_acidity,
    'volatile acidity': volatile_acidity,
    'citric acid': citric_acid,
    'residual sugar': residual_sugar,
    'chlorides': chlorides,
    'free sulfur dioxide': free_sulfur_dioxide,
    'total sulfur dioxide': total_sulfur_dioxide,
    'density': density,
    'pH': pH,
    'sulphates': sulphates,
    'alcohol': alcohol
}
input_df = pd.DataFrame([data])

# Button to trigger prediction
if st.button("Predict Quality"):
    # Scale the input data using the trained scaler
    input_scaled = scaler.transform(input_df)

    # Get the prediction and confidence score
    prediction = xgb_model.predict(input_scaled)[0]
    prediction_proba = xgb_model.predict_proba(input_scaled)[0]
    
    # Display the results
    st.markdown("---")
    st.header("Prediction Results")

    if prediction == 1:
        st.success("✅ Prediction: Good Quality")
        st.write(f"The model is **{prediction_proba[1]*100:.2f}%** confident this wine is of good quality.")
    else:
        st.error("Prediction: Not Good Quality")
        st.write(f"The model is **{prediction_proba[0]*100:.2f}%** confident this wine is not of good quality.")

    st.balloons()