import streamlit as st
import pickle
import numpy as np
import pandas as pd 
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv("DATA.csv")
X = df.drop(columns="CO2")
y = df["CO2"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

# Load saved model, poly transformer, and feature list
model_path = "model.pkl"  
with open(model_path, "rb") as file:
    saved = pickle.load(file)
    model = saved["model"]
    poly = saved["poly"]
    features = saved["features"]

# Streamlit UI
st.title("🚗 Polynomial CO₂ Emission Prediction")
st.write("Enter Volume (m³) and Weight (kg) of Your Car to predict CO₂ emissions:")

Volume = st.number_input("Volume (m³)", min_value=0, max_value=50000, step=50, value=100)
Weight = st.number_input("Weight (kg)", min_value=0, max_value=50000, step=100, value=100)

if st.button("🔮 Predict CO₂"):
    input_data = np.array([[Volume, Weight]])
    input_poly = poly.transform(input_data)
    predicted_co2 = model.predict(input_poly)[0]
    
    st.success(f"🌿 Predicted CO₂ Emission: **{predicted_co2:.2f} g/km**")

    # Evaluate on test data
    X_test_poly = poly.transform(X_test[features])
    y_pred = model.predict(X_test_poly)
    mae = mean_absolute_error(y_test, y_pred)
    st.info(f"📊 Mean Absolute Error (MAE) on test data: **{mae:.2f}**")
