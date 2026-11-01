import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("model.joblib")

st.set_page_config(page_title="Taxi Fare Prediction", page_icon=":taxi:", layout="centered")

st.title(" Taxi Fare Prediction")

Distance = st.number_input("Distance(km)", min_value=1.0, max_value=100.0,value=1.0)

Passengers = st.number_input("Passengers", min_value=1, max_value=10,value=2)

Pickup_City = st.selectbox("Pickup City", ["Chennai", "Hyderabad", "Mumbai", "Mysuru","Bengaluru"])

Drop_City = st.selectbox("Drop City", ["Chennai", "Hyderabad", "Mumbai", "Mysuru","Bengaluru"])

Hour=st.number_input("Hour", min_value=0, max_value=23,value=12)

Day_Type = st.selectbox("Day Type", ["Weekday", "Weekend"])

Traffic = st.selectbox("Traffic", ["Low", "Medium", "High"])

Vehicle_Type = st.selectbox("Vehicle Type", ["Mini", "Sedan", "SUV"])

Weather = st.selectbox("Weather", ["Clear", "Rain", "Cloudy"])


if st.button("Predict Taxi Fare"):
    input_data = pd.DataFrame({
        "Distance": [Distance],
        "Passengers": [Passengers],
        "Pickup_City": [Pickup_City],
        "Drop_City": [Drop_City],
        "Hour": [Hour],
        "Day_Type": [Day_Type],
        "Traffic": [Traffic],
        "Vehicle_Type": [Vehicle_Type],
        "Weather": [Weather]
    })

    input_data = pd.get_dummies(input_data, drop_first=True)

    
    model_columns = model.feature_names_in_

    input_data = input_data.reindex(columns=model_columns, fill_value=0)

    prediction = model.predict(input_data)
    st.success(f"Estimated Taxi Fare: ₹{prediction[0]:.2f}")