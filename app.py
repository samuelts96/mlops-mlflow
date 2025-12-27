import streamlit as st
import requests

st.title("Credit Card Fraud Detection")

features = st.text_input("Enter features (comma-separated)")

if st.button("Predict"):
    payload = {
        "features": [float(x) for x in features.split(",")]
    }
    r = requests.post("http://localhost:8000/predict", json=payload)
    st.write(r.json())
