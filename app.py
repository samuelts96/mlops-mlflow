import streamlit as st
import requests
import json

st.title("Credit Card Fraud Detection")

st.write("Paste a JSON payload with named features:")

payload_text = st.text_area(
    "Input JSON",
    value="""{
  "Time": 0,
  "V1": 0,
  "V2": 0,
  "V3": 0,
  "V4": 0,
  "V5": 0,
  "V6": 0,
  "V7": 0,
  "V8": 0,
  "V9": 0,
  "V10": 0,
  "V11": 0,
  "V12": 0,
  "V13": 0,
  "V14": 0,
  "V15": 0,
  "V16": 0,
  "V17": 0,
  "V18": 0,
  "V19": 0,
  "V20": 0,
  "V21": 0,
  "V22": 0,
  "V23": 0,
  "V24": 0,
  "V25": 0,
  "V26": 0,
  "V27": 0,
  "V28": 0,
  "Amount": 100
}"""
)

if st.button("Predict"):
    try:
        payload = json.loads(payload_text)
        r = requests.post(
            "http://localhost:8000/predict",
            json=payload,
            timeout=5
        )
        st.json(r.json())
    except Exception as e:
        st.error(str(e))
