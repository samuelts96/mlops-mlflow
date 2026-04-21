import json
import os

import requests
import streamlit as st

INFERENCE_BASE = os.getenv("INFERENCE_URL", "http://localhost:8000")
PREDICT_URL = f"{INFERENCE_BASE.rstrip('/')}/predict"
HEALTH_URL = f"{INFERENCE_BASE.rstrip('/')}/health"

st.title("Credit Card Fraud Detection")

# Backend health indicator
try:
    health = requests.get(HEALTH_URL, timeout=2)
    if health.ok:
        st.success(f"Backend online — {INFERENCE_BASE}")
    else:
        st.warning(f"Backend returned {health.status_code}")
except Exception:
    st.warning(f"Backend unreachable at {INFERENCE_BASE}")

st.write("Paste a JSON payload with named features:")

payload_text = st.text_area(
    "Input JSON",
    value=json.dumps({
        "Time": 0,
        "V1": 0, "V2": 0, "V3": 0, "V4": 0, "V5": 0, "V6": 0,
        "V7": 0, "V8": 0, "V9": 0, "V10": 0, "V11": 0, "V12": 0,
        "V13": 0, "V14": 0, "V15": 0, "V16": 0, "V17": 0,
        "V18": 0, "V19": 0, "V20": 0, "V21": 0, "V22": 0,
        "V23": 0, "V24": 0, "V25": 0, "V26": 0, "V27": 0,
        "V28": 0, "Amount": 100
    }, indent=2),
    height=300,
)

if st.button("Predict"):
    try:
        payload = json.loads(payload_text)
        r = requests.post(PREDICT_URL, json=payload, timeout=5)
        r.raise_for_status()
        result = r.json()
        label = "FRAUD" if result.get("prediction") == 1 else "LEGITIMATE"
        if label == "FRAUD":
            st.error(f"Result: {label}")
        else:
            st.success(f"Result: {label}")
        st.json(result)
    except json.JSONDecodeError:
        st.error("Invalid JSON in input.")
    except Exception as e:
        st.error(str(e))
