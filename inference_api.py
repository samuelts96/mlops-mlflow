import os
import pandas as pd

import mlflow.pyfunc
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.freeze_feature_contract import freeze_feature_contract


TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
if not TRACKING_URI:
    raise RuntimeError("MLFLOW_TRACKING_URI not set")

MODEL_NAME = os.getenv("MODEL_NAME", "CreditCardFraudModel")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")

model_uri = f"models:/{MODEL_NAME}@{MODEL_STAGE}"
model = mlflow.pyfunc.load_model(model_uri)

app = FastAPI(title="Fraud Detection Inference API")


class Transaction(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(tx: Transaction):
    try:
        df = pd.DataFrame([tx.dict()])
        df = freeze_feature_contract(df, mode="inference")
        pred = model.predict(df)
        return {"prediction": int(pred[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
