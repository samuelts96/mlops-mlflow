import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.freeze_feature_contract import freeze_feature_contract

mlflow.set_tracking_uri("http://localhost:5000")

model = mlflow.pyfunc.load_model(
    "models:/CreditCardFraudModel@production"
)

app = FastAPI(title="Credit Card Fraud Inference API")


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


@app.post("/predict")
def predict(tx: Transaction):
    try:
        # 1. Build DataFrame from named fields
        df = pd.DataFrame([tx.dict()])

        # 2. Enforce feature contract
        df = freeze_feature_contract(
            df,
            mode="inference"
        )

        # 3. Predict
        pred = model.predict(df)

        return {"prediction": int(pred[0])}

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Inference failed: {str(e)}"
        )
