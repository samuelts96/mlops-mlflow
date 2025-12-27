import mlflow
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

mlflow.set_tracking_uri("http://localhost:5000")

model = mlflow.pyfunc.load_model(
    "models:/CreditCardFraudModel@production"
)

app = FastAPI()

class Transaction(BaseModel):
    features: list[float]

@app.post("/predict")
def predict(tx: Transaction):
    df = pd.DataFrame([tx.features])
    pred = model.predict(df)
    return {"prediction": int(pred[0])}
