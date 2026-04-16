import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Fraud Detection Inference API")
FEATURES_PATH = Path("features.json")


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


class DummyFraudModel:
    """Deterministic mock used when no real model is available."""

    def predict(self, features: dict[str, float]) -> int:
        risk_score = 0.0

        amount = abs(features["Amount"])
        risk_score += min(amount / 1000.0, 1.5)

        risk_score += min(abs(features["V4"]) / 10.0, 0.5)
        risk_score += min(abs(features["V10"]) / 10.0, 0.5)
        risk_score += min(abs(features["V14"]) / 10.0, 0.5)
        risk_score += min(abs(features["V17"]) / 10.0, 0.5)

        return int(risk_score >= 1.2)


def load_feature_names() -> list[str]:
    if not FEATURES_PATH.exists():
        raise RuntimeError("features.json not found. Backend cannot validate inputs.")

    feature_list = json.loads(FEATURES_PATH.read_text())
    return [feature["name"] for feature in feature_list if feature["name"] != "Class"]


FEATURE_NAMES = load_feature_names()
model = DummyFraudModel()


def build_feature_payload(tx: Transaction) -> dict[str, float]:
    payload = tx.model_dump()
    missing = [feature for feature in FEATURE_NAMES if feature not in payload]
    if missing:
        raise ValueError(f"Missing required features: {', '.join(missing)}")

    return {feature: float(payload[feature]) for feature in FEATURE_NAMES}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(tx: Transaction):
    try:
        features = build_feature_payload(tx)
        pred = model.predict(features)
        return {"prediction": pred, "model_source": "mock"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
