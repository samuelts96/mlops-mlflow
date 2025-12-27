# MLflow Credit Card Fraud – Runbook

## Overview
This system provides:
- **Model tracking & registry** via MLflow
- **Inference API** via FastAPI
- **User interface** via Streamlit
- **Artifacts & data** stored in S3
- **Metadata** stored in PostgreSQL (RDS)

Architecture:
Streamlit → FastAPI → MLflow Registry → S3
                         ↓
                        RDS

---

## Services & Ports

| Service     | Port | Description |
|------------|------|-------------|
| MLflow     | 5000 | Tracking server & registry |
| FastAPI    | 8000 | Inference API |
| Streamlit  | 8501 | UI |

---

## Start Services (EC2)

### MLflow
mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri postgresql://postgres:dsclp-mlflowuser@mlflow-sam.crunhuiqgoz4.eu-west-2.rds.amazonaws.com:5432/mlflow-sam?sslmode=require \
  --default-artifact-root s3://mlflow-artifacts-sam

### FastAPI
uvicorn inference_api:app --host 0.0.0.0 --port 8000

### Streamlit
streamlit run app.py --server.port 8501 --server.address 0.0.0.0

---

## Local Access (SSH Tunnel)

ssh -L 18000:localhost:8000 -L 18501:localhost:8501 ec2-user@<EC2-IP>

- FastAPI docs → http://localhost:18000/docs
- Streamlit UI → http://localhost:18501

---

## Model Resolution

The inference service loads the model using **MLflow alias**:

models:/CreditCardFraudModel@production

To change the live model:
- Update the **production** alias in MLflow UI
- No redeploy required

---

## Test Inference

### FastAPI (JSON)
{
  "features": [
    0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,100
  ]
}

Expected response:
{ "prediction": 0 }

---

## Streamlit Input Example

0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,100

---

## Data Location

Training data:
s3://mlflow-artifacts-sam/creditcard.csv

Artifacts:
s3://mlflow-artifacts-sam/<run-id>/artifacts/

---

## IAM Requirements (EC2 Role)

Required permissions:
- s3:GetObject
- s3:PutObject
- s3:ListBucket

Scoped to:
arn:aws:s3:::mlflow-artifacts-sam
arn:aws:s3:::mlflow-artifacts-sam/*

---

## Common Failures & Fixes

### 403 Forbidden (S3)
- EC2 IAM role missing S3 permissions

### Model not found
- production alias not set
- Wrong model name

### Feature mismatch
- Input feature count mismatch

---

## Operational Rules
- Training jobs run on EC2
- Inference never loads models locally
- Promotion to production is manual or CI-driven
- Rollback = repoint alias

---

## Owner
ML Platform / MLOps
