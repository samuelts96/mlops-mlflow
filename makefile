.PHONY: install mlflow api ui tunnel all

PYTHON=python3
MLFLOW_PORT=5000
API_PORT=8000
UI_PORT=8501

MLFLOW_BACKEND=postgresql://postgres:dsclp-mlflowuser@mlflow-sam.crunhuiqgoz4.eu-west-2.rds.amazonaws.com:5432/mlflow-sam?sslmode=require
MLFLOW_ARTIFACT_ROOT=s3://mlflow-artifacts-sam

install:
	pip install --upgrade pip
	pip install mlflow psycopg2-binary boto3 s3fs fastapi uvicorn streamlit requests

mlflow:
	mlflow server \
		--host 0.0.0.0 \
		--port $(MLFLOW_PORT) \
		--backend-store-uri $(MLFLOW_BACKEND) \
		--default-artifact-root $(MLFLOW_ARTIFACT_ROOT)

api:
	uvicorn inference_api:app --host 0.0.0.0 --port $(API_PORT)

ui:
	streamlit run app.py --server.port $(UI_PORT) --server.address 0.0.0.0

all:
	make mlflow & \
	make api & \
	make ui
