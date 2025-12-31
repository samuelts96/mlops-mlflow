import os
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import mlflow
import mlflow.sklearn

from src.freeze_feature_contract import freeze_feature_contract


# Set MLflow tracking URI (SQLite)
# mlflow.set_tracking_uri(
#     f"sqlite:///{os.path.abspath('mlflow.db').replace('\\', '/')}"
# )

# mlflow.set_tracking_uri("postgresql://postgres:dsclp-mlflowuser@mlflow-sam.crunhuiqgoz4.eu-west-2.rds.amazonaws.com:5432/mlflow-sam?sslmode=require")
tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
if not tracking_uri:
    raise RuntimeError("MLFLOW_TRACKING_URI is not set")

mlflow.set_tracking_uri(tracking_uri)


def train_baseline():
    # Load scaled dataset
    data_path = 'data/creditcard_scaled.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Scaled data file not found: {data_path}")

    df = pd.read_csv(data_path)

    # Enforce feature contract
    df = freeze_feature_contract(df)

    # Split features / target
    target_column = 'Class'
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # MLflow experiment
    mlflow.set_experiment('Credit Card Fraud Baseline')

    with mlflow.start_run(run_name='baseline_logistic_regression'):
        model = LogisticRegression(max_iter=2000, random_state=42)
        model.fit(X_train, y_train)

        # Validation predictions
        y_val_pred = model.predict(X_val)

        # Metrics
        accuracy = accuracy_score(y_val, y_val_pred)
        precision = precision_score(y_val, y_val_pred)
        recall = recall_score(y_val, y_val_pred)
        f1 = f1_score(y_val, y_val_pred)

        # Log metrics
        mlflow.log_metric('accuracy', accuracy)
        mlflow.log_metric('precision', precision)
        mlflow.log_metric('recall', recall)
        mlflow.log_metric('f1_score', f1)

        # Log params
        mlflow.log_param('model_type', 'LogisticRegression')
        mlflow.log_param('max_iter', 2000)
        mlflow.log_param('random_state', 42)

        # Log & register model (auto-versioned)
        mlflow.sklearn.log_model(
            model,
            name='model',
            registered_model_name='CreditCardFraudModel'
        )

        # Tag run
        mlflow.set_tag('baseline', 'true')

        print(
            f"Run completed | "
            f"accuracy={accuracy:.4f}, "
            f"precision={precision:.4f}, "
            f"recall={recall:.4f}, "
            f"f1={f1:.4f}"
        )


if __name__ == '__main__':
    train_baseline()
