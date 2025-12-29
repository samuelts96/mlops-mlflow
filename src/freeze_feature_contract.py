import json
import pandas as pd
from pathlib import Path

FEATURES_PATH = Path("features.json")


def freeze_feature_contract(
    df: pd.DataFrame,
    target_column: str = "Class",
    mode: str = "train",  # "train" | "inference"
) -> pd.DataFrame:
    if mode == "train":
        feature_names = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", target_column]

        df = df[feature_names]

        for col in feature_names:
            if col == target_column:
                df[col] = df[col].astype(int)
            else:
                df[col] = df[col].astype(float)

        feature_list = [
            {"name": col, "dtype": str(df[col].dtype)}
            for col in feature_names
        ]

        FEATURES_PATH.write_text(json.dumps(feature_list, indent=4))
        return df

    elif mode == "inference":
        if not FEATURES_PATH.exists():
            raise RuntimeError("features.json not found. Train a model first.")

        feature_list = json.loads(FEATURES_PATH.read_text())
        feature_names = [f["name"] for f in feature_list if f["name"] != target_column]

        df = df[feature_names]

        for f in feature_list:
            if f["name"] != target_column:
                df[f["name"]] = df[f["name"]].astype(f["dtype"])

        return df

    else:
        raise ValueError("mode must be 'train' or 'inference'")
