import pandas as pd

def ingest_data(
    data_path: str = "s3://mlflow-artifacts-sam/creditcard.csv"
) -> pd.DataFrame:
    """
    Ingests the credit card dataset from S3.
    Args:
        data_path (str): S3 path to the CSV file.
    Returns:
        pd.DataFrame: Loaded dataset.
    Raises:
        ValueError: If the file is empty.
    """
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {data_path}")
    except pd.errors.EmptyDataError:
        raise ValueError(f"File is empty: {data_path}")

    return df


if __name__ == "__main__":
    df = ingest_data()
    print(f"Data ingested successfully. Shape: {df.shape}")
