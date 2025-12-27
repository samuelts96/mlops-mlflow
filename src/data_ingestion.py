import pandas as pd
import os

def ingest_data(data_path: str = 'creditcard.csv') -> pd.DataFrame:
    """
    Ingests the credit card dataset from the specified path.
    Args:
        data_path (str): Path to the creditcard.csv file.
    Returns:
        pd.DataFrame: Loaded dataset as a pandas DataFrame.
    Raises:
        FileNotFoundError: If the file does not exist at the given path.
        pd.errors.EmptyDataError: If the file is empty.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"File not found: {data_path}")
    try:
        df = pd.read_csv(data_path)
    except pd.errors.EmptyDataError:
        raise pd.errors.EmptyDataError(f"File is empty: {data_path}")
    return df

if __name__ == '__main__':
    try:
        df = ingest_data()
        print(f"Data ingested successfully. Shape: {df.shape}")
    except Exception as e:
        print(f"Error during ingestion: {e}")
