
import pandas as pd
import numpy as np
import logging
from src.data_ingestion import ingest_data
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Data cleaning pipeline
def clean_data(df):
    logging.info('Starting data cleaning...')
    # 1. Remove duplicates
    before = df.shape[0]
    df = df.drop_duplicates()
    after = df.shape[0]
    logging.info(f'Removed {before - after} duplicate rows.')

    # 2. Handle missing values (if any)
    missing = df.isnull().sum()
    if missing.any():
        logging.info('Handling missing values...')
        # Drop columns with too many missing values (threshold: 50%)
        thresh = len(df) * 0.5
        df = df.dropna(axis=1, thresh=thresh)
        # Fill remaining missing values with median
        df = df.fillna(df.median(numeric_only=True))
    else:
        logging.info('No missing values found.')

    # 3. Convert data types if needed (example: ensure numeric columns are float)
    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

if __name__ == '__main__':
    try:
        df = ingest_data()
        logging.info(f'Original shape: {df.shape}')
        df_clean = clean_data(df)
        logging.info(f'Cleaned shape: {df_clean.shape}')
        logging.info(f'Missing values after cleaning: {df_clean.isnull().sum().sum()}')
        df_clean.to_csv('data/creditcard_cleaned.csv', index=False)
        logging.info('Cleaned data saved to creditcard_cleaned.csv')
    except Exception as e:
        logging.error(f'Error during ingestion or cleaning: {e}')
