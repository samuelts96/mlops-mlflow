import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

def scale_features(input_path='data/creditcard_cleaned.csv', output_path='data/creditcard_scaled.csv'):
    """
    Scales the 'Time' and 'Amount' features using StandardScaler and saves the result.
    Args:
        input_path (str): Path to the cleaned data CSV.
        output_path (str): Path to save the scaled data CSV.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    df = pd.read_csv(input_path)
    scaler = StandardScaler()
    df[[ 'Amount']] = scaler.fit_transform(df[[ 'Amount']])
    df.to_csv(output_path, index=False)
    print(f"Scaled data saved to {output_path}")

if __name__ == '__main__':
    try:
        scale_features()
    except Exception as e:
        print(f"Error during scaling: {e}")
