import json
import pandas as pd

def freeze_feature_contract(df: pd.DataFrame, target_column: str = 'Class') -> None:
    # Define the feature contract
    feature_names = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount', target_column]
    
    # Fix feature names and order
    df = df[feature_names]
    
    # Fix data types according to the feature contract
    # Assuming all V features and Time, Amount are float64, target is int or categorical
    for col in feature_names:
        if col == target_column:
            # Convert target to int
            df[col] = df[col].astype(int)
        else:
            df[col] = df[col].astype(float)
    
    # Persist the feature list (names, order, dtypes) to features.json
    feature_list = []
    for col in feature_names:
        feature_list.append({
            'name': col,
            'dtype': str(df[col].dtype)
        })
    with open('features.json', 'w') as f:
        json.dump(feature_list, f, indent=4)

    return df

if __name__ == '__main__':
    import src.data_ingestion as di
    df = di.ingest_data('data/creditcard_cleaned.csv')
    df_fixed = freeze_feature_contract(df)
    print('Feature contract frozen and saved to features.json')
