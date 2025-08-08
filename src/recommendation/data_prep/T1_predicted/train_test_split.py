import pandas as pd
from sklearn.model_selection import train_test_split

def train_test_split_and_save(df, target_columns, test_size=0.2, random_state=42, 
                             train_file='train_data.csv', test_file='test_data.csv'):
    """
    Perform train-test split and save to CSV files
    
    Parameters:
    - df: Pandas DataFrame containing the data
    - target_columns: List of column names for the target variables
    - test_size: Proportion of data for test set (default 0.2)
    - random_state: Random seed for reproducibility (default 42)
    - train_file: Filename for training data (default 'train_data.csv')
    - test_file: Filename for test data (default 'test_data.csv')
    
    Returns:
    - train_df: Training DataFrame
    - test_df: Test DataFrame
    """
    
    # Separate features and targets
    feature_cols = [col for col in df.columns if col not in target_columns]
    X = df[feature_cols]
    y = df[target_columns]
    
    # Perform train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state
    )
    
    # Combine features and targets back into DataFrames
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    # Save to CSV files
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)
    
    print(f"Training data saved to {train_file} ({len(train_df)} rows)")
    print(f"Test data saved to {test_file} ({len(test_df)} rows)")
    
    return train_df, test_df