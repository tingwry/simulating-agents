import pandas as pd
import joblib
import os
import numpy as np
from train_model import (
    load_and_preprocess_data_binary_class,
    preprocess_unknown_values,
    MODEL_DIR,
    TEST_SIZE,
    RANDOM_STATE
)
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin

TEST_DATA_PATH = 'src/data/T0/test_with_lifestyle.csv'
PREDICTION_OUTPUT = 'src/recommendation/binary_classification/T0/predictions/transaction_predictions.csv'

class MultiColumnLabelEncoder(BaseEstimator, TransformerMixin):
    """Label encoder that handles multiple columns properly"""
    def __init__(self):
        self.encoders_ = {}
        self.feature_names_ = []
    
    def fit(self, X, y=None):
        # Convert to DataFrame if numpy array
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        
        self.feature_names_ = list(X.columns)
        
        for col in X.columns:
            le = LabelEncoder()
            # Handle NaN values by converting to string first
            le.fit(X[col].fillna('missing').astype(str))
            self.encoders_[col] = le
        return self
    
    def transform(self, X):
        # Convert to DataFrame if numpy array
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names_)
        
        result = np.zeros((X.shape[0], len(self.feature_names_)))
        
        for i, col in enumerate(X.columns):
            if col in self.encoders_:
                # Handle NaN values and transform
                encoded = self.encoders_[col].transform(X[col].fillna('missing').astype(str))
                result[:, i] = encoded
        
        return result

class BinaryEncoder(BaseEstimator, TransformerMixin):
    """Custom transformer for binary encoding"""
    def __init__(self):
        self.encoders_ = {}
        self.max_lens_ = {}
        self.feature_names_ = []
        
    def fit(self, X, y=None):
        # Convert to DataFrame if numpy array
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        
        self.feature_names_ = list(X.columns)
        
        for col in X.columns:
            le = LabelEncoder()
            # Handle NaN values
            col_data = X[col].fillna('missing').astype(str)
            encoded = le.fit_transform(col_data)
            self.encoders_[col] = le
            max_val = max(encoded) if len(encoded) > 0 else 0
            self.max_lens_[col] = len(format(max_val, 'b')) if max_val > 0 else 1
        return self
    
    def transform(self, X):
        # Convert to DataFrame if numpy array
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names_)
        
        result_cols = []
        result_data = []
        
        for col in X.columns:
            if col in self.encoders_:
                # Handle NaN values and transform
                col_data = X[col].fillna('missing').astype(str)
                encoded = self.encoders_[col].transform(col_data)
                binary_strings = [format(val, 'b').zfill(self.max_lens_[col]) for val in encoded]
                
                for i in range(self.max_lens_[col]):
                    result_cols.append(f'{col}_bin_{i}')
                    result_data.append([int(b[i]) for b in binary_strings])
        
        if result_data:
            result = np.column_stack(result_data)
        else:
            result = np.empty((X.shape[0], 0))
        
        return result

def run_predictions():
    # Load and preprocess full dataset
    df = pd.read_csv(TEST_DATA_PATH)
    df = preprocess_unknown_values(df)
    
    _, preprocessor, categories = load_and_preprocess_data_binary_class(TEST_DATA_PATH)
    
    # Prepare features and labels
    feature_cols = [col for col in df.columns if col not in categories]
    X_df = df[feature_cols]
    X_all = preprocessor.fit_transform(X_df)

    # Prepare results DataFrames
    predictions_df = pd.DataFrame()
    probabilities_df = pd.DataFrame()
    
    # Handle customer ID
    if 'CUST_ID' in df.columns:
        predictions_df['cust_id'] = df['CUST_ID']
        probabilities_df['cust_id'] = df['CUST_ID']
    elif 'cust_id' in df.columns:
        predictions_df['cust_id'] = df['cust_id']
        probabilities_df['cust_id'] = df['cust_id']
    else:
        raise ValueError("No customer ID column found in input data")
    
    for category in categories:
        model_path = f"{MODEL_DIR}/{category}_model.pkl"
        
        if not os.path.exists(model_path):
            print(f"Warning: Model for '{category}' not found. Using default zeros...")
            predictions_df[category] = 0
            probabilities_df[category] = 0.0
            continue

        model_data = joblib.load(model_path)
        clf = model_data['model']

        # Make predictions
        y_pred = clf.predict(X_all)
        
        # Handle probabilities carefully
        try:
            if hasattr(clf, 'predict_proba'):
                proba = clf.predict_proba(X_all)
                # Check if we have probabilities for both classes
                if proba.shape[1] == 2:
                    y_prob = proba[:, 1]  # Positive class probability
                else:
                    # Only one class predicted, use predictions as probabilities
                    y_prob = y_pred.astype(float)
            else:
                # Model doesn't support predict_proba
                y_prob = y_pred.astype(float)
        except Exception as e:
            print(f"Error getting probabilities for {category}: {str(e)}")
            y_prob = y_pred.astype(float)
        
        predictions_df[category] = y_pred.astype(int)
        probabilities_df[category] = y_prob
        print(f"Predictions done for: {category}")
    
    # Save outputs
    os.makedirs(os.path.dirname(PREDICTION_OUTPUT), exist_ok=True)
    predictions_df.to_csv(PREDICTION_OUTPUT, index=False)
    
    prob_output = PREDICTION_OUTPUT.replace('.csv', '_probabilities.csv')
    probabilities_df.to_csv(prob_output, index=False)
    
    print(f"\nBinary predictions saved to: {PREDICTION_OUTPUT}")
    print(f"Prediction probabilities saved to: {prob_output}")

if __name__ == "__main__":
    run_predictions()