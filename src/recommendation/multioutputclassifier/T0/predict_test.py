import pandas as pd
import joblib
import os
import numpy as np
from train_model import (
    load_and_preprocess_data,
    preprocess_unknown_values,
    MODEL_DIR
)
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
import torch
from torch.utils.data import DataLoader, TensorDataset

TEST_DATA_PATH = 'src/data/T0/test_with_lifestyle.csv'
PREDICTION_OUTPUT = 'src/recommendation/multioutputclassifier/T0/predictions/transaction_predictions_grouped_catbased.csv'

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
    # Load and preprocess data
    df = pd.read_csv(TEST_DATA_PATH)
    df = preprocess_unknown_values(df)
    _, preprocessor, categories = load_and_preprocess_data(TEST_DATA_PATH)

    model_data = joblib.load(f"{MODEL_DIR}/multioutput_model.pkl")
    pipeline = model_data['model']
    categories = ['loan','utility','finance','shopping','financial_services', 
                 'health_and_care', 'home_lifestyle', 'transport_travel',	
                 'leisure', 'public_services']
    
    preprocessor = model_data['preprocessor']
    scaler = model_data['scaler']

    # Prepare features
    feature_cols = [col for col in df.columns if col not in categories]
    X_all = preprocessor.transform(df[feature_cols])  # Use transform instead of fit_transform
    
    # Standardize features
    X_all = scaler.transform(X_all)
    

    # Initialize output DataFrames
    binary_predictions = pd.DataFrame()
    prediction_scores = pd.DataFrame()
    
    # Handle customer ID
    id_col = 'CUST_ID' if 'CUST_ID' in df.columns else 'cust_id'
    binary_predictions['cust_id'] = df[id_col]
    prediction_scores['cust_id'] = df[id_col]
    
    # Get predictions
    y_pred = pipeline.predict(X_all)
    y_proba = pipeline.predict_proba(X_all)  # List of arrays (one per class)
    
    # Fill DataFrames
    for i, category in enumerate(categories):
        binary_predictions[category] = y_pred[:, i]
        # Get probabilities for positive class
        prediction_scores[category] = y_proba[i][:, 1]
    
    # Save outputs
    os.makedirs(os.path.dirname(PREDICTION_OUTPUT), exist_ok=True)
    binary_predictions.to_csv(PREDICTION_OUTPUT, index=False)
    
    scores_output = PREDICTION_OUTPUT.replace('.csv', '_scores.csv')
    prediction_scores.to_csv(scores_output, index=False)
    
    print(f"\nBinary predictions saved to: {PREDICTION_OUTPUT}")
    print(f"Prediction scores saved to: {scores_output}")

    return binary_predictions, prediction_scores



if __name__ == "__main__":
    run_predictions()