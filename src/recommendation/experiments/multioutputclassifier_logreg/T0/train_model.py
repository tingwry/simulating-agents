import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import json
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline

# DATA_PATH = 'src/recommendation/binary_classification_rand_reg/data/demog_ranking_grouped_catbased.csv'
DATA_PATH = 'src/recommendation/multilabel/data/demog_grouped_catbased.csv'
MODEL_DIR = 'src/recommendation/multioutputclassifier_logreg/T0/models_grouped_catbased'
METRICS_DIR = 'src/recommendation/multioutputclassifier_logreg/T0/metrics_grouped_catbased'

RANDOM_STATE = 42
TEST_SIZE = 0.2

# categories = ['charity', 'loan', 'utility', 'investment', 'finance', 'shopping',
#                  'personal_care', 'medical', 'home_and_living', 'insurance', 'automotive',
#                  'restaurant', 'business', 'entertainment', 'bank', 'education', 'pet_care',
#                  'government', 'travel', 'transportation', 'visit', 'system_dpst', 'other', 
#                  'financial_services', 'health_and_care', 'home_lifestyle', 'transport_travel',	
#                  'leisure', 'public_services']

categories = ['loan', 'utility', 'finance', 'shopping', 
                 'financial_services', 'health_and_care', 'home_lifestyle', 'transport_travel',	
                 'leisure', 'public_services']

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

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

def preprocess_unknown_values(df):
    """Convert all 'Unknown' strings to NaN values"""
    df = df.copy()
    df = df.replace('Unknown', np.nan)
    
    # Also handle case variations and different representations
    unknown_variations = ['unknown', 'UNKNOWN', 'Unknown', 'None', 'none', 'N/A', 'n/a', 'NaN', 'nan']
    for variation in unknown_variations:
        df = df.replace(variation, np.nan)
    
    return df

def load_and_preprocess_data(DATA_PATH):
    """Load and preprocess data with improved encoding"""
    df = pd.read_csv(DATA_PATH)
    df = preprocess_unknown_values(df)
    
    # Define features and categories
    numerical_features = ['Number of Children', 'Age']
    label_encode_features = ['Gender', 'Education level']
    binary_encode_features = ['Marital status', 'Region', 'Occupation Group']

    # Filter features that actually exist in the dataframe
    numerical_features = [col for col in numerical_features if col in df.columns]
    label_encode_features = [col for col in label_encode_features if col in df.columns]
    binary_encode_features = [col for col in binary_encode_features if col in df.columns]
    
    # Create transformers list
    transformers = []
    
    if numerical_features:
        transformers.append(('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median'))
        ]), numerical_features))
    
    if label_encode_features:
        transformers.append(('label', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', MultiColumnLabelEncoder())
        ]), label_encode_features))
    
    if binary_encode_features:
        transformers.append(('binary', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', BinaryEncoder())
        ]), binary_encode_features))
    
    preprocessor = ColumnTransformer(transformers=transformers)

    # Fit the preprocessor to the data
    feature_cols = [col for col in df.columns if col not in categories]
    preprocessor.fit(df[feature_cols])
    
    return df, preprocessor, categories


def train_and_evaluate_model():
    """Train and evaluate sklearn multi-output classifier"""
    # Load and preprocess data (same as before)
    df, preprocessor, categories = load_and_preprocess_data(DATA_PATH)
    
    # Get feature columns (exclude target categories)
    feature_cols = [col for col in df.columns if col not in categories]
    X = preprocessor.fit_transform(df[feature_cols])
    y = df[categories].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Initialize base classifier
    # lr = LogisticRegression(solver='lbfgs', random_state=RANDOM_STATE)
    lr = LogisticRegression(
            random_state=RANDOM_STATE,
            class_weight='balanced',
            max_iter=1000,
            solver='liblinear'
        )
    # Create MultiOutputClassifier
    classifier = MultiOutputClassifier(lr)
    
    # Create pipeline
    pipeline = Pipeline([
        ('classifier', classifier)
    ])
    
    # Train model
    print("Training MultiOutput CatBoost model...")
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)
    
    # Calculate metrics for each category
    results = {}
    for i, category in enumerate(categories):
        # Get probabilities for positive class (assuming binary classification)
        proba = y_pred_proba[i][:, 1] if len(y_pred_proba[i].shape) > 1 else y_pred_proba[i]
        
        tp = np.sum((y_pred[:, i] == 1) & (y_test[:, i] == 1))
        fp = np.sum((y_pred[:, i] == 1) & (y_test[:, i] == 0))
        fn = np.sum((y_pred[:, i] == 0) & (y_test[:, i] == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        results[category] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    # Save model and preprocessing components
    joblib.dump({
        'model': pipeline,
        'preprocessor': preprocessor,
        'scaler': scaler,
        'categories': categories
    }, f"{MODEL_DIR}/multioutput_model.pkl")
    
    # Save metrics
    with open(f"{METRICS_DIR}/training_metrics.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nTraining completed. Model and metrics saved.")
    
    if results:
        print("\nTraining Metrics Summary:")
        results_df = pd.DataFrame(results).T
        print(results_df.round(4))
        
        print(f"\nAverage Metrics:")
        print(f"Precision: {results_df['precision'].mean():.4f}")
        print(f"Recall: {results_df['recall'].mean():.4f}")
        print(f"F1 Score: {results_df['f1_score'].mean():.4f}")

if __name__ == "__main__":
    train_and_evaluate_model()