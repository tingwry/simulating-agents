from catboost import CatBoostRegressor, CatBoostClassifier
import pandas as pd
import joblib
import os
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin

# DATA_PATH = 'src/recommendation/binary_classification_rand_reg/data/demog_ranking_grouped.csv'
# MODEL_DIR = 'src/recommendation/catboost/models_grouped'
# METRICS_DIR = 'src/recommendation/catboost/metrics_grouped'
DATA_PATH = 'src/recommendation/binary_classification_rand_reg/data/demog_ranking_grouped_catbased.csv'
MODEL_DIR = 'src/recommendation/catboost/models_grouped_catbased'
METRICS_DIR = 'src/recommendation/catboost/metrics_grouped_catbased'

RANDOM_STATE = 42
TEST_SIZE = 0.2

categories = ['charity', 'loan', 'utility', 'investment', 'finance', 'shopping',
                 'personal_care', 'medical', 'home_and_living', 'insurance', 'automotive',
                 'restaurant', 'business', 'entertainment', 'bank', 'education', 'pet_care',
                 'government', 'travel', 'transportation', 'visit', 'system_dpst', 'other', 
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

def load_and_preprocess_data_binary_class(DATA_PATH):
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
    
    return df, preprocessor, categories

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


def train_and_evaluate_models():
    """Train and evaluate CatBoost models for each category using transaction counts"""
    df, preprocessor, categories = load_and_preprocess_data(DATA_PATH)
    
    # Get feature columns (exclude target categories)
    feature_cols = [col for col in df.columns if col not in categories]
    X_df = df[feature_cols]
    
    X = preprocessor.fit_transform(X_df)
    results = {}
    
    for category in categories:
        if category not in df.columns:
            print(f"Warning: Category '{category}' not found in data. Skipping...")
            continue
            
        print(f"\nTraining CatBoost model for {category}...")
        y = df[category]
        
        # Skip if all zeros (no transactions)
        if y.sum() == 0:
            print(f"Warning: Category '{category}' has no transactions. Skipping...")
            continue
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        
        # Use CatBoostRegressor
        model = CatBoostRegressor(
            iterations=1000,
            learning_rate=0.1,
            depth=6,
            random_seed=RANDOM_STATE,
            verbose=100,  # Shows progress every 100 iterations
            task_type='CPU'
        )
        
        model.fit(
            X_train, y_train,
            eval_set=(X_test, y_test),
            early_stopping_rounds=50
        )
        
        # Evaluate
        y_pred = model.predict(X_test)
        
        # Regression metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Print metrics
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R²: {r2:.4f}")
        
        results[category] = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
        
        # Save model with metadata
        model_data = {
            'model': model,
            'category': category,
            'preprocessor': preprocessor,
            'is_regression': True
        }
        
        joblib.dump(model_data, f"{MODEL_DIR}/{category}_model.pkl")
    
    # Save preprocessor and metrics
    joblib.dump(preprocessor, f"{MODEL_DIR}/preprocessor.pkl")
    
    with open(f"{METRICS_DIR}/training_metrics.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nTraining completed. Models and metrics saved.")
    
    if results:
        print("\nTraining Metrics Summary:")
        results_df = pd.DataFrame(results).T
        print(results_df.round(4))
        
        print(f"\nAverage Metrics:")
        print(f"RMSE: {results_df['rmse'].mean():.4f}")
        print(f"MAE: {results_df['mae'].mean():.4f}")
        print(f"R²: {results_df['r2'].mean():.4f}")

if __name__ == "__main__":
    train_and_evaluate_models()