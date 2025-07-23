import pandas as pd
import joblib
import os
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin

DATA_PATH = 'src/recommendation/binary_classification/data/demog_labels.csv'
MODEL_DIR = 'src/recommendation/binary_classification/T0/models'
METRICS_DIR = 'src/recommendation/binary_classification/T0/metrics'
RANDOM_STATE = 42
TEST_SIZE = 0.2

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
    categories = ['charity', 'loan', 'utility', 'investment', 'finance', 'shopping',
                 'personal_care', 'medical', 'home_and_living', 'insurance', 'automotive',
                 'restaurant', 'business', 'entertainment', 'bank', 'education', 'pet_care',
                 'government', 'travel', 'transportation', 'visit', 'system_dpst']
    
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

def train_and_evaluate_models():
    """Train and evaluate models for each category"""
    df, preprocessor, categories = load_and_preprocess_data(DATA_PATH)
    
    # Get feature columns (exclude target categories)
    feature_cols = [col for col in df.columns if col not in categories]
    X_df = df[feature_cols]
    
    # Fit preprocessor on feature columns only
    X = preprocessor.fit_transform(X_df)
    results = {}
    
    for category in categories:
        if category not in df.columns:
            print(f"Warning: Category '{category}' not found in data. Skipping...")
            continue
            
        print(f"\nTraining model for {category}...")
        y = df[category]
        
        # Check if we have both classes
        unique_classes = y.unique()
        if len(unique_classes) < 2:
            print(f"Warning: Category '{category}' has only one class. Skipping...")
            continue
        
        # Check class distribution for stratification
        class_counts = y.value_counts()
        min_class_count = class_counts.min()
        
        # If any class has fewer than 2 samples, we can't use stratification
        if min_class_count < 2:
            print(f"Warning: Category '{category}' has classes with too few samples (min: {min_class_count}). Using random split...")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        else:
            # Safe to use stratification
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
        
        clf = RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight='balanced'
        )
        clf.fit(X_train, y_train)
        
        # Evaluate with better handling of warnings
        y_pred = clf.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        # Print class distribution for debugging
        print(f"  Class distribution - Training: {y_train.value_counts().to_dict()}")
        print(f"  Class distribution - Test: {y_test.value_counts().to_dict()}")
        print(f"  Accuracy: {report['accuracy']:.4f}")
        
        results[category] = {
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1': report['weighted avg']['f1-score'],
            'accuracy': report['accuracy'],
            'support': report['weighted avg']['support']
        }
        
        # Save model with metadata
        model_data = {
            'model': clf,
            'category': category,
            'preprocessor': preprocessor  # Include preprocessor with each model
        }
        
        joblib.dump(model_data, f"{MODEL_DIR}/{category}_model.pkl")
    
    # Save preprocessor separately as well
    joblib.dump(preprocessor, f"{MODEL_DIR}/preprocessor.pkl")
    
    # Save metrics
    with open(f"{METRICS_DIR}/training_metrics.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nTraining completed. Models and metrics saved.")
    
    if results:
        print("\nTraining Metrics Summary:")
        results_df = pd.DataFrame(results).T
        print(results_df.round(4))
        
        # Print average metrics
        print(f"\nAverage Metrics:")
        print(f"Precision: {results_df['precision'].mean():.4f}")
        print(f"Recall: {results_df['recall'].mean():.4f}")
        print(f"F1-Score: {results_df['f1'].mean():.4f}")
        print(f"Accuracy: {results_df['accuracy'].mean():.4f}")
    else:
        print("No models were trained successfully.")

if __name__ == "__main__":
    train_and_evaluate_models()