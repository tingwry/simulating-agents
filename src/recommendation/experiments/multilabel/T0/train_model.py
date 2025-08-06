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

# DATA_PATH = 'src/recommendation/binary_classification_rand_reg/data/demog_ranking_grouped_catbased.csv'
DATA_PATH = 'src/recommendation/multilabel/data/demog_grouped_catbased.csv'
MODEL_DIR = 'src/recommendation/multilabel/T0/models_grouped_catbased'
METRICS_DIR = 'src/recommendation/multilabel/T0/metrics_grouped_catbased'

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

# Define a custom Dataset class
class TransactionDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return {
            'features': torch.tensor(self.features[idx], dtype=torch.float32),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float32)
        }

# Define the neural network model
class MultiLabelClassifier(nn.Module):
    def __init__(self, input_size, output_size):
        super(MultiLabelClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            
            nn.Linear(64, output_size),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.network(x)

def train_and_evaluate_pytorch_model():
    """Train and evaluate PyTorch model for multi-label classification"""
    # Load and preprocess data
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
    
    # Create datasets and dataloaders
    train_dataset = TransactionDataset(X_train, y_train)
    test_dataset = TransactionDataset(X_test, y_test)
    
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize model
    input_size = X_train.shape[1]
    output_size = len(categories)
    model = MultiLabelClassifier(input_size, output_size)
    
    # Define loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 50
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for batch in train_loader:
            features = batch['features']
            labels = batch['labels']
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                features = batch['features']
                labels = batch['labels']
                outputs = model(features)
                val_loss += criterion(outputs, labels).item()
        
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(test_loader)
        
        print(f'Epoch {epoch+1}/{num_epochs}, '
              f'Train Loss: {avg_train_loss:.4f}, '
              f'Val Loss: {avg_val_loss:.4f}')
        
        # Save best model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            # Save model weights only (safest option)
            torch.save(model.state_dict(), f"{MODEL_DIR}/best_model_weights.pth")
            
            # Save metadata separately
            joblib.dump({
                'epoch': epoch,
                'categories': categories,
                'input_size': input_size,
                'output_size': output_size,
                'scaler_mean': scaler.mean_,
                'scaler_scale': scaler.scale_,
            }, f"{MODEL_DIR}/model_metadata.pkl")
            
            # Save preprocessor and scaler
            joblib.dump(preprocessor, f"{MODEL_DIR}/preprocessor.pkl")
            joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
    
    
    # Load best model - modified version
    checkpoint = torch.load(f"{MODEL_DIR}/best_model.pth", weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load preprocessor and scaler
    preprocessor = joblib.load(f"{MODEL_DIR}/preprocessor.pkl")
    scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")



    
    # Calculate metrics
    y_pred = []
    y_true = []
    with torch.no_grad():
        for batch in test_loader:
            features = batch['features']
            labels = batch['labels']
            outputs = model(features)
            y_pred.extend(outputs.numpy())
            y_true.extend(labels.numpy())
    
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    
    # Threshold predictions (you can adjust this threshold)
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    # Calculate metrics for each category
    results = {}
    for i, category in enumerate(categories):
        tp = np.sum((y_pred_binary[:, i] == 1) & (y_true[:, i] == 1))
        fp = np.sum((y_pred_binary[:, i] == 1) & (y_true[:, i] == 0))
        fn = np.sum((y_pred_binary[:, i] == 0) & (y_true[:, i] == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        results[category] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
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
    train_and_evaluate_pytorch_model()