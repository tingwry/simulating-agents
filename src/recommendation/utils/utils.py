import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import f1_score, precision_recall_curve, precision_score, recall_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
from d3rlpy.dataset import MDPDataset
from d3rlpy.algos import DiscreteCQLConfig
import joblib

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
    
def train_model_path_indicator(method, is_regressor, method_model, threshold=None, data='T0'):
    """Determine the appropriate paths based on modeling approach and method.
    
    Args:
        method: Either "binary" or "multilabel" or "reinforcement_learning"
        is_regressor: Boolean indicating if using regression (True) or classification (False)
        method_model: String specifying the modeling approach 
                     ("random_forests", "catboost", "multioutputclassifier", "neural_network")
        data: Data version ('T0', 'T1', or 'T1_predicted')
    
    Returns:
        tuple: (DATA_PATH, MODEL_DIR, METRICS_DIR)
    """

    base_data_path = 'src/recommendation/data'
    base_model_path = 'src/recommendation/models'
    base_metrics_path = 'src/recommendation/metrics'

    OPTIMAL_THRS = ""
    
    if method == "reinforcement_learning":
        DATA_PATH = f'{base_data_path}/rl'
        MODEL_DIR = f'{base_model_path}/rl'
        METRICS_DIR = f'{base_metrics_path}/rl'
    else:
        if method == "binary":
            # Use non-normalized data for binary regression cases
            if is_regressor:
                DATA_PATH = f'{base_data_path}/{data}/demog_ranking_grouped_catbased_no_norm.csv'
            else:
                DATA_PATH = f'{base_data_path}/{data}/demog_ranking_grouped_catbased.csv'

            model_type = "regressor" if is_regressor else "classifier"
            
            if data in ['T1', 'T1_predicted']:
                MODEL_DIR = f'{base_model_path}/binary_classification/{data}/{method_model}_{model_type}'
                METRICS_DIR = f'{base_metrics_path}/binary_classification/{data}/{method_model}_{model_type}'
            else:  # T0 case
                MODEL_DIR = f'{base_model_path}/binary_classification/{method_model}_{model_type}'
                METRICS_DIR = f'{base_metrics_path}/binary_classification/{method_model}_{model_type}'
            
        elif method == "multilabel":
            DATA_PATH = f'{base_data_path}/{data}/demog_grouped_catbased.csv'

            if data in ['T1', 'T1_predicted']:
                MODEL_DIR = f'{base_model_path}/multilabel/{data}/{method_model}'
                METRICS_DIR = f'{base_metrics_path}/multilabel/{data}/{method_model}'
            else:  # T0 case
                MODEL_DIR = f'{base_model_path}/multilabel/{method_model}'
                METRICS_DIR = f'{base_metrics_path}/multilabel/{method_model}'
            
        else:
            raise ValueError(f"Unknown method: {method}. Must be 'binary', 'multilabel' or 'reinforcement_learning'")
        
    if threshold == None:
        OPTIMAL_THRS = "_optimal_thrs"

    return DATA_PATH, MODEL_DIR, METRICS_DIR, OPTIMAL_THRS


def prediction_path_indicator(method, is_regressor, method_model, threshold=None, data='T0'):
    """Determine the appropriate paths for prediction outputs.
    
    Args:
        method: Either "binary", "multilabel", or "reinforcement_learning"
        is_regressor: Boolean indicating if using regression (True) or classification (False)
        method_model: String specifying the modeling approach 
                     ("random_forests", "catboost", "multioutputclassifier", "neural_network")
        data: Data version ('T0', 'T1', or 'T1_predicted')
    
    Returns:
        tuple: (DATA_DIR, MODEL_DIR, PREDICTION_OUTPUT, TEST_DATA_PATH, OPTIMAL_THRS)
    """
    base_data_path = 'src/recommendation/data'
    base_model_path = 'src/recommendation/models'
    base_prediction_path = 'src/recommendation/predictions'

    OPTIMAL_THRS = ""
    
    # Set TEST_DATA_PATH based on data version
    TEST_DATA_PATH = f'{base_data_path}/{data}/test_with_lifestyle.csv'
    
    if method == "reinforcement_learning":
        DATA_DIR = f'{base_data_path}/rl'
        MODEL_DIR = f'{base_model_path}/rl'
        PREDICTION_OUTPUT = f'{base_prediction_path}/rl'
    else:
        DATA_DIR = f'{base_data_path}/{data}'
        
        if method == "binary":
            model_type = "regressor" if is_regressor else "classifier"
            
            if data in ['T1', 'T1_predicted']:
                MODEL_DIR = f'{base_model_path}/binary_classification/{data}/{method_model}_{model_type}'
                PREDICTION_OUTPUT = f'{base_prediction_path}/binary_classification/{data}/{method_model}_{model_type}'
            else:  # T0 case
                MODEL_DIR = f'{base_model_path}/binary_classification/{method_model}_{model_type}'
                PREDICTION_OUTPUT = f'{base_prediction_path}/binary_classification/{method_model}_{model_type}'
            
        elif method == "multilabel":
            if data in ['T1', 'T1_predicted']:
                MODEL_DIR = f'{base_model_path}/multilabel/{data}/{method_model}'
                PREDICTION_OUTPUT = f'{base_prediction_path}/multilabel/{data}/{method_model}'
            else:  # T0 case
                MODEL_DIR = f'{base_model_path}/multilabel/{method_model}'
                PREDICTION_OUTPUT = f'{base_prediction_path}/multilabel/{method_model}'
            
        else:
            raise ValueError(f"Unknown method: {method}. Must be 'binary', 'multilabel' or 'reinforcement_learning'")
        
    if threshold == None:
        OPTIMAL_THRS = "_optimal_thrs"
    
    PREDICTION_OUTPUT += f'/transaction_predictions{OPTIMAL_THRS}.csv'

    return DATA_DIR, MODEL_DIR, PREDICTION_OUTPUT, TEST_DATA_PATH, OPTIMAL_THRS

def evaluation_path_indicator(method, is_regressor, method_model, threshold=None, data='T0'):
    """Determine the appropriate paths for evaluation outputs.
    
    Args:
        method: Either "binary", "multilabel", or "reinforcement_learning"
        is_regressor: Boolean indicating if using regression (True) or classification (False)
        method_model: String specifying the modeling approach 
                     ("random_forests", "catboost", "multioutputclassifier", "neural_network")
        data: Data version ('T0', 'T1', or 'T1_predicted')
    
    Returns:
        tuple: (PREDICTIONS_DIR, SCORES_DIR, EVAL_RESULTS_DIR, ANS_KEY_DIR, OPTIMAL_THRS)
    """
    base_prediction_path = 'src/recommendation/predictions'
    base_eval_path = 'src/recommendation/evaluation/eval_results'
    base_ans_key_path = 'src/recommendation/data/ans_key'

    OPTIMAL_THRS = ""
    
    # Set answer key path based on method and model type
    if method == "binary" and is_regressor:
        ANS_KEY_DIR = f'{base_ans_key_path}/grouped_catbased_no_norm.csv'
    else:
        ANS_KEY_DIR = f'{base_ans_key_path}/grouped_catbased.csv'
    
    if method == "reinforcement_learning":
        PREDICTIONS_DIR = f'{base_prediction_path}/rl'
        SCORES_DIR = f'{base_prediction_path}/rl'
        EVAL_RESULTS_DIR = f'{base_eval_path}/rl'
    else:
        if method == "binary":
            model_type = "regressor" if is_regressor else "classifier"
            
            if data in ['T1', 'T1_predicted']:
                PREDICTIONS_DIR = f'{base_prediction_path}/binary_classification/{data}/{method_model}_{model_type}'
                SCORES_DIR = f'{base_prediction_path}/binary_classification/{data}/{method_model}_{model_type}'
                EVAL_RESULTS_DIR = f'{base_eval_path}/binary_classification/{data}/{method_model}_{model_type}'
            else:  # T0 case
                PREDICTIONS_DIR = f'{base_prediction_path}/binary_classification/{method_model}_{model_type}'
                SCORES_DIR = f'{base_prediction_path}/binary_classification/{method_model}_{model_type}'
                EVAL_RESULTS_DIR = f'{base_eval_path}/binary_classification/{method_model}_{model_type}'
            
        elif method == "multilabel":
            if data in ['T1', 'T1_predicted']:
                PREDICTIONS_DIR = f'{base_prediction_path}/multilabel/{data}/{method_model}'
                SCORES_DIR = f'{base_prediction_path}/multilabel/{data}/{method_model}'
                EVAL_RESULTS_DIR = f'{base_eval_path}/multilabel/{data}/{method_model}'
            else:  # T0 case
                PREDICTIONS_DIR = f'{base_prediction_path}/multilabel/{method_model}'
                SCORES_DIR = f'{base_prediction_path}/multilabel/{method_model}'
                EVAL_RESULTS_DIR = f'{base_eval_path}/multilabel/{method_model}'
            
        else:
            raise ValueError(f"Unknown method: {method}. Must be 'binary', 'multilabel' or 'reinforcement_learning'")
        
    if threshold == None:
        OPTIMAL_THRS = "_optimal_thrs"

    PREDICTIONS_DIR += f'/transaction_predictions{OPTIMAL_THRS}.csv'
    SCORES_DIR += f'/transaction_predictions{OPTIMAL_THRS}_scores.csv'
    
    return PREDICTIONS_DIR, SCORES_DIR, EVAL_RESULTS_DIR, ANS_KEY_DIR, OPTIMAL_THRS


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

    # Fit the preprocessor to the data
    feature_cols = [col for col in df.columns if col not in categories]
    preprocessor.fit(df[feature_cols])
    
    return df, preprocessor

# def find_optimal_regression_threshold(y_true, y_pred):
#     """Find optimal threshold for converting regression outputs to binary predictions"""
#     # Convert true values to binary (1 if > 0, else 0)
#     y_true_binary = (y_true > 0).astype(int)
    
#     # Try different thresholds to find the one that maximizes F1 score
#     thresholds = np.linspace(0, np.max(y_pred), 100)
#     best_threshold = 0
#     best_f1 = -1
    
#     for threshold in thresholds:
#         y_pred_binary = (y_pred >= threshold).astype(int)
#         f1 = f1_score(y_true_binary, y_pred_binary)
#         if f1 > best_f1:
#             best_f1 = f1
#             best_threshold = threshold
    
#     return best_threshold


def find_optimal_regression_threshold(y_true, y_pred, beta=0.5):
    """Find optimal threshold for converting regression outputs to binary predictions
    using a weighted harmonic mean of precision and recall. (F-beta Score)
    
    Args:
        y_true: True values
        y_pred: Predicted values
        beta: Weight for recall in harmonic mean (higher beta favors recall)
    """
    y_true_binary = (y_true > 0).astype(int)
    thresholds = np.linspace(0, np.max(y_pred), 100)
    best_threshold = 0
    best_score = -1
    
    for threshold in thresholds:
        y_pred_binary = (y_pred >= threshold).astype(int)
        
        precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
        recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
        
        if (precision + recall) > 0:
            score = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)
        else:
            score = 0
            
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    return best_threshold

# def find_optimal_classification_threshold(y_true, y_proba):
#     """Find optimal threshold that maximizes F1 score"""
#     precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
#     # Convert to F1 score
#     f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
#     # Find threshold that gives maximum F1 score
#     optimal_idx = np.argmax(f1_scores)
#     optimal_threshold = thresholds[optimal_idx]
#     return optimal_threshold

def find_optimal_classification_threshold(y_true, y_proba, beta=0.5):
    """Find optimal threshold that maximizes F-beta score
    
    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities
        beta: Weight for recall in harmonic mean (higher beta favors recall)
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    
    # Calculate F-beta scores
    f_beta_scores = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall + 1e-9)
    
    # Find threshold that gives maximum F-beta score
    optimal_idx = np.argmax(f_beta_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    return optimal_threshold

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
    
# RL
def load_dataset_components(data_dir):
    """Load dataset using different methods."""
    
    # Method 1: Load from numpy arrays
    try:
        observations = np.load(f'{data_dir}/observations.npy')
        actions = np.load(f'{data_dir}/actions.npy')
        rewards = np.load(f'{data_dir}/rewards.npy')
        terminals = np.load(f'{data_dir}/terminals.npy')
        
        dataset = MDPDataset(
            observations=observations,
            actions=actions,
            rewards=rewards,
            terminals=terminals,
        )
        print("✅ Dataset loaded from numpy arrays")
        return dataset
        
    except Exception as e:
        print(f"Method 1 failed: {e}")
    
    # Method 2: Load from pickle
    try:
        with open(f'{data_dir}/dataset_components.pkl', 'rb') as f:
            components = pickle.load(f)
        
        dataset = MDPDataset(
            observations=components['observations'],
            actions=components['actions'],
            rewards=components['rewards'],
            terminals=components['terminals'],
        )
        print("✅ Dataset loaded from pickle components")
        return dataset
        
    except Exception as e:
        print(f"Method 2 failed: {e}")
    
    # Method 3: Load MDPDataset directly
    try:
        with open(f'{data_dir}/mdp_dataset.pkl', 'rb') as f:
            dataset = pickle.load(f)
        print("✅ Dataset loaded as MDPDataset pickle")
        return dataset
        
    except Exception as e:
        print(f"Method 3 failed: {e}")
    
    raise Exception("Could not load dataset using any method")

def find_optimal_rl_thresholds(data_dir, model_dir, OPTIMAL_THRS, beta=0.5):
    """Find optimal thresholds for RL model by maximizing F-beta score"""
    
    # Load dataset components
    dataset = load_dataset_components(data_dir)
    categories = joblib.load(f'{data_dir}/categories.pkl')
    preprocessor = joblib.load(f'{data_dir}/preprocessor.pkl')
    
    # Load model
    model_path = f'{model_dir}/cql_model_txn_counts{OPTIMAL_THRS}.d3'
    model = DiscreteCQLConfig().create(device=None)
    model.build_with_dataset(dataset)
    model.load_model(model_path)
    
    # Get observations and true labels
    try:
        # Try to load preprocessed test data
        test_df = pd.read_csv(f'{data_dir}/test_with_lifestyle.csv')
        test_df = preprocess_unknown_values(test_df)
        X_test = preprocessor.transform(test_df.drop(columns=categories))
        y_true = test_df[categories].values
    except:
        # Fallback to using training data if test data not available
        observations = np.array([dataset.episodes[i].observations for i in range(len(dataset.episodes))])
        observations = observations.reshape(-1, observations.shape[-1])
        X_test = observations
        y_true = np.zeros((len(X_test), len(categories)))  # Dummy labels
    
    # Get Q-values for test data
    q_values = np.zeros((len(X_test), len(categories)))
    for action_idx in range(len(categories)):
        actions = np.full(len(X_test), action_idx)
        q_values[:, action_idx] = model.predict_value(X_test, actions)
    
    # Find optimal threshold for each category
    optimal_thresholds = {}
    for i, category in enumerate(categories):
        y_true_binary = (y_true[:, i] > 0).astype(int)
        y_pred_scores = q_values[:, i]
        
        # Find threshold that maximizes F-beta score
        optimal_thresholds[category] = find_optimal_regression_threshold(
            y_true_binary, 
            y_pred_scores,
            beta=beta
        )
    
    return optimal_thresholds