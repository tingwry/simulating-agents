import pandas as pd
import numpy as np
import joblib
import os
from d3rlpy.algos import DQNConfig, DiscreteCQLConfig
from src.recommendation.offline_rl.T0.step2_train_offline_rl import load_dataset_components
from src.recommendation.binary_classification.T0.train_model import preprocess_unknown_values

# def predict_transaction_categories_alternative(model_type='cql', top_k=5):
#     """Predict with transaction count ranking"""
    
#     DATA_DIR = 'src/recommendation/offline_rl/T0/ranking_based/data' 
#     MODEL_DIR = 'src/recommendation/offline_rl/T0/ranking_based/models'
#     OUTPUT_DIR = 'src/recommendation/offline_rl/T0/ranking_based/predictions'
#     TEST_DATA_PATH = 'src/data/T0/test_with_lifestyle.csv'
    
#     os.makedirs(OUTPUT_DIR, exist_ok=True)
    
#     # Load preprocessor and categories
#     preprocessor = joblib.load(f'{DATA_DIR}/preprocessor.pkl')
#     categories = joblib.load(f'{DATA_DIR}/categories.pkl')

#     # Load and preprocess test data
#     df = pd.read_csv(TEST_DATA_PATH)
#     df = preprocess_unknown_values(df)
#     feature_cols = [col for col in df.columns if col not in categories and col in df.columns]
    
#     X = preprocessor.transform(df[feature_cols])
    
#     # Load model
#     model_path = f'{MODEL_DIR}/cql_model_txn_counts.d3'
#     model = DiscreteCQLConfig().create(device=None)
#     model.build_with_dataset(load_dataset_components(DATA_DIR))
#     model.load_model(model_path)
    
#     # Get Q-values for all actions (representing expected transaction counts)
#     q_values = np.zeros((len(X), len(categories)))
#     for action_idx in range(len(categories)):
#         actions = np.full(len(X), action_idx)
#         q_values[:, action_idx] = model.predict_value(X, actions)
    
#     # Create output DataFrames
#     predictions_df = pd.DataFrame()
#     scores_df = pd.DataFrame()
    
#     # Handle customer ID
#     id_col = 'CUST_ID' if 'CUST_ID' in df.columns else 'cust_id'
#     predictions_df['cust_id'] = df[id_col]
#     scores_df['cust_id'] = df[id_col]
    
#     # Store predictions and scores
#     for i, category in enumerate(categories):
#         # Binary prediction: 1 if in top-k expected counts
#         top_k_indices = np.argsort(q_values, axis=1)[:, -top_k:]
#         predictions_df[category] = (top_k_indices == i).any(axis=1).astype(int)
        
#         # Raw Q-values (expected transaction counts)
#         scores_df[category] = q_values[:, i]
    
#     # Save outputs
#     predictions_df.to_csv(f'{OUTPUT_DIR}/transaction_predictions.csv', index=False)
#     scores_df.to_csv(f'{OUTPUT_DIR}/transaction_scores.csv', index=False)
    
#     print(f"✅ Predictions saved with transaction count ranking")
#     return predictions_df, scores_df

# if __name__ == "__main__":
#     predict_transaction_categories_alternative()

def predict_transaction_categories_alternative(model_type='cql', threshold=0.1):
    """Predict with transaction count ranking using a threshold"""
    
    DATA_DIR = 'src/recommendation/offline_rl/T0/ranking_based/data' 
    MODEL_DIR = 'src/recommendation/offline_rl/T0/ranking_based/models'
    OUTPUT_DIR = 'src/recommendation/offline_rl/T0/ranking_based/predictions'
    TEST_DATA_PATH = 'src/data/T0/test_with_lifestyle.csv'
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load preprocessor and categories
    preprocessor = joblib.load(f'{DATA_DIR}/preprocessor.pkl')
    categories = joblib.load(f'{DATA_DIR}/categories.pkl')

    # Load and preprocess test data
    df = pd.read_csv(TEST_DATA_PATH)
    df = preprocess_unknown_values(df)
    feature_cols = [col for col in df.columns if col not in categories and col in df.columns]
    
    X = preprocessor.transform(df[feature_cols])
    
    # Load model
    model_path = f'{MODEL_DIR}/cql_model_txn_counts.d3'
    model = DiscreteCQLConfig().create(device=None)
    model.build_with_dataset(load_dataset_components(DATA_DIR))
    model.load_model(model_path)
    
    # Get Q-values for all actions (representing expected transaction counts)
    q_values = np.zeros((len(X), len(categories)))
    for action_idx in range(len(categories)):
        actions = np.full(len(X), action_idx)
        q_values[:, action_idx] = model.predict_value(X, actions)
    
    # Create output DataFrames
    predictions_df = pd.DataFrame()
    scores_df = pd.DataFrame()
    
    # Handle customer ID
    id_col = 'CUST_ID' if 'CUST_ID' in df.columns else 'cust_id'
    predictions_df['cust_id'] = df[id_col]
    scores_df['cust_id'] = df[id_col]
    
    # Store predictions and scores
    for i, category in enumerate(categories):
        # Binary prediction: 1 if Q-value > threshold
        predictions_df[category] = (q_values[:, i] > threshold).astype(int)
        
        # Raw Q-values (expected transaction counts)
        scores_df[category] = q_values[:, i]
    
    # Save outputs
    predictions_df.to_csv(f'{OUTPUT_DIR}/transaction_predictions.csv', index=False)
    scores_df.to_csv(f'{OUTPUT_DIR}/transaction_scores.csv', index=False)
    
    print(f"✅ Predictions saved with transaction count ranking (threshold={threshold})")
    return predictions_df, scores_df

if __name__ == "__main__":
    predict_transaction_categories_alternative(threshold=0.04)