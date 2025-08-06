# step1_alternative: prepare_rl_dataset.py
import pandas as pd
import numpy as np
import joblib
import os
import pickle
from d3rlpy.dataset import MDPDataset
from src.recommendation.binary_classification.T0.train_model import load_and_preprocess_data

# DATA_PATH = 'src/recommendation/binary_classification/data/demog_labels.csv'
DATA_PATH = 'src/recommendation/binary_classification_rand_reg/data/demog_ranking.csv'
DATA_DIR = 'src/recommendation/offline_rl/T0/ranking_based/data'

def create_offline_rl_dataset_alternative():
    """Create offline RL dataset using transaction counts as rewards"""
    
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Load and preprocess data
    df, preprocessor, categories = load_and_preprocess_data(DATA_PATH)
    
    # Save preprocessor and categories
    joblib.dump(preprocessor, f'{DATA_DIR}/preprocessor.pkl')
    joblib.dump(categories, f'{DATA_DIR}/categories.pkl')
    
    # Get features and transaction counts
    feature_cols = [col for col in df.columns if col not in categories]
    X = preprocessor.fit_transform(df[feature_cols])
    transaction_counts = df[categories].values  # Actual transaction counts
    
    print(f"Dataset shape: {df.shape}")
    print(f"Features shape: {X.shape}")
    print(f"Categories: {categories}")
    
    # Prepare RL dataset components
    observations = []
    actions = []
    rewards = []
    terminals = []
    
    # Create dataset with transaction counts as rewards
    for i in range(len(df)):
        customer_features = X[i]
        customer_transactions = transaction_counts[i]
        
        # Add all possible actions with their rewards
        for action_idx in range(len(categories)):
            observations.append(customer_features.copy())
            actions.append(action_idx)
            rewards.append(float(customer_transactions[action_idx]))  # Use actual count as reward
            terminals.append(True)
    
    # Convert to numpy arrays
    observations = np.array(observations, dtype=np.float32)
    actions = np.array(actions, dtype=np.int64)
    rewards = np.array(rewards, dtype=np.float32)
    terminals = np.array(terminals, dtype=bool)
    
    print(f"\nDataset Statistics:")
    print(f"Total transitions: {len(observations)}")
    print(f"Average reward: {np.mean(rewards):.2f}")
    print(f"Max reward: {np.max(rewards):.2f}")
    print(f"Min reward: {np.min(rewards):.2f}")
    
    # Save dataset components
    np.save(f'{DATA_DIR}/observations.npy', observations)
    np.save(f'{DATA_DIR}/actions.npy', actions)
    np.save(f'{DATA_DIR}/rewards.npy', rewards)
    np.save(f'{DATA_DIR}/terminals.npy', terminals)
    
    # Also save as MDPDataset
    try:
        dataset = MDPDataset(
            observations=observations,
            actions=actions,
            rewards=rewards,
            terminals=terminals,
        )
        dataset.dump(f'{DATA_DIR}/mdp_dataset.h5')
        print("✅ Dataset saved in multiple formats")
        return dataset
    except Exception as e:
        print(f"Warning: Could not create MDPDataset: {e}")
        return None
    
if __name__ == "__main__":
    create_offline_rl_dataset_alternative()