import pandas as pd
import numpy as np
import joblib
from d3rlpy.dataset import MDPDataset
from sklearn.preprocessing import LabelEncoder
from src.recommendation.binary_classification.T0.train_model import preprocess_unknown_values, load_and_preprocess_data

DATA_PATH = 'src/recommendation/binary_classification/data/demog_labels.csv'
MODEL_DIR = 'src/recommendation/binary_classification/T0/models'

def create_offline_rl_dataset(save_path='src/recommendation/offline_rl/T0/data/offline_rl_dataset.h5'):
    # Load and preprocess data
    df, preprocessor, categories = load_and_preprocess_data(DATA_PATH)
    
    # Get features (excluding target categories)
    feature_cols = [col for col in df.columns if col not in categories]
    X = preprocessor.fit_transform(df[feature_cols])

    print(X)
    
    # Prepare RL dataset components
    observations = []
    actions = []
    rewards = []
    terminals = []


    # Create one transition per customer-category pair
    for i in range(len(df)):
        state = X[i].reshape(1, -1)  # Reshape to (1, n_features)
        
        for action_idx, category in enumerate(categories):
            reward = float(df.iloc[i][category])
            # is_terminal = (action_idx == len(categories) - 1)  # last category
            is_terminal = True

            print('state: ', state)
            print('action: ', action_idx)
            print('reward: ', reward)
            print('is_terminal: ', is_terminal)

            observations.append(state)
            actions.append(action_idx)
            rewards.append(reward)
            terminals.append(is_terminal)


    # Convert to numpy arrays with proper shapes
    observations = np.vstack(observations)  # Shape: (n_transitions, n_features)
    actions = np.array(actions)  # Shape: (n_transitions,)
    rewards = np.array(rewards)  # Shape: (n_transitions,)
    terminals = np.array(terminals)  # Shape: (n_transitions,)

    # Create dataset
    dataset = MDPDataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
    )
    
    # Save dataset
    dataset.dump(save_path)
    print(f"✅ Offline RL dataset saved to {save_path}")
    print(f"📦 Dataset contains {len(observations)} transitions")

if __name__ == "__main__":
    create_offline_rl_dataset()
