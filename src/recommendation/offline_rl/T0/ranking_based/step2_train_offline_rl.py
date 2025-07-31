# step2_alternative: train_offline_rl.py
import os
import numpy as np
import pickle
from d3rlpy.dataset import MDPDataset
from d3rlpy.algos import DQNConfig, DiscreteCQLConfig
from d3rlpy.metrics import TDErrorEvaluator
from d3rlpy.preprocessing import MinMaxRewardScaler

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

def train_offline_rl_alternative():
    """Train offline RL model with transaction count rewards"""
    
    data_dir = 'src/recommendation/offline_rl/T0/ranking_based/data'
    model_dir = 'src/recommendation/offline_rl/T0/ranking_based/models'
    os.makedirs(model_dir, exist_ok=True)
    
    # Load dataset
    try:
        dataset = MDPDataset.load(f'{data_dir}/mdp_dataset.h5')
    except:
        dataset = load_dataset_components(data_dir)
    
    reward_scaler = MinMaxRewardScaler(minimum=0.0, maximum=1.0)

    # Initialize CQL with continuous reward support
    cql = DiscreteCQLConfig(
        learning_rate=6.25e-05,
        batch_size=32,
        # n_critics=2,
        # alpha=1.5,
        reward_scaler=reward_scaler  # Important for continuous rewards
    ).create(device=None)
    
    cql.build_with_dataset(dataset)
    
    # Training
    print("Starting CQL training with transaction count rewards...")
    cql.fit(
        dataset,
        n_steps=200000,
        save_interval=10000,
    )
    
    # Save model
    model_path = f'{model_dir}/cql_model_txn_counts.d3'
    cql.save_model(model_path)
    print(f"✅ Model saved to {model_path}")
    
    return cql

if __name__ == "__main__":
    # Use CQL for offline RL (recommended)
    train_offline_rl_alternative()
    
    # Or use DQN (alternative)
    # train_dqn_alternative()