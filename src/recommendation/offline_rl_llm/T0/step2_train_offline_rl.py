# step2_alternative: train_offline_rl.py
import os
import numpy as np
import pickle
from d3rlpy.dataset import MDPDataset
from d3rlpy.algos import DQNConfig, DiscreteCQLConfig
from d3rlpy.metrics import TDErrorEvaluator
from sklearn.preprocessing import StandardScaler

def load_dataset_components(data_dir):
    """Load dataset using different methods."""
    
    # Method 1: Load from numpy arrays
    try:
        observations = np.load(f'{data_dir}/observations.npy')
        actions = np.load(f'{data_dir}/actions.npy')
        rewards = np.load(f'{data_dir}/rewards.npy')
        terminals = np.load(f'{data_dir}/terminals.npy')

        scaler = StandardScaler()
        observations = scaler.fit_transform(observations)
        
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
    """Train offline RL model with alternative dataset loading."""
    
    data_dir = 'src/recommendation/offline_rl_llm/T0/data'
    model_dir = 'src/recommendation/offline_rl_llm/T0/models'
    os.makedirs(model_dir, exist_ok=True)
    
    # Load dataset
    dataset = load_dataset_components(data_dir)
    # print(f"Loaded dataset with {len(dataset)} transitions")
    
    # Split dataset
    # train_episodes, test_episodes = dataset.split(test_size=0.2)
    
    # Initialize CQL
    cql = DiscreteCQLConfig(
        learning_rate=6.25e-05,
        batch_size=32
    ).create(device=None)
    
    cql.build_with_dataset(dataset)
    
    # Setup evaluator
    td_error_evaluator = TDErrorEvaluator(episodes=dataset.episodes)
    
    # Training
    print("Starting CQL training...")
    cql.fit(
        dataset,
        n_steps=100000,
        evaluators={'td_error': td_error_evaluator},
        save_interval=10000,
    )
    
    # Save model
    model_path = f'{model_dir}/cql_model.d3'
    cql.save_model(model_path)
    print(f"✅ Model saved to {model_path}")
    
    return cql

if __name__ == "__main__":
    # Use CQL for offline RL (recommended)
    train_offline_rl_alternative()
    
    # Or use DQN (alternative)
    # train_dqn_alternative()