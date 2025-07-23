# import numpy as np
# from d3rlpy.dataset import ReplayBuffer, InfiniteBuffer
# from d3rlpy.algos import DQNConfig  # Needed to pass config when loading buffer

# # load from HDF5
# with open("src/recommendation/offline_rl/T0/data/offline_rl_dataset.h5", "rb") as f:
#     new_dataset = ReplayBuffer.load(f, InfiniteBuffer())

# import h5py
# f = h5py.File("src/recommendation/offline_rl/T0/data/offline_rl_dataset.h5", 'r')
# print(list(f.keys()))
# keys = list(f.keys())

# for key in keys[50:55]:
#         print(f"\nDataset '{key}':")
#         print(f"  shape: {f[key].shape}")
#         print(f"  dtype: {f[key].dtype}")
#         # Optionally peek at first few entries
#         print(f"  first 5 entries:\n{f[key][:5]}")





# # step2: train_offline_rl
# # from d3rlpy.dataset import MDPDataset
# from d3rlpy.algos import DQNConfig, CQL, CQLConfig
# from d3rlpy.dataset import ReplayBuffer, InfiniteBuffer
# # from d3rlpy.metrics.scorer import evaluate_on_environment
# from d3rlpy.models.encoders import DefaultEncoderFactory
# # from d3rlpy.preprocessing import DiscreteActionScaler
# from d3rlpy.metrics import TDErrorEvaluator

# import os

# dataset_path = 'src/recommendation/offline_rl/T0/data/offline_rl_dataset.h5'
# model_dir = 'src/recommendation/offline_rl/T0/models'
# os.makedirs(model_dir, exist_ok=True)

# def train_cql():
#     with open(dataset_path, "rb") as f:
#         dataset = ReplayBuffer.load(f, InfiniteBuffer())
    
#     # cql = CQL(use_gpu=False, action_scaler=DiscreteActionScaler())
#     # cql = CQL().create(device="cuda:0")

#     # cql = CQLConfig().create(device=None)
#     # cql.build_with_dataset(dataset)
#     dqn = DQNConfig().create(device=None)
#     dqn.build_with_dataset(dataset)

#     # calculate metrics with training dataset
#     td_error_evaluator = TDErrorEvaluator(episodes=dataset.episodes)

#     # Training
#     # cql.fit(
#     #     dataset,
#     #     # n_epochs=50,
#     #     n_steps=10000,
#     #     # scorers={
#     #     #     'average_reward': lambda algo, dataset: dataset.rewards.mean()
#     #     # }
#     #     evaluators={
#     #         'td_error': td_error_evaluator,
#     #     }
#     # )
#     dqn.fit(
#         dataset,
#         n_steps=10000,
#         evaluators={
#             # 'average_reward': lambda algo, dataset: dataset.rewards.mean()
#             'td_error': td_error_evaluator,
#         }
#     )

#     # Save model
#     # cql.save_model(f'{model_dir}/cql_model.d3')
#     # print(f"✅ Trained CQL model saved to {model_dir}/cql_model.d3")
#     dqn.save_model(f'{model_dir}/dqn_model.d3')
#     print(f"✅ Trained dqn model saved to {model_dir}/dqn_model.d3")

# if __name__ == "__main__":
#     train_cql()


# # step2: train_offline_rl.py
# import os
# from d3rlpy.dataset import MDPDataset
# from d3rlpy.algos import DQNConfig, CQLConfig
# from d3rlpy.metrics import TDErrorEvaluator, AverageValueEstimationEvaluator

# dataset_path = 'src/recommendation/offline_rl/T0/data/offline_rl_dataset.h5'
# model_dir = 'src/recommendation/offline_rl/T0/models'

# def train_offline_rl():
#     """Train offline RL model for transaction category prediction."""
    
#     os.makedirs(model_dir, exist_ok=True)
    
#     # Load dataset
#     dataset = MDPDataset.load(dataset_path)
#     print(f"Loaded dataset with {len(dataset)} transitions")
    
#     # Split dataset for training and evaluation
#     train_episodes, test_episodes = dataset.split(test_size=0.2)
    
#     # Initialize algorithm - CQL is better for offline RL
#     cql = CQLConfig(
#         actor_learning_rate=3e-4,
#         critic_learning_rate=3e-4,
#         batch_size=256,
#         target_update_interval=8000,
#     ).create(device=None)  # Use CPU, change to "cuda:0" if GPU available
    
#     # Build with dataset to get action space info
#     cql.build_with_dataset(dataset)
    
#     # Setup evaluators
#     td_error_evaluator = TDErrorEvaluator(episodes=test_episodes)
#     value_estimator_evaluator = AverageValueEstimationEvaluator(episodes=test_episodes)
    
#     # Training
#     print("Starting training...")
#     cql.fit(
#         train_episodes,
#         n_steps=50000,
#         evaluators={
#             'td_error': td_error_evaluator,
#             'average_value': value_estimator_evaluator,
#         },
#         save_interval=10000,
#         experiment_name="transaction_category_prediction"
#     )
    
#     # Save final model
#     model_path = f'{model_dir}/cql_model.d3'
#     cql.save_model(model_path)
#     print(f"✅ Trained CQL model saved to {model_path}")
    
#     return cql

# def train_dqn_alternative():
#     """Alternative: Train DQN model."""
    
#     os.makedirs(model_dir, exist_ok=True)
    
#     # Load dataset
#     dataset = MDPDataset.load(dataset_path)
    
#     # Split dataset
#     train_episodes, test_episodes = dataset.split(test_size=0.2)
    
#     # Initialize DQN
#     dqn = DQNConfig(
#         learning_rate=1e-3,
#         batch_size=256,
#         target_update_interval=1000,
#     ).create(device=None)
    
#     dqn.build_with_dataset(dataset)
    
#     # Setup evaluators
#     td_error_evaluator = TDErrorEvaluator(episodes=test_episodes)
    
#     # Training
#     print("Starting DQN training...")
#     dqn.fit(
#         train_episodes,
#         n_steps=30000,
#         evaluators={
#             'td_error': td_error_evaluator,
#         }
#     )
    
#     # Save model
#     model_path = f'{model_dir}/dqn_model.d3'
#     dqn.save_model(model_path)
#     print(f"✅ Trained DQN model saved to {model_path}")
    
#     return dqn

# if __name__ == "__main__":
#     # Use CQL for offline RL (recommended)
#     train_offline_rl()
    
#     # Or use DQN (alternative)
#     # train_dqn_alternative()



# step2_alternative: train_offline_rl.py
import os
import numpy as np
import pickle
from d3rlpy.dataset import MDPDataset
from d3rlpy.algos import DQNConfig, DiscreteCQLConfig
from d3rlpy.metrics import TDErrorEvaluator

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
    """Train offline RL model with alternative dataset loading."""
    
    data_dir = 'src/recommendation/offline_rl/T0/data'
    model_dir = 'src/recommendation/offline_rl/T0/models'
    os.makedirs(model_dir, exist_ok=True)
    
    # Load dataset
    dataset = load_dataset_components(data_dir)
    # print(f"Loaded dataset with {len(dataset)} transitions")
    
    # Split dataset
    # train_episodes, test_episodes = dataset.split(test_size=0.2)
    
    # Initialize CQL
    cql = DiscreteCQLConfig(
        # actor_learning_rate=3e-4,
        # critic_learning_rate=3e-4,
        # batch_size=256,
        # target_update_interval=8000,
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