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






# from d3rlpy.dataset import MDPDataset
from d3rlpy.algos import DQNConfig, CQL, CQLConfig
from d3rlpy.dataset import ReplayBuffer, InfiniteBuffer
# from d3rlpy.metrics.scorer import evaluate_on_environment
from d3rlpy.models.encoders import DefaultEncoderFactory
# from d3rlpy.preprocessing import DiscreteActionScaler
from d3rlpy.metrics import TDErrorEvaluator

import os

dataset_path = 'src/recommendation/offline_rl/T0/data/offline_rl_dataset.h5'
model_dir = 'src/recommendation/offline_rl/T0/models'
os.makedirs(model_dir, exist_ok=True)

def train_cql():
    with open(dataset_path, "rb") as f:
        dataset = ReplayBuffer.load(f, InfiniteBuffer())
    
    # cql = CQL(use_gpu=False, action_scaler=DiscreteActionScaler())
    # cql = CQL().create(device="cuda:0")
    cql = CQLConfig().create(device=None)
    cql.build_with_dataset(dataset)
    # dqn = DQNConfig().create(device=None)
    # dqn.build_with_dataset(dataset)

    # calculate metrics with training dataset
    td_error_evaluator = TDErrorEvaluator(episodes=dataset.episodes)

    # Training
    cql.fit(
        dataset,
        # n_epochs=50,
        n_steps=10000,
        # scorers={
        #     'average_reward': lambda algo, dataset: dataset.rewards.mean()
        # }
        evaluators={
            'td_error': td_error_evaluator,
        }
    )
    # dqn.fit(
    #     dataset,
    #     n_steps=10000,
    #     evaluators={
    #         # 'average_reward': lambda algo, dataset: dataset.rewards.mean()
    #         'td_error': td_error_evaluator,
    #     }
    # )

    # Save model
    cql.save_model(f'{model_dir}/cql_model.d3')
    print(f"✅ Trained CQL model saved to {model_dir}/cql_model.d3")
    # dqn.save_model(f'{model_dir}/dqn_model.d3')
    # print(f"✅ Trained dqn model saved to {model_dir}/dqn_model.d3")

if __name__ == "__main__":
    train_cql()
