# # step1: prepare_rl_dataset
# import pandas as pd
# import numpy as np
# import joblib
# from d3rlpy.dataset import MDPDataset
# from sklearn.preprocessing import LabelEncoder
# from src.recommendation.binary_classification.T0.train_model import preprocess_unknown_values, load_and_preprocess_data

# DATA_PATH = 'src/recommendation/binary_classification/data/demog_labels.csv'
# MODEL_DIR = 'src/recommendation/binary_classification/T0/models'

# def create_offline_rl_dataset(save_path='src/recommendation/offline_rl/T0/data/offline_rl_dataset.h5'):
#     # Load and preprocess data
#     df, preprocessor, categories = load_and_preprocess_data(DATA_PATH)
    
#     # Get features (excluding target categories)
#     feature_cols = [col for col in df.columns if col not in categories]
#     X = preprocessor.fit_transform(df[feature_cols])

#     print(X)
    
#     # Prepare RL dataset components
#     observations = []
#     actions = []
#     rewards = []
#     terminals = []


#     # Create one transition per customer-category pair
#     for i in range(len(df)):
#         state = X[i].reshape(1, -1)  # Reshape to (1, n_features)
        
#         for action_idx, category in enumerate(categories):
#             reward = float(df.iloc[i][category])
#             # is_terminal = (action_idx == len(categories) - 1)  # last category
#             is_terminal = True

#             print('state: ', state)
#             print('action: ', action_idx)
#             print('reward: ', reward)
#             print('is_terminal: ', is_terminal)

#             observations.append(state)
#             actions.append(action_idx)
#             rewards.append(reward)
#             terminals.append(is_terminal)


#     # Convert to numpy arrays with proper shapes
#     observations = np.vstack(observations)  # Shape: (n_transitions, n_features)
#     actions = np.array(actions)  # Shape: (n_transitions,)
#     rewards = np.array(rewards)  # Shape: (n_transitions,)
#     terminals = np.array(terminals)  # Shape: (n_transitions,)

#     # Create dataset
#     dataset = MDPDataset(
#         observations=observations,
#         actions=actions,
#         rewards=rewards,
#         terminals=terminals,
#     )
    
#     # Save dataset
#     dataset.dump(save_path)
#     print(f"✅ Offline RL dataset saved to {save_path}")
#     print(f"📦 Dataset contains {len(observations)} transitions")

# if __name__ == "__main__":
#     create_offline_rl_dataset()


# # step1: prepare_rl_dataset.py
# import pandas as pd
# import numpy as np
# import joblib
# import os
# from d3rlpy.dataset import MDPDataset
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from src.recommendation.binary_classification.T0.train_model import preprocess_unknown_values, load_and_preprocess_data

# DATA_PATH = 'src/recommendation/binary_classification/data/demog_labels.csv'
# DATA_DIR = 'src/recommendation/offline_rl/T0/data'

# def create_offline_rl_dataset(save_path='src/recommendation/offline_rl/T0/data/offline_rl_dataset.h5'):
#     """Create offline RL dataset for transaction category prediction."""
    
#     # Create directories
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)

#     # Load and preprocess data
#     df, preprocessor, categories = load_and_preprocess_data(DATA_PATH)
    
#     # Save preprocessor and categories for later use
#     joblib.dump(preprocessor, f'{DATA_DIR}/preprocessor.pkl')
#     joblib.dump(categories, f'{DATA_DIR}/categories.pkl')
    
#     # Get features (excluding target categories)
#     feature_cols = [col for col in df.columns if col not in categories]
#     X = preprocessor.fit_transform(df[feature_cols])
    
#     print(f"Dataset shape: {df.shape}")
#     print(f"Features shape: {X.shape}")
#     print(f"Categories: {categories}")
#     print(f"Number of categories (action space): {len(categories)}")
    
#     # Prepare RL dataset components
#     observations = []
#     actions = []
#     rewards = []
#     terminals = []
    
#     # Strategy: Create balanced dataset with positive and negative examples
#     for i in range(len(df)):
#         customer_features = X[i]  # Already a 1D array
#         customer_labels = df.iloc[i][categories].values
        
#         # Get positive and negative categories for this customer
#         positive_categories = np.where(customer_labels == 1)[0]
#         negative_categories = np.where(customer_labels == 0)[0]
        
#         # Add all positive examples
#         for action_idx in positive_categories:
#             observations.append(customer_features.copy())
#             actions.append(action_idx)
#             rewards.append(1.0)  # Positive reward for correct prediction
#             terminals.append(True)  # Each recommendation is independent
        
#         # Add balanced number of negative examples
#         n_negative_samples = len(positive_categories)  # Balance positive/negative
#         if n_negative_samples > 0 and len(negative_categories) > 0:
#             # Randomly sample negative examples
#             sampled_negative = np.random.choice(
#                 negative_categories, 
#                 min(n_negative_samples, len(negative_categories)), 
#                 replace=False
#             )
#             for action_idx in sampled_negative:
#                 observations.append(customer_features.copy())
#                 actions.append(action_idx)
#                 rewards.append(0.0)  # No reward for incorrect prediction
#                 terminals.append(True)
    
#     # Convert to numpy arrays with proper shapes
#     observations = np.array(observations, dtype=np.float32)  # Shape: (n_transitions, n_features)
#     actions = np.array(actions, dtype=np.int64)  # Shape: (n_transitions,)
#     rewards = np.array(rewards, dtype=np.float32)  # Shape: (n_transitions,)
#     terminals = np.array(terminals, dtype=bool)  # Shape: (n_transitions,)
    
#     print(f"\nDataset Statistics:")
#     print(f"Total transitions: {len(observations)}")
#     print(f"Observations shape: {observations.shape}")
#     print(f"Actions shape: {actions.shape}")
#     print(f"Positive samples: {np.sum(rewards == 1.0)}")
#     print(f"Negative samples: {np.sum(rewards == 0.0)}")
#     print(f"Action distribution:")
#     unique_actions, counts = np.unique(actions, return_counts=True)
#     for action, count in zip(unique_actions, counts):
#         category_name = categories[action] if action < len(categories) else f"Action_{action}"
#         print(f"  {category_name}: {count}")
    
#     # Create dataset
#     dataset = MDPDataset(
#         observations=observations,
#         actions=actions,
#         rewards=rewards,
#         terminals=terminals,
#     )
    
#     # Save dataset
#     dataset.dump(save_path)
#     print(f"\n✅ Offline RL dataset saved to {save_path}")
#     print(f"📦 Dataset contains {len(observations)} transitions")
    
#     return dataset

# if __name__ == "__main__":
#     create_offline_rl_dataset()


# step1_alternative: prepare_rl_dataset.py
import pandas as pd
import numpy as np
import joblib
import os
import pickle
from d3rlpy.dataset import MDPDataset
from src.recommendation.binary_classification.T0.train_model import load_and_preprocess_data

DATA_PATH = 'src/recommendation/binary_classification/data/demog_labels.csv'
DATA_DIR = 'src/recommendation/offline_rl/T0/data'

def create_offline_rl_dataset_alternative():
    """Create offline RL dataset with alternative saving method."""
    
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Load and preprocess data
    df, preprocessor, categories = load_and_preprocess_data(DATA_PATH)
    
    # Save preprocessor and categories
    joblib.dump(preprocessor, f'{DATA_DIR}/preprocessor.pkl')
    joblib.dump(categories, f'{DATA_DIR}/categories.pkl')
    
    # Get features
    feature_cols = [col for col in df.columns if col not in categories]
    X = preprocessor.fit_transform(df[feature_cols])
    
    print(f"Dataset shape: {df.shape}")
    print(f"Features shape: {X.shape}")
    print(f"Categories: {categories}")
    
    # Prepare RL dataset components
    observations = []
    actions = []
    rewards = []
    terminals = []
    
    # Create balanced dataset
    for i in range(len(df)):
        customer_features = X[i]
        customer_labels = df.iloc[i][categories].values
        
        positive_categories = np.where(customer_labels == 1)[0]
        negative_categories = np.where(customer_labels == 0)[0]
        
        # Add positive examples
        for action_idx in positive_categories:
            observations.append(customer_features.copy())
            actions.append(action_idx)
            rewards.append(1.0)
            terminals.append(True)
        
        # Add balanced negative examples
        n_negative_samples = len(positive_categories)
        if n_negative_samples > 0 and len(negative_categories) > 0:
            sampled_negative = np.random.choice(
                negative_categories, 
                min(n_negative_samples, len(negative_categories)), 
                replace=False
            )
            for action_idx in sampled_negative:
                observations.append(customer_features.copy())
                actions.append(action_idx)
                rewards.append(0.0)
                terminals.append(True)
    
    # Convert to numpy arrays
    observations = np.array(observations, dtype=np.float32)
    actions = np.array(actions, dtype=np.int64)
    rewards = np.array(rewards, dtype=np.float32)
    terminals = np.array(terminals, dtype=bool)
    
    print(f"\nDataset Statistics:")
    print(f"Total transitions: {len(observations)}")
    print(f"Positive samples: {np.sum(rewards == 1.0)}")
    print(f"Negative samples: {np.sum(rewards == 0.0)}")
    
    # Method 1: Save as separate numpy arrays
    np.save(f'{DATA_DIR}/observations.npy', observations)
    np.save(f'{DATA_DIR}/actions.npy', actions)
    np.save(f'{DATA_DIR}/rewards.npy', rewards)
    np.save(f'{DATA_DIR}/terminals.npy', terminals)
    
    # Method 2: Save as pickle
    dataset_dict = {
        'observations': observations,
        'actions': actions,
        'rewards': rewards,
        'terminals': terminals
    }
    with open(f'{DATA_DIR}/dataset_components.pkl', 'wb') as f:
        pickle.dump(dataset_dict, f)
    
    # Method 3: Create and save MDPDataset (if this works)
    try:
        dataset = MDPDataset(
            observations=observations,
            actions=actions,
            rewards=rewards,
            terminals=terminals,
        )
        
        # Try different saving methods
        with open(f'{DATA_DIR}/mdp_dataset.pkl', 'wb') as f:
            pickle.dump(dataset, f)
        
        print("✅ All dataset formats saved successfully")
        return dataset
        
    except Exception as e:
        print(f"Warning: Could not create MDPDataset: {e}")
        print("✅ Dataset components saved as numpy arrays and pickle")
        return None
    
if __name__ == "__main__":
    create_offline_rl_dataset_alternative()