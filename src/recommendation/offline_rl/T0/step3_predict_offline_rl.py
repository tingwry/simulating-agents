# # step3: predict_offline_rl
# import pandas as pd
# import numpy as np
# import joblib
# from d3rlpy.algos import DQNConfig, CQL, CQLConfig
# from d3rlpy.dataset import ReplayBuffer, InfiniteBuffer
# from src.recommendation.binary_classification.T0.train_model import load_and_preprocess_data

# TEST_DATA_PATH = 'src/data/T0/test_with_lifestyle.csv'
# model_path = 'src/recommendation/offline_rl/T0/models/dqn_model.d3'
# output_path = 'src/recommendation/offline_rl/T0/predictions/transaction_predictions.csv'

# dataset_path = 'src/recommendation/offline_rl/T0/data/offline_rl_dataset.h5'

# def predict_top_actions():
#     with open(dataset_path, "rb") as f:
#         dataset = ReplayBuffer.load(f, InfiniteBuffer())


#     df, preprocessor, categories = load_and_preprocess_data(TEST_DATA_PATH)
#     feature_cols = [col for col in df.columns if col not in categories]
#     X_df = df[feature_cols]
#     X = preprocessor.fit_transform(X_df)

#     # cql = CQLConfig().create(device=None)
#     # cql.build_with_dataset(dataset)
#     # cql.load_model(model_path)
#     dqn = DQNConfig().create(device=None)
#     dqn.build_with_dataset(dataset)
#     dqn.load_model(model_path)

#     pred_df = pd.DataFrame()
#     pred_df['cust_id'] = df['cust_id'] if 'cust_id' in df.columns else np.arange(len(df))

#     # Predict Q-values for each customer and choose top categories
#     # q_values = dqn.predict_value(X)
#     actions = dqn.predict(X)
#     # q_values = dqn.predict_value(X, actions)
#     # actions = cql.predict(X)

#     for i, category in enumerate(categories):
#         pred_df[category] = (actions == i).astype(int)

#     # for i, category in enumerate(categories):
#     #     pred_df[category] = (q_values[:, i] > 0).astype(int)

#     pred_df.to_csv(output_path, index=False)
#     print(f"✅ RL Predictions saved to {output_path}")

# if __name__ == "__main__":
#     predict_top_actions()

import pandas as pd
import numpy as np
import joblib
import os
from d3rlpy.algos import DQNConfig, DiscreteCQLConfig
from src.recommendation.offline_rl.T0.step2_train_offline_rl import load_dataset_components
from src.recommendation.binary_classification.T0.train_model import preprocess_unknown_values

# step3_alternative: predict_offline_rl.py
def predict_transaction_categories_alternative(model_type='cql', top_k=5):
    """Predict with alternative dataset loading."""

    DATA_DIR = 'src/recommendation/offline_rl/T0/data' 
    MODEL_DIR = 'src/recommendation/offline_rl/T0/models'
    OUTPUT_DIR = 'src/recommendation/offline_rl/T0/predictions'
    TEST_DATA_PATH = 'src/data/T0/test_with_lifestyle.csv'
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load preprocessor and categories
    preprocessor = joblib.load(f'{DATA_DIR}/preprocessor.pkl')
    categories = joblib.load(f'{DATA_DIR}/categories.pkl')


    
    # Load and preprocess test data
    df = pd.read_csv(TEST_DATA_PATH)
    df = preprocess_unknown_values(df)
    feature_cols = [col for col in df.columns if col not in categories and col in df.columns]
    
    # Handle missing columns
    # for col in preprocessor.feature_names_in_:
    #     if col not in df.columns:
    #         print(f"Warning: Column '{col}' missing, filling with 0")
    #         df[col] = 0
    
    X_df = df[feature_cols]
    X = preprocessor.transform(X_df)
    
    # Load dataset for model building
    dataset = load_dataset_components(DATA_DIR)
    
    # Load model
    if model_type == 'cql':
        model_path = f'{MODEL_DIR}/cql_model.d3'
        model = DiscreteCQLConfig().create(device=None)
    else:
        model_path = f'{MODEL_DIR}/dqn_model.d3'
        model = DQNConfig().create(device=None)
    
    model.build_with_dataset(dataset)
    model.load_model(model_path)
    
    print(f"Loaded model from {model_path}")
    
    # Make predictions
    pred_df = pd.DataFrame()
    pred_df['cust_id'] = df.get('cust_id', np.arange(len(df)))
    
    # Get Q-values for all actions
    q_values_list = []
    for i in range(len(categories)):
        actions = np.full(len(X), i)
        q_vals = model.predict_value(X, actions)
        q_values_list.append(q_vals)
    
    all_q_values = np.column_stack(q_values_list)
    
    # Top-k predictions
    for i, category in enumerate(categories):
        top_k_indices = np.argsort(all_q_values, axis=1)[:, -top_k:]
        pred_df[f'{category}'] = (top_k_indices == i).any(axis=1).astype(int)
    
    # Save predictions
    output_path = f'{OUTPUT_DIR}/transaction_predictions.csv'
    # output_path = f'{OUTPUT_DIR}/transaction_predictions_{model_type}.csv'
    pred_df.to_csv(output_path, index=False)
    print(f"✅ Predictions saved to {output_path}")
    
    return pred_df

if __name__ == "__main__":
    predict_transaction_categories_alternative()