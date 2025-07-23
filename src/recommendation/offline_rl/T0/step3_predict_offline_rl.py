import pandas as pd
import numpy as np
import joblib
from d3rlpy.algos import DQNConfig, CQL, CQLConfig
from d3rlpy.dataset import ReplayBuffer, InfiniteBuffer
from src.recommendation.binary_classification.T0.train_model import load_and_preprocess_data

TEST_DATA_PATH = 'src/data/T0/test_with_lifestyle.csv'
model_path = 'src/recommendation/offline_rl/T0/models/dqn_model.d3'
output_path = 'src/recommendation/offline_rl/T0/predictions/transaction_predictions.csv'

dataset_path = 'src/recommendation/offline_rl/T0/data/offline_rl_dataset.h5'

def predict_top_actions():
    with open(dataset_path, "rb") as f:
        dataset = ReplayBuffer.load(f, InfiniteBuffer())


    df, preprocessor, categories = load_and_preprocess_data(TEST_DATA_PATH)
    feature_cols = [col for col in df.columns if col not in categories]
    X_df = df[feature_cols]
    X = preprocessor.fit_transform(X_df)

    cql = CQLConfig().create(device=None)
    cql.build_with_dataset(dataset)
    cql.load_model(model_path)
    # dqn = DQNConfig().create(device=None)
    # dqn.build_with_dataset(dataset)
    # dqn.load_model(model_path)

    pred_df = pd.DataFrame()
    pred_df['cust_id'] = df['cust_id'] if 'cust_id' in df.columns else np.arange(len(df))

    # Predict Q-values for each customer and choose top categories
    # q_values = dqn.predict_value(X)
    # actions = dqn.predict(X)
    # q_values = dqn.predict_value(X, actions)
    actions = cql.predict(X)

    for i, category in enumerate(categories):
        pred_df[category] = (actions == i).astype(int)

    # for i, category in enumerate(categories):
    #     pred_df[category] = (q_values[:, i] > 0).astype(int)

    pred_df.to_csv(output_path, index=False)
    print(f"✅ RL Predictions saved to {output_path}")

if __name__ == "__main__":
    predict_top_actions()
