import pandas as pd
import numpy as np
import joblib
import os
from d3rlpy.algos import DQNConfig, DiscreteCQLConfig
from src.recommendation.offline_rl.T0.step2_train_offline_rl import load_dataset_components
from src.recommendation.binary_classification.T0.train_model import preprocess_unknown_values
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


tqdm.pandas()

test_T0_demog_summ = pd.read_csv('src/data/cf_demog_summary/test_T0_demog_summ.csv/test_T0_demog_summ.csv')
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# Embed summaries
def embed_summ(df):
    result = df.copy()

    # Add progress bar for the embedding process
    print("Generating embeddings...")
    result["embedding"] = result["Demog Summary"].progress_apply(
        lambda x: model.encode(x)
    )
    
    print(f"\nCompleted! Processed {len(result)} rows.")
    return result

# embedded_demog = embed_summ(test_T0_demog_summ)
# embedded_demog.to_csv('src/data/cf_demog_summary/embedded_demog/test_embedded_demog.csv', index=False)





def predict_transaction_categories_alternative(model_type='cql', top_k=5):
    """Predict with alternative dataset loading."""

    DATA_DIR = 'src/recommendation/offline_rl_llm/T0/data' 
    MODEL_DIR = 'src/recommendation/offline_rl_llm/T0/models'
    OUTPUT_DIR = 'src/recommendation/offline_rl_llm/T0/predictions'
    TEST_DATA_PATH = 'src/data/cf_demog_summary/embedded_demog/test_embedded_demog.csv'
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load preprocessor and categories
    preprocessor = joblib.load(f'{DATA_DIR}/preprocessor.pkl')
    categories = joblib.load(f'{DATA_DIR}/categories.pkl')

    # Load and preprocess test data
    df = pd.read_csv(TEST_DATA_PATH)
    df = preprocess_unknown_values(df)

    # Handle embedding column if present
    if 'embedding' in df.columns:
        # Convert string representation of embedding to numpy array
        df['embedding'] = df['embedding'].apply(lambda x: np.fromstring(x.strip("[]"), sep=' '))
        embeddings = np.stack(df['embedding'].values)
    else:
        embeddings = None
    
    # Get feature columns (excluding categories and embedding)
    feature_cols = [col for col in df.columns 
                   if col not in categories 
                   and col != 'embedding'
                   and col in df.columns]
    
    X_df = df[feature_cols]
    X = preprocessor.transform(X_df)

    # Combine features with embeddings if available
    if embeddings is not None:
        X = np.concatenate([X, embeddings], axis=1)
    
    # Load dataset for model building
    dataset = load_dataset_components(DATA_DIR)
    
    # Load model
    if model_type == 'cql':
        model_path = f'{MODEL_DIR}/cql_model.d3'
        model = DiscreteCQLConfig(
            # n_critics=2
            ).create(device=None)
    else:
        model_path = f'{MODEL_DIR}/dqn_model.d3'
        model = DQNConfig().create(device=None)
    
    model.build_with_dataset(dataset)
    model.load_model(model_path)
    print(f"Loaded model from {model_path}")
    
    # Initialize prediction DataFrame with customer IDs
    pred_df = pd.DataFrame()
    
    # Get customer ID from input data (checking multiple possible column names)
    if 'CUST_ID' in df.columns:
        pred_df['cust_id'] = df['CUST_ID']
    elif 'cust_id' in df.columns:
        pred_df['cust_id'] = df['cust_id']
    else:
        raise ValueError("No customer ID column found in input data (looked for 'CUST_ID' or 'cust_id')")
    
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
    pred_df.to_csv(output_path, index=False)
    print(f"✅ Predictions saved to {output_path}")
    
    return pred_df

if __name__ == "__main__":
    predict_transaction_categories_alternative()