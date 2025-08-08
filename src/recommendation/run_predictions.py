import pandas as pd
import joblib
import os
import numpy as np
from src.recommendation.utils.utils import *
import torch
from torch.utils.data import DataLoader, TensorDataset
import json

# PREDICTION_OUTPUT = 'src/recommendation/catboost/predictions/transaction_predictions_grouped_catbased.csv'
# PREDICTION_OUTPUT_T1 = 'src/recommendation/catboost/T1/predictions/transaction_predictions_grouped_catbased.csv'


def run_predictions(method, method_model, is_regressor, categories, threshold=None, percentile=75):
    DATA_DIR, MODEL_DIR, PREDICTION_OUTPUT, OPTIMAL_THRS = prediction_path_indicator(method, is_regressor, method_model, threshold)
    # Load and preprocess full dataset
    df = pd.read_csv(TEST_DATA_PATH)
    df = preprocess_unknown_values(df)
    
    # Prepare features and labels
    feature_cols = [col for col in df.columns if col not in categories]
    

    if method == "reinforcement_learning":
        # Load preprocessor and categories
        preprocessor = joblib.load(f'{DATA_DIR}/preprocessor.pkl')
        categories = joblib.load(f'{DATA_DIR}/categories.pkl')

        X = preprocessor.transform(df[feature_cols])
    
        # Load model
        model_path = f'{MODEL_DIR}/cql_model_txn_counts{OPTIMAL_THRS}.d3'
        model = DiscreteCQLConfig().create(device=None)
        model.build_with_dataset(load_dataset_components(DATA_DIR))
        model.load_model(model_path)
        
        # Get optimal thresholds for each category
        optimal_thresholds = find_percentile_thresholds(DATA_DIR, MODEL_DIR, OPTIMAL_THRS, percentile)
        
        # Get Q-values for all actions (representing expected transaction counts)
        q_values = np.zeros((len(X), len(categories)))
        for action_idx in range(len(categories)):
            actions = np.full(len(X), action_idx)
            q_values[:, action_idx] = model.predict_value(X, actions)
        
        # Create output DataFrames
        predictions_df = pd.DataFrame()
        scores_df = pd.DataFrame()
        
        # Handle customer ID
        id_col = 'CUST_ID' if 'CUST_ID' in df.columns else 'cust_id'
        predictions_df['cust_id'] = df[id_col]
        scores_df['cust_id'] = df[id_col]
        
        # Store predictions and scores with percentile-based thresholds
        for i, category in enumerate(categories):
            threshold = optimal_thresholds[category]
            predictions_df[category] = (q_values[:, i] > threshold).astype(int)
            scores_df[category] = q_values[:, i]
        
        # Save outputs
        predictions_df.to_csv(PREDICTION_OUTPUT, index=False)

        scores_output = PREDICTION_OUTPUT.replace('.csv', '_scores.csv')
        scores_df.to_csv(scores_output, index=False)
        
        print(f"✅ Predictions saved with percentile-based thresholds (percentile={percentile})")
        print("Category thresholds:", optimal_thresholds)
        return predictions_df, scores_df
    
    else:
        _, preprocessor = load_and_preprocess_data(TEST_DATA_PATH)

        X_df = df[feature_cols]
        # Initialize output DataFrames
        binary_predictions = pd.DataFrame()
        prediction_scores = pd.DataFrame()
        
        # Handle customer ID
        id_col = 'CUST_ID' if 'CUST_ID' in df.columns else 'cust_id'
        if id_col not in df.columns:
            raise ValueError("No customer ID column found")
        binary_predictions['cust_id'] = df[id_col]
        prediction_scores['cust_id'] = df[id_col]

        if method == "binary":
            X_all = preprocessor.fit_transform(X_df)

            for category in categories:
                model_path = f"{MODEL_DIR}/{category}_model{OPTIMAL_THRS}.pkl"
                
                if not os.path.exists(model_path):
                    print(f"Warning: Model for '{category}' not found. Using defaults...")
                    binary_predictions[category] = 0
                    prediction_scores[category] = 0.0
                    continue

                model_data = joblib.load(model_path)
                model = model_data['model']

                binary_threshold = model_data.get('optimal_threshold', threshold)

                if is_regressor:
                    # Regression model - predict transaction counts
                    pred_counts = model.predict(X_all)

                    # For binary evaluation: 1 if count > 0, else 0
                    binary_predictions[category] = (pred_counts >= binary_threshold).astype(int)
                    
                    # For scores: use actual predicted counts (will be normalized in evaluation)
                    prediction_scores[category] = pred_counts

                    print(f"Processed: {category} (Regression, Threshold={binary_threshold:.4f})")
                else:
                    # Get probabilities and predictions using optimal threshold
                    y_proba = model.predict_proba(X_all)[:, 1]
                    y_pred = (y_proba >= binary_threshold).astype(int)
                    
                    binary_predictions[category] = y_pred
                    prediction_scores[category] = y_proba
                    
                    print(f"Processed: {category} (Classification, Threshold={binary_threshold:.4f})")

            
        elif method == "multilabel":
            if method_model == "multioutputclassifier":
                model_data = joblib.load(f"{MODEL_DIR}/multioutput_model{OPTIMAL_THRS}.pkl")
                pipeline = model_data['model']
                preprocessor = model_data['preprocessor']
                scaler = model_data['scaler']
                optimal_thresholds = model_data.get('optimal_thresholds', {})

                X_all = preprocessor.transform(X_df)
                X_all = scaler.transform(X_all)

                # Get probabilities for each class
                y_proba = np.array([est.predict_proba(X_all)[:, 1] for est in pipeline.steps[-1][1].estimators_]).T
                
                # Apply thresholds
                for i, category in enumerate(categories):
                    # Use provided threshold if available, otherwise use optimal threshold from training
                    current_threshold = threshold if threshold is not None else optimal_thresholds.get(category, 0.5)
                    binary_predictions[category] = (y_proba[:, i] >= current_threshold).astype(int)
                    prediction_scores[category] = y_proba[:, i]
                    
                    print(f"Processed: {category} (Threshold={current_threshold:.4f})")

            elif method_model == "neural_network":
                # Load model weights and metadata
                weights_path = f"{MODEL_DIR}/best_model_weights{OPTIMAL_THRS}.pth"
                metadata = joblib.load(f"{MODEL_DIR}/model_metadata{OPTIMAL_THRS}.pkl")
                
                # Initialize model
                model = MultiLabelClassifier(metadata['input_size'], metadata['output_size'])
                model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
                model.eval()
                
                # Load preprocessor and scaler
                preprocessor = joblib.load(f"{MODEL_DIR}/preprocessor{OPTIMAL_THRS}.pkl")
                scaler = joblib.load(f"{MODEL_DIR}/scaler{OPTIMAL_THRS}.pkl")
                
                # Load optimal thresholds if available
                thresholds_path = f"{MODEL_DIR}/optimal_thresholds.json"
                if os.path.exists(thresholds_path):
                    with open(thresholds_path, 'r') as f:
                        optimal_thresholds = json.load(f)
                else:
                    optimal_thresholds = {}

                X_all = preprocessor.transform(X_df)
                X_all = scaler.transform(X_all)
                
                # Convert to PyTorch tensors
                X_tensor = torch.tensor(X_all, dtype=torch.float32)
                
                # Create DataLoader
                dataset = TensorDataset(X_tensor)
                dataloader = DataLoader(dataset, batch_size=256, shuffle=False)
                
                # Get predictions in batches
                all_preds = []
                with torch.no_grad():
                    for batch in dataloader:
                        inputs = batch[0]
                        outputs = model(inputs)
                        all_preds.append(outputs.numpy())
                
                # Concatenate all predictions
                preds = np.concatenate(all_preds, axis=0)
                
                # Fill DataFrames
                for i, category in enumerate(categories):
                    # Use provided threshold if available, otherwise use optimal threshold from training
                    current_threshold = threshold if threshold is not None else optimal_thresholds.get(category, 0.5)
                    binary_predictions[category] = (preds[:, i] > current_threshold).astype(int)
                    prediction_scores[category] = preds[:, i]
                    
                    print(f"Processed: {category} (Threshold={current_threshold:.4f})")
            
        # Save outputs
        os.makedirs(os.path.dirname(PREDICTION_OUTPUT), exist_ok=True)
        binary_predictions.to_csv(PREDICTION_OUTPUT, index=False)
        
        scores_output = PREDICTION_OUTPUT.replace('.csv', '_scores.csv')
        prediction_scores.to_csv(scores_output, index=False)
        
        print(f"\nBinary predictions saved to: {PREDICTION_OUTPUT}")
        print(f"Prediction scores saved to: {scores_output}")

        return binary_predictions, prediction_scores



if __name__ == "__main__":
    TEST_DATA_PATH = 'src/recommendation/data/T0/test_with_lifestyle.csv'

    categories = ['loan','utility','finance','shopping','financial_services', 'health_and_care', 'home_lifestyle', 'transport_travel',	
                 'leisure', 'public_services']

    # run_predictions(method="binary", is_regressor=True, categories=categories, method_model="random_forests", threshold=None)
    # run_predictions(method="binary", is_regressor=False, categories=categories, method_model="random_forests", threshold=None)
    # run_predictions(method="binary", is_regressor=True, categories=categories, method_model="random_forests", threshold=0.2)
    # run_predictions(method="binary", is_regressor=False, categories=categories, method_model="random_forests", threshold=0.5)
    # run_predictions(method="binary", is_regressor=True, categories=categories, method_model="catboost", threshold=None)
    # run_predictions(method="binary", is_regressor=False, categories=categories, method_model="catboost", threshold=None)
    # run_predictions(method="binary", is_regressor=True, categories=categories, method_model="catboost", threshold=0.2)
    # run_predictions(method="binary", is_regressor=False, categories=categories, method_model="catboost", threshold=0.5)

    # run_predictions(method="multilabel", is_regressor=False, categories=categories, method_model="multioutputclassifier", threshold=None)
    # run_predictions(method="multilabel", is_regressor=False, categories=categories, method_model="neural_network", threshold=None)
    # run_predictions(method="multilabel", is_regressor=False, categories=categories, method_model="multioutputclassifier", threshold=0.5)
    run_predictions(method="multilabel", is_regressor=False, categories=categories, method_model="neural_network", threshold=0.5)

    # run_predictions(method="reinforcement_learning", is_regressor=False,categories=categories, method_model=None, threshold=None, percentile=75)