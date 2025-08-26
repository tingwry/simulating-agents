import re
import pandas as pd
import joblib
import os
import numpy as np
from tqdm import tqdm
from src.recommendation.utils.utils import *
import torch
from torch.utils.data import DataLoader, TensorDataset
import json
from src.recommendation.prompts import *
from src.client.llm import get_aoi_client

os.environ["TOKENIZERS_PARALLELISM"] = "false" 

def run_predictions(method, method_model, is_regressor, categories, threshold=None, percentile=75, data='T0'):
    DATA_DIR, MODEL_DIR, PREDICTION_OUTPUT, TEST_DATA_PATH, OPTIMAL_THRS, = prediction_path_indicator(
        method, is_regressor, method_model, threshold, data
    )
    # Load and preprocess full dataset
    df = pd.read_csv(TEST_DATA_PATH)
    df = preprocess_unknown_values(df)
    
    # Prepare features and labels
    # feature_cols = [col for col in df.columns if col not in categories and col != 'CUST_ID']
    X_df = df[feature_cols]

    # Handle customer ID
    id_col = 'CUST_ID' if 'CUST_ID' in df.columns else 'cust_id'
    if id_col not in df.columns:
        raise ValueError("No customer ID column found")
    

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
        optimal_thresholds = find_optimal_rl_thresholds(DATA_DIR, MODEL_DIR, OPTIMAL_THRS, beta=0.5)
        
        # Get Q-values for all actions (representing expected transaction counts)
        q_values = np.zeros((len(X), len(categories)))
        for action_idx in range(len(categories)):
            actions = np.full(len(X), action_idx)
            q_values[:, action_idx] = model.predict_value(X, actions)
        
        # Create output DataFrames
        predictions_df = pd.DataFrame()
        scores_df = pd.DataFrame()
        
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
        # Initialize output DataFrames
        binary_predictions = pd.DataFrame()
        prediction_scores = pd.DataFrame()
        
        binary_predictions['cust_id'] = df[id_col]
        prediction_scores['cust_id'] = df[id_col]

        if method == "binary":
            try:
                # preprocessor = joblib.load(f"{MODEL_DIR}/preprocessor{OPTIMAL_THRS}.pkl")
                print(f"{MODEL_DIR}/preprocessor{OPTIMAL_THRS}.pkl")
                preprocessor = joblib.load(f"{MODEL_DIR}/preprocessor{OPTIMAL_THRS}.pkl")
                print("✅ Loaded preprocessor from saved models")
            except FileNotFoundError:
                print("⚠️ Preprocessor not found in model directory, falling back to data loading")
                # _, preprocessor = load_and_preprocess_data(TEST_DATA_PATH)
            
            # Transform features using the loaded preprocessor
            X_all = preprocessor.transform(X_df)

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
                    binary_predictions[category] = (pred_counts > binary_threshold).astype(int)
                    
                    # For scores: use actual predicted counts (will be normalized in evaluation)
                    prediction_scores[category] = pred_counts

                    print(f"Processed: {category} (Regression, Threshold={binary_threshold:.4f})")
                else:
                    # Get probabilities and predictions using optimal threshold
                    y_proba = model.predict_proba(X_all)[:, 1]
                    y_pred = (y_proba > binary_threshold).astype(int)
                    
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
                    binary_predictions[category] = (y_proba[:, i] > current_threshold).astype(int)
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

def run_predictions_llm(method="indiv", is_regressor=False, categories=None, threshold=None, data='T0', cust_ids_to_repredict=None):
    """
    Run LLM-based predictions for transaction categories using demographic data.
    
    Args:
        method: Prediction method ("indiv" for individual)
        is_regressor: Not used for LLM but kept for consistency  
        categories: List of transaction categories to predict
        threshold: Not used for LLM but kept for consistency
        data: Data version ('T0', 'T1', 'T1_predicted')
        cust_ids_to_repredict: List of customer IDs to repredict (will replace existing predictions)
    
    Returns:
        tuple: (binary_predictions_df, prediction_scores_df)
    """
    
    # Use similar path structure as run_predictions
    DATA_DIR, MODEL_DIR, PREDICTION_OUTPUT, TEST_DATA_PATH, OPTIMAL_THRS = prediction_path_indicator(
        "llm", is_regressor, "llm", threshold, data
    )
    
    # Load and preprocess full dataset
    df = pd.read_csv(TEST_DATA_PATH)
    df = preprocess_unknown_values(df)

    if 'CUST_ID' in df.columns:
        df = df.rename(columns={'CUST_ID': 'cust_id'})
    
    # Handle customer ID
    id_col = 'CUST_ID' if 'CUST_ID' in df.columns else 'cust_id'
    if id_col not in df.columns:
        raise ValueError("No customer ID column found")
    
    # Initialize output DataFrames by loading existing results if they exist
    binary_predictions = pd.DataFrame()
    prediction_scores = pd.DataFrame()
    reasoning_df = pd.DataFrame()
    
    try:
        binary_predictions = pd.read_csv(PREDICTION_OUTPUT)
        scores_output = PREDICTION_OUTPUT.replace('.csv', '_scores.csv')
        prediction_scores = pd.read_csv(scores_output)
        reasoning_output = PREDICTION_OUTPUT.replace('.csv', '_reasoning.csv')
        reasoning_df = pd.read_csv(reasoning_output)
        print(f"Loaded existing results with {len(binary_predictions)} customers")
    except FileNotFoundError:
        print("No existing results found, creating new prediction files")
    
    # If we have specific customers to repredict, filter the test_df
    if cust_ids_to_repredict is not None:
        df = df[df[id_col].isin(cust_ids_to_repredict)]
        if len(df) == 0:
            print("No matching customers found in test data")
            return binary_predictions, prediction_scores
    
    # Initialize DataFrames if they're empty
    if binary_predictions.empty:
        binary_predictions = pd.DataFrame(columns=[id_col] + categories)
        prediction_scores = pd.DataFrame(columns=[id_col] + categories)
        reasoning_df = pd.DataFrame(columns=[id_col] + categories)
    
    # Track processed and failed customers
    processed = []
    failed = []
    
    print(f"Processing {len(df)} customers for transaction category predictions...")
    
    # Process each customer
    for i in tqdm(range(len(df)), desc="Processing customers"):
        customer_row = df.iloc[i]
        customer_id = customer_row[id_col]
        
        try:
            # Create prediction prompt for transaction categories
            prompt = create_transaction_prediction_prompt(customer_row, categories)
            print(prompt)
            
            # Get LLM prediction
            response = get_llm_prediction(prompt)
            
            # Parse response
            json_str = re.sub(r'^```json|```$', '', response.strip(), flags=re.MULTILINE).strip()
            prediction = json.loads(json_str)
            
            # Check if this customer already exists in results
            existing_idx = binary_predictions[binary_predictions[id_col] == customer_id].index
            
            if not existing_idx.empty:
                # Update existing row
                row_idx = existing_idx[0]
                update_type = "Updated"
            else:
                # Add new row
                row_idx = len(binary_predictions)
                binary_predictions.loc[row_idx, id_col] = customer_id
                prediction_scores.loc[row_idx, id_col] = customer_id
                reasoning_df.loc[row_idx, id_col] = customer_id
                update_type = "Added"
            
            # Extract predictions and scores for each category
            for category in categories:
                # Get binary prediction (0 or 1)
                binary_pred = prediction.get('predictions', {}).get(category, 0)
                binary_predictions.loc[row_idx, category] = int(binary_pred)
                
                # Get confidence score (0.0 to 1.0)
                confidence = prediction.get('confidence_scores', {}).get(category, 0.0)
                prediction_scores.loc[row_idx, category] = float(confidence)

                # Get reasoning text
                reasoning = prediction.get('reasoning', {}).get(category, 'No reasoning provided')
                reasoning_df.loc[row_idx, category] = str(reasoning)
            
            processed.append(customer_id)
            
        except Exception as e:
            print(f"❌ Error processing customer {customer_id}: {str(e)}")
            failed.append(customer_id)
            
            # Ensure the customer exists in the results (with default values)
            existing_idx = binary_predictions[binary_predictions[id_col] == customer_id].index
            row_idx = existing_idx[0] if not existing_idx.empty else len(binary_predictions)
            
            if existing_idx.empty:
                binary_predictions.loc[row_idx, id_col] = customer_id
                prediction_scores.loc[row_idx, id_col] = customer_id
                reasoning_df.loc[row_idx, id_col] = customer_id
            
            for category in categories:
                binary_predictions.loc[row_idx, category] = 0
                prediction_scores.loc[row_idx, category] = 0.0
                reasoning_df.loc[row_idx, category] = f"Failed to process: {str(e)}"
            continue
    
    # Save outputs
    os.makedirs(os.path.dirname(PREDICTION_OUTPUT), exist_ok=True)
    binary_predictions.to_csv(PREDICTION_OUTPUT, index=False)
    
    scores_output = PREDICTION_OUTPUT.replace('.csv', '_scores.csv')
    prediction_scores.to_csv(scores_output, index=False)

    reasoning_output = PREDICTION_OUTPUT.replace('.csv', '_reasoning.csv')
    reasoning_df.to_csv(reasoning_output, index=False)
    
    # Print summary
    print(f"\nProcessing complete:")
    print(f"- Successfully processed: {len(processed)} customers")
    print(f"- Failed to process: {len(failed)} customers")
    if failed:
        print(f"Failed customer IDs: {failed[:10]}{'...' if len(failed) > 10 else ''}")
    
    print(f"\nBinary predictions saved to: {PREDICTION_OUTPUT}")
    print(f"Prediction scores saved to: {scores_output}")
    print(f"Reasoning saved to: {reasoning_output}")
    
    return binary_predictions, prediction_scores

def get_llm_prediction(prompt, predicted_actions=None):
    client = get_aoi_client()

    messages = [
        {"role": "system", "content": "You are a transactional categories prediction assistant."},
        {"role": "user", "content": prompt}
    ]
    
    # Add predicted actions if provided
    # if predicted_actions:
    #     messages.append({
    #         "role": "user", 
    #         "content": f"Predicted actions at time T1: {predicted_actions}"
    #     })
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.7,
        max_tokens=1000
    )
    
    return response.choices[0].message.content




def run_rag_transaction_predictions(test_df, collection_name, categories, output_dir=None, top_k=5, cust_ids_to_repredict=None):
    """
    Main function to run RAG-based transaction category predictions.
    
    Args:
        test_df: Test dataset DataFrame
        collection_name: Qdrant collection name for similar customer retrieval
        categories: List of transaction categories to predict
        output_dir: Directory to save results
        top_k: Number of similar customers to retrieve
        cust_ids_to_repredict: List of customer IDs to repredict (will replace existing predictions)
    
    Returns:
        tuple: (binary_predictions_df, prediction_scores_df, reasoning_df)
    """
    
    print("Starting RAG-based transaction category prediction...")
    id_col = 'CUST_ID' if 'CUST_ID' in test_df.columns else 'cust_id'
    
    # Initialize output DataFrames by loading existing results if they exist
    binary_predictions = pd.DataFrame()
    prediction_scores = pd.DataFrame()
    reasoning_df = pd.DataFrame()
    
    if output_dir:
        try:
            binary_predictions = pd.read_csv(os.path.join(output_dir, "rag_transaction_predictions_t0.csv"))
            prediction_scores = pd.read_csv(os.path.join(output_dir, "rag_transaction_prediction_scores_t0.csv"))
            reasoning_df = pd.read_csv(os.path.join(output_dir, "rag_transaction_prediction_reasoning_t0.csv"))
            
            print(f"Loaded existing results with {len(binary_predictions)} customers")
        except FileNotFoundError:
            print("No existing results found, creating new prediction files")
    
    # If we have specific customers to repredict, filter the test_df
    if cust_ids_to_repredict is not None:
        test_df = test_df[test_df[id_col].isin(cust_ids_to_repredict)]
        if len(test_df) == 0:
            print("No matching customers found in test data")
            return binary_predictions, prediction_scores, reasoning_df
    
    # Initialize DataFrames if they're empty
    if binary_predictions.empty:
        binary_predictions = pd.DataFrame(columns=[id_col] + categories)
        prediction_scores = pd.DataFrame(columns=[id_col] + categories)
        reasoning_df = pd.DataFrame(columns=[id_col] + categories)
    
    # Retrieve similar customers for all test customers
    print("Step 1: Retrieving similar customers...")
    similar_customers_data = retrieve_similar_customers_for_recommendations(
        test_df, collection_name, top_k=top_k
    )
    
    # Track processed and failed customers
    processed = []
    failed = []
    
    print(f"Step 2: Processing {len(test_df)} customers for predictions...")
    
    # Process each customer
    for i in tqdm(range(len(test_df)), desc="Processing customers"):
        customer_row = test_df.iloc[i]
        customer_id = customer_row[id_col]
        
        try:
            # Get similar customers data for this customer
            cust_similar_data = similar_customers_data.get(customer_id, {})
            
            if not cust_similar_data.get('similar_customers'):
                print(f"\nWarning: No similar customers found for {customer_id}")
                cust_similar_data = {'similar_customers': []}
            
            # Create RAG-enhanced prediction prompt
            prompt = create_rag_transaction_prediction_prompt(
                customer_row, categories, cust_similar_data
            )
            
            # Get LLM prediction
            response = get_llm_prediction(prompt)
            
            # Parse response
            json_str = re.sub(r'^```json|```$', '', response.strip(), flags=re.MULTILINE).strip()
            prediction = json.loads(json_str)
            
            # Check if this customer already exists in results
            existing_idx = binary_predictions[binary_predictions[id_col] == customer_id].index
            
            if not existing_idx.empty:
                # Update existing row
                row_idx = existing_idx[0]
                update_type = "Updated"
            else:
                # Add new row
                row_idx = len(binary_predictions)
                binary_predictions.loc[row_idx, id_col] = customer_id
                prediction_scores.loc[row_idx, id_col] = customer_id
                reasoning_df.loc[row_idx, id_col] = customer_id
                update_type = "Added"
            
            # Update predictions for each category
            for category in categories:
                # Get binary prediction (0 or 1)
                binary_pred = prediction.get('predictions', {}).get(category, 0)
                binary_predictions.loc[row_idx, category] = int(binary_pred)
                
                # Get confidence score (0.0 to 1.0)
                confidence = prediction.get('confidence_scores', {}).get(category, 0.0)
                prediction_scores.loc[row_idx, category] = float(confidence)
                
                # Get reasoning text
                reasoning = prediction.get('reasoning', {}).get(category, 'No reasoning provided')
                reasoning_df.loc[row_idx, category] = str(reasoning)
            
            processed.append(customer_id)
            print(f"{update_type} predictions for customer {customer_id}")
            
        except Exception as e:
            print(f"❌ Error processing customer {customer_id}: {str(e)}")
            failed.append(customer_id)
            
            # Ensure the customer exists in the results (with default values)
            existing_idx = binary_predictions[binary_predictions[id_col] == customer_id].index
            row_idx = existing_idx[0] if not existing_idx.empty else len(binary_predictions)
            
            if existing_idx.empty:
                binary_predictions.loc[row_idx, id_col] = customer_id
                prediction_scores.loc[row_idx, id_col] = customer_id
                reasoning_df.loc[row_idx, id_col] = customer_id
            
            for category in categories:
                binary_predictions.loc[row_idx, category] = 0
                prediction_scores.loc[row_idx, category] = 0.0
                reasoning_df.loc[row_idx, category] = f"Failed to process: {str(e)}"
            continue
    
    # Save outputs
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        binary_output = os.path.join(output_dir, "rag_transaction_predictions_t0.csv")
        binary_predictions.to_csv(binary_output, index=False)
        
        scores_output = os.path.join(output_dir, "rag_transaction_prediction_scores_t0.csv")
        prediction_scores.to_csv(scores_output, index=False)
        
        reasoning_output = os.path.join(output_dir, "rag_transaction_prediction_reasoning_t0.csv")
        reasoning_df.to_csv(reasoning_output, index=False)
        
        print(f"\nResults saved to:")
        print(f"- Binary predictions: {binary_output}")
        print(f"- Prediction scores: {scores_output}")
        print(f"- Reasoning: {reasoning_output}")
    
    # Print summary
    print(f"\nProcessing complete:")
    print(f"- Successfully processed: {len(processed)} customers")
    print(f"- Failed to process: {len(failed)} customers")
    if failed:
        print(f"Failed customer IDs: {failed[:10]}{'...' if len(failed) > 10 else ''}")
    
    return binary_predictions, prediction_scores, reasoning_df



if __name__ == "__main__":
    categories = ['loan','utility','finance','shopping','financial_services', 'health_and_care', 'home_lifestyle', 'transport_travel',	
                 'leisure', 'public_services']

    transaction_amount_cols = [f'{cat}_t0' for cat in categories]

    demographic_features = ['Number of Children', 'Age', 'Gender', 'Education level', 
                            'Marital status', 'Region', 'Occupation Group']

    feature_cols = demographic_features + transaction_amount_cols



    COLLECTION_NAME = "customers"
    OUTPUT_DIR = "src/recommendation/predictions/llm/rag/results"
    test_df = pd.read_csv("src/recommendation/data/rag/test_T0_demog_summ/test_T0_demog_summ_v1.csv")
    testtest = test_df.head()
    # testtest = test_df[test_df['CUST_ID'].isin([1052, 1171, 2214, 2930, 2964, 3463, 4095, 4225])]

    # run_predictions(method="binary", is_regressor=True, categories=categories, method_model="random_forests", threshold=None)
    # run_predictions(method="binary", is_regressor=False, categories=categories, method_model="random_forests", threshold=None)
    # run_predictions(method="binary", is_regressor=True, categories=categories, method_model="random_forests", threshold=0)
    # run_predictions(method="binary", is_regressor=True, categories=categories, method_model="catboost", threshold=None)
    # run_predictions(method="binary", is_regressor=False, categories=categories, method_model="catboost", threshold=None)
    # run_predictions(method="binary", is_regressor=True, categories=categories, method_model="catboost", threshold=0)

    # run_predictions(method="multilabel", is_regressor=False, categories=categories, method_model="multioutputclassifier", threshold=None)
    # run_predictions(method="multilabel", is_regressor=False, categories=categories, method_model="neural_network", threshold=None)

    # run_predictions(method="reinforcement_learning", is_regressor=False,categories=categories, method_model=None, threshold=None, percentile=75)


    
    # T1 predictions
    # run_predictions(method="binary", is_regressor=False, categories=categories, 
    #                method_model="catboost", threshold=None, data='T1')
    # run_predictions(method="multilabel", is_regressor=False, categories=categories, method_model="neural_network", threshold=None, data='T1')
    # run_predictions(method="binary", is_regressor=False, categories=categories, method_model="random_forests", threshold=None, data='T1')
    
    # T1_predicted predictions
    # run_predictions(method="binary", is_regressor=False, categories=categories, 
    #                method_model="catboost", threshold=None, data='T1_predicted')
    # run_predictions(method="multilabel", is_regressor=False, categories=categories, method_model="neural_network", threshold=None, data='T1_predicted')
    # run_predictions(method="binary", is_regressor=False, categories=categories, method_model="random_forests", threshold=None, data='T1_predicted')

#     cust_ids_to_repredict = [2993, 3211, 3594, 3900]

#     binary_preds, scores = run_predictions_llm(
#     method="indiv", 
#     categories=categories, 
#     data='T0',
#     cust_ids_to_repredict=cust_ids_to_repredict
# )

    binary_preds, scores_preds, reasoning_preds = run_rag_transaction_predictions(
        test_df=testtest,
        collection_name=COLLECTION_NAME,
        categories=categories,
        output_dir=OUTPUT_DIR,
        top_k=5,
    )
    

    # List of customer IDs you want to repredict
    # cust_ids_to_repredict = [2214]

    # binary_preds, scores_preds, reasoning_preds = run_rag_transaction_predictions(
    #     test_df=test_df,  # Your full test DataFrame
    #     collection_name=COLLECTION_NAME,
    #     categories=categories,
    #     output_dir=OUTPUT_DIR,
    #     top_k=5,
    #     cust_ids_to_repredict=cust_ids_to_repredict  # This specifies which customers to repredict
    # )

    # print("RAG-based transaction category prediction completed!")