import pandas as pd
import numpy as np
import json
import os
from sklearn.metrics import ndcg_score

def evaluate_transaction_predictions(transaction_predictions, ans_key, probabilities_df=None, output_folder="results"):
    """
    Evaluate transaction predictions against answer key and save results.
    
    Parameters:
    - transaction_predictions: DataFrame with predictions (0/1 values)
    - ans_key: DataFrame with ground truth (0.0/1.0 values)
    - output_folder: Folder to save results
    
    Returns:
    - detailed_results: DataFrame with per-customer results
    - metrics: Dictionary with overall metrics
    """
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get category columns (exclude cust_id)
    category_cols = [col for col in transaction_predictions.columns if col != 'cust_id']
    
    # Merge dataframes on cust_id
    merged = pd.merge(transaction_predictions, ans_key, on='cust_id', how='inner', suffixes=('_pred', '_true'))
    
    # Merge probabilities if provided
    if probabilities_df is not None:
        merged = pd.merge(merged, probabilities_df, on='cust_id', suffixes=('', '_prob'))

    # Initialize results list
    results = []
    metrics = {
        'binary_metrics': {},
        'ranking_metrics': {}
    }
    
    # Binary metrics initialization
    total_customers = len(merged)
    total_predictions = 0
    total_actual = 0
    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0
    
    # Prepare arrays for sklearn's ndcg_score
    all_true_scores = []
    all_pred_scores = []
    all_prob_scores = [] if probabilities_df is not None else None

    # Process each customer
    for _, row in merged.iterrows():
        cust_id = row['cust_id']
        
        # Get predictions and true values for this customer
        pred_values = []
        true_values = []
        prob_values = [] if probabilities_df is not None else None
        
        for category in category_cols:
            pred_col = f"{category}_pred" if f"{category}_pred" in row else category
            true_col = f"{category}_true" if f"{category}_true" in row else category

            pred_values.append(row[pred_col])
            true_values.append(row.get(true_col, 0))

            if probabilities_df is not None:
                prob_values.append(row.get(category, 0.0))
        
        pred_array = np.array(pred_values)
        true_array = np.array(true_values)
        prob_array = np.array(prob_values) if probabilities_df is not None else None
        
        # Store scores for ndcg calculation
        all_true_scores.append(true_array)
        all_pred_scores.append(pred_array)
        if probabilities_df is not None:
            all_prob_scores.append(prob_array)
        
        # Calculate binary metrics
        true_positives = np.sum((pred_array == 1) & (true_array > 0))
        false_positives = np.sum((pred_array == 1) & (true_array == 0))
        false_negatives = np.sum((pred_array == 0) & (true_array > 0))

        # Get class names for each type
        tp_classes = [category_cols[i] for i in range(len(category_cols)) 
                     if pred_array[i] > 0 and true_array[i] > 0]
        fp_classes = [category_cols[i] for i in range(len(category_cols)) 
                     if pred_array[i] > 0 and true_array[i] == 0]
        fn_classes = [category_cols[i] for i in range(len(category_cols)) 
                     if pred_array[i] == 0 and true_array[i] > 0]
        
        # Calculate hit rate for this customer
        total_predictions_cust = np.sum(pred_array)
        if total_predictions_cust > 0:
            hit_rate = true_positives / total_predictions_cust
        else:
            hit_rate = 0.0
        
        # Store results
        results.append({
            'cust_id': cust_id,
            'hit_rate': hit_rate,
            'true_positives': true_positives,
            'classes_of_true_positives': ', '.join(tp_classes),
            'false_positives': false_positives,
            'classes_of_false_positives': ', '.join(fp_classes),
            'false_negatives': false_negatives,
            'classes_of_false_negatives': ', '.join(fn_classes)
        })
        
        # Update overall metrics
        total_predictions += total_predictions_cust
        total_actual += np.sum(true_array)
        total_true_positives += true_positives
        total_false_positives += false_positives
        total_false_negatives += false_negatives
    
    # Calculate NDCG scores using sklearn's implementation
    if len(all_true_scores) > 0:
        # Convert to 2D arrays (customers × categories)
        true_scores = np.vstack(all_true_scores)
        pred_scores = np.vstack(all_pred_scores)
        
        # Calculate NDCG
        avg_ndcg = ndcg_score(true_scores, pred_scores)
        
        if probabilities_df is not None:
            prob_scores = np.vstack(all_prob_scores)
            avg_ndcg_with_probs = ndcg_score(true_scores, prob_scores)
        else:
            avg_ndcg_with_probs = None
    else:
        avg_ndcg = 0.0
        avg_ndcg_with_probs = None
    
    # Calculate binary metrics
    precision = total_true_positives / (total_true_positives + total_false_positives) if (total_true_positives + total_false_positives) > 0 else 0
    recall = total_true_positives / (total_true_positives + total_false_negatives) if (total_true_positives + total_false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate accuracy (correctly predicted categories / total categories)
    total_categories = len(category_cols) * total_customers
    correct_predictions = 0
    
    for _, row in merged.iterrows():
        for category in category_cols:
            pred_col = f"{category}_pred"
            true_col = f"{category}_true"
            if pred_col in row and true_col in row:
                if row[pred_col] == row[true_col]:
                    correct_predictions += 1
    
    accuracy = correct_predictions / total_categories if total_categories > 0 else 0
    overall_hit_rate = total_true_positives / total_predictions if total_predictions > 0 else 0
    
    metrics['binary_metrics'] = {
        "total_customers": float(total_customers),
        "total_predictions": float(total_predictions),
        "total_actual": float(total_actual),
        "true_positives": float(total_true_positives),
        "false_positives": float(total_false_positives),
        "false_negatives": float(total_false_negatives),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "hit_rate": round(overall_hit_rate, 4)
    }
    
    # Calculate ranking metrics
    metrics['ranking_metrics'] = {
        'average_ndcg': avg_ndcg,
        'average_ndcg_with_probs': avg_ndcg_with_probs
    }
    
    # Add NDCG scores to detailed results
    detailed_results = pd.DataFrame(results)
    if 'ndcg' not in detailed_results.columns:
        detailed_results['ndcg'] = avg_ndcg
    if 'ndcg_with_probs' not in detailed_results.columns and avg_ndcg_with_probs is not None:
        detailed_results['ndcg_with_probs'] = avg_ndcg_with_probs

    # Save results and metrics
    csv_path = os.path.join(output_folder, "detailed_evaluation_results.csv")
    detailed_results.to_csv(csv_path, index=False)
    print(f"Detailed results saved to: {csv_path}")
    
    json_path = os.path.join(output_folder, "evaluation_metrics.json")
    with open(json_path, 'w') as f:
        json.dump({"Evaluation Metrics": metrics}, f, indent=2)
    print(f"Metrics saved to: {json_path}")
    
    return detailed_results, metrics


if __name__ == "__main__":
    binary_predictions = pd.read_csv('src/recommendation/binary_classification_rand_reg/T0/predictions/transaction_predictions_grouped_catbased.csv')
    probability_predictions = pd.read_csv('src/recommendation/binary_classification_rand_reg/T0/predictions/transaction_predictions_grouped_catbased_scores.csv')
    ans_key = pd.read_csv('src/data/cf_demog_summary/user_item_matrix/user_item_matrix_grouped_catbased.csv')
    eval_result_path = "src/recommendation/binary_classification_rand_reg/T0/eval_results_grouped_catbased"

    # Run evaluation
    detailed_results, metrics = evaluate_transaction_predictions(
        transaction_predictions=binary_predictions, 
        ans_key=ans_key, 
        probabilities_df=probability_predictions,
        output_folder=eval_result_path
    )

    # View first few rows of detailed results
    print(detailed_results.head())