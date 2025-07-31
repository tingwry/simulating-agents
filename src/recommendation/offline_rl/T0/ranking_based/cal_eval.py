import pandas as pd
import numpy as np
import json
import os

from sklearn.metrics import ndcg_score

def evaluate_transaction_predictions(transaction_predictions, ans_key, scores_df=None, output_folder="results"):
    """Evaluate with transaction count ranking while maintaining original binary metrics"""
    
    os.makedirs(output_folder, exist_ok=True)
    category_cols = [col for col in transaction_predictions.columns if col != 'cust_id']
    
    # Merge all data
    merged = pd.merge(transaction_predictions, ans_key, on='cust_id', suffixes=('_pred', '_true'))
    if scores_df is not None:
        merged = pd.merge(merged, scores_df, on='cust_id', suffixes=('', '_score'))
    
    # Initialize results list and overall metrics
    results = []
    total_customers = len(merged)
    total_predictions = 0
    total_actual = 0
    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0
    
    # Process each customer
    for _, row in merged.iterrows():
        cust_id = row['cust_id']
        pred_values = []
        true_values = []
        score_values = []
        
        for category in category_cols:
            pred_values.append(row[f'{category}_pred'])
            true_values.append(row[f'{category}_true'])
            if scores_df is not None:
                score_values.append(row[category])
        
        pred_array = np.array(pred_values)
        true_array = np.array(true_values)
        score_array = np.array(score_values) if scores_df is not None else None
        
        # Binary metrics
        true_positives = np.sum((pred_array == 1) & (true_array > 0))
        false_positives = np.sum((pred_array == 1) & (true_array == 0))
        false_negatives = np.sum((pred_array == 0) & (true_array > 0))
        
        # Get class names for each type
        tp_classes = [category_cols[i] for i in range(len(category_cols)) 
                     if pred_array[i] == 1 and true_array[i] > 0]
        fp_classes = [category_cols[i] for i in range(len(category_cols)) 
                     if pred_array[i] == 1 and true_array[i] == 0]
        fn_classes = [category_cols[i] for i in range(len(category_cols)) 
                     if pred_array[i] == 0 and true_array[i] > 0]
        
        # Calculate hit rate for this customer
        total_predictions_cust = np.sum(pred_array)
        if total_predictions_cust > 0:
            hit_rate = true_positives / total_predictions_cust
        else:
            hit_rate = 0.0
        
        # Ranking metrics (if scores available)
        if score_array is not None:
            # Normalize scores and true values for nDCG
            norm_scores = (score_array - score_array.min()) / (score_array.max() - score_array.min() + 1e-8)
            norm_true = (true_array - true_array.min()) / (true_array.max() - true_array.min() + 1e-8)
            
            # Calculate nDCG
            try:
                ndcg = ndcg_score([norm_true], [norm_scores])
            except:
                ndcg = 0.0
        else:
            ndcg = None
        
        # Store results
        results.append({
            'cust_id': cust_id,
            'hit_rate': hit_rate,
            'true_positives': true_positives,
            'classes_of_true_positives': ', '.join(tp_classes),
            'false_positives': false_positives,
            'classes_of_false_positives': ', '.join(fp_classes),
            'false_negatives': false_negatives,
            'classes_of_false_negatives': ', '.join(fn_classes),
            'ndcg': ndcg
        })
        
        # Update overall metrics
        total_predictions += total_predictions_cust
        total_actual += np.sum(true_array)
        total_true_positives += true_positives
        total_false_positives += false_positives
        total_false_negatives += false_negatives
    
    # Create detailed results DataFrame
    detailed_results = pd.DataFrame(results)
    
    # Calculate overall binary metrics
    precision = total_true_positives / (total_true_positives + total_false_positives) if (total_true_positives + total_false_positives) > 0 else 0
    recall = total_true_positives / (total_true_positives + total_false_negatives) if (total_true_positives + total_false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate accuracy (correctly predicted categories / total categories)
    total_categories = len(category_cols) * total_customers
    correct_predictions = 0
    
    for _, row in merged.iterrows():
        for category in category_cols:
            pred_col = f"{category}_pred"
            true_col = f"{category}_true"
            if pred_col in row and true_col in row:
                if row[pred_col] == (1 if row[true_col] > 0 else 0):
                    correct_predictions += 1
    
    accuracy = correct_predictions / total_categories if total_categories > 0 else 0
    overall_hit_rate = total_true_positives / total_predictions if total_predictions > 0 else 0
    
    # Create metrics dictionary
    metrics = {
        "binary_metrics": {
            "total_customers": float(total_customers),
            "total_predictions": float(total_predictions),
            "total_actual": float(total_actual),
            "true_positives": float(total_true_positives),
            "false_positives": float(total_false_positives),
            "false_negatives": float(total_false_negatives),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1_score, 4),
            "accuracy": round(accuracy, 4),
            "hit_rate": round(overall_hit_rate, 4)
        }
    }
    
    # Add ranking metrics if scores were provided
    if scores_df is not None:
        valid_ndcgs = [r['ndcg'] for r in results if r['ndcg'] is not None]
        metrics["ranking_metrics"] = {
            "average_ndcg_with_probs": np.mean(valid_ndcgs) if valid_ndcgs else 0.0
        }
    
    # Save detailed results as CSV
    csv_path = os.path.join(output_folder, "detailed_evaluation_results.csv")
    detailed_results.to_csv(csv_path, index=False)
    print(f"Detailed results saved to: {csv_path}")
    
    # Save metrics as JSON
    json_path = os.path.join(output_folder, "evaluation_metrics.json")
    with open(json_path, 'w') as f:
        json.dump({"Evaluation Metrics": metrics}, f, indent=2)
    print(f"Metrics saved to: {json_path}")
    
    # Print metrics in the requested format
    print("\nEvaluation Metrics:")
    for metric_type, metric_values in metrics.items():
        print(f"\n{metric_type}:")
        for k, v in metric_values.items():
            print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    
    return detailed_results, metrics



# Load your data
binary_predictions = pd.read_csv('src/recommendation/offline_rl/T0/ranking_based/predictions/transaction_predictions.csv')
probability_predictions = pd.read_csv('src/recommendation/offline_rl/T0/ranking_based/predictions/transaction_scores.csv')
# ans_key = pd.read_csv('src/recommendation/cluster_based/eval/ans_key.csv')
ans_key = pd.read_csv('src/data/cf_demog_summary/user_item_matrix/user_item_matrix.csv')

# Run evaluation
detailed_results, metrics = evaluate_transaction_predictions(
    binary_predictions, 
    ans_key, 
    probability_predictions,
    output_folder="src/recommendation/offline_rl/T0/ranking_based/eval_results"
)

# View first few rows of detailed results
print(detailed_results.head())