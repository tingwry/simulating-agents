import pandas as pd
import numpy as np
import json
import os

def evaluate_transaction_predictions(transaction_predictions, ans_key, output_folder="results"):
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
    
    # Initialize results list
    results = []
    
    # Overall metrics initialization
    total_customers = len(merged)
    total_predictions = 0
    total_actual = 0
    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0
    
    # Process each customer
    for _, row in merged.iterrows():
        cust_id = row['cust_id']
        
        # Get predictions and true values for this customer
        pred_values = []
        true_values = []
        
        for category in category_cols:
            pred_col = f"{category}_pred"
            true_col = f"{category}_true"
            
            if pred_col in row and true_col in row:
                pred_values.append(row[pred_col])
                true_values.append(row[true_col])
        
        pred_array = np.array(pred_values)
        true_array = np.array(true_values)
        
        # Calculate metrics for this customer
        true_positives = np.sum((pred_array == 1) & (true_array == 1))
        false_positives = np.sum((pred_array == 1) & (true_array == 0))
        false_negatives = np.sum((pred_array == 0) & (true_array == 1))
        
        # Get class names for each type
        tp_classes = [category_cols[i] for i in range(len(category_cols)) 
                     if pred_array[i] == 1 and true_array[i] == 1]
        fp_classes = [category_cols[i] for i in range(len(category_cols)) 
                     if pred_array[i] == 1 and true_array[i] == 0]
        fn_classes = [category_cols[i] for i in range(len(category_cols)) 
                     if pred_array[i] == 0 and true_array[i] == 1]
        
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
    
    # Create detailed results DataFrame
    detailed_results = pd.DataFrame(results)
    
    # Calculate overall metrics
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
                if row[pred_col] == row[true_col]:
                    correct_predictions += 1
    
    accuracy = correct_predictions / total_categories if total_categories > 0 else 0
    overall_hit_rate = total_true_positives / total_predictions if total_predictions > 0 else 0
    
    # Create metrics dictionary
    metrics = {
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
    print(f"total_customers: {metrics['total_customers']:.4f}")
    print(f"total_predictions: {metrics['total_predictions']:.4f}")
    print(f"total_actual: {metrics['total_actual']:.4f}")
    print(f"true_positives: {metrics['true_positives']:.4f}")
    print(f"false_positives: {metrics['false_positives']:.4f}")
    print(f"false_negatives: {metrics['false_negatives']:.4f}")
    print(f"precision: {metrics['precision']:.4f}")
    print(f"recall: {metrics['recall']:.4f}")
    print(f"f1_score: {metrics['f1_score']:.4f}")
    print(f"accuracy: {metrics['accuracy']:.4f}")
    print(f"hit_rate: {metrics['hit_rate']:.4f}")
    
    return detailed_results, metrics


# Load your data
transaction_predictions = pd.read_csv('src/recommendation/binary_classification_log_reg/T0/predictions/transaction_predictions.csv')
ans_key = pd.read_csv('src/recommendation/cluster_based/eval/ans_key.csv')

# Prep transaction_predictions
transaction_predictions = transaction_predictions.loc[:, ~transaction_predictions.columns.str.endswith('_reasoning')]

# Run evaluation
detailed_results, metrics = evaluate_transaction_predictions(
    transaction_predictions, 
    ans_key, 
    output_folder="src/recommendation/binary_classification_log_reg/T0/eval_results"
)

# View first few rows of detailed results
print(detailed_results.head())
