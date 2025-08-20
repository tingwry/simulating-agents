import os
import json
import pandas as pd

def extract_all_evaluation_metrics(root_dir, output_csv_path, baseline_paths=None):
    """
    Traverse the evaluation results folder and compile both default and optimal threshold 
    evaluation metrics into a single CSV file, including harmonic mean and external baseline evaluations.
    
    Args:
        root_dir (str): Root folder containing the eval_results.
        output_csv_path (str): Path to save the final CSV summary.
        baseline_paths (list): List of paths to baseline evaluation_metrics.json files.
    """
    records = []

    # ----------- MAIN TREE SCAN -----------
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in ["evaluation_metrics.json", "evaluation_metrics_optimal_thrs.json"]:
            if fname in filenames:
                json_path = os.path.join(dirpath, fname)
                with open(json_path, 'r') as f:
                    data = json.load(f)

                # Relative path from root_dir to current file location
                rel_path = os.path.relpath(dirpath, root_dir)
                path_parts = rel_path.split(os.sep)

                # Handle normal case
                if len(path_parts) >= 2:
                    task_type = path_parts[0].replace("_", " ").title()
                    model_name = path_parts[1].replace("_", " ").title()
                # Special case: folder like 'rl'
                elif len(path_parts) == 1:
                    task_type = path_parts[0].replace("_", " ").title()
                    model_name = '-'
                else:
                    continue

                threshold_type = "Optimal Threshold" if "optimal_thrs" in fname else "Default Threshold"

                binary_metrics = data.get("Evaluation Metrics", {}).get("binary_metrics", {})
                ranking_metrics = data.get("Evaluation Metrics", {}).get("ranking_metrics", {})

                f1_score = binary_metrics.get("f1_score")
                f_beta_score = binary_metrics.get("f_beta_score")
                ndcg = ranking_metrics.get("average_ndcg_with_probs")

                # Harmonic mean of F1 and NDCG
                if f1_score and ndcg and (f1_score + ndcg) > 0:
                    harmonic_mean = (2 * f1_score * ndcg) / (f1_score + ndcg)
                else:
                    harmonic_mean = None

                # Harmonic mean of F-Beta and NDCG
                if f_beta_score and ndcg and (f_beta_score + ndcg) > 0:
                    harmonic_mean_fbeta = (2 * f_beta_score * ndcg) / (f_beta_score + ndcg)
                else:
                    harmonic_mean_fbeta = None

                record = {
                    "Method": task_type,
                    "Model": model_name,
                    "Threshold": threshold_type,
                    "Total Predictions": binary_metrics.get("total_predictions"),
                    "True Positives": binary_metrics.get("true_positives"),
                    "False Positives": binary_metrics.get("false_positives"),
                    "False Negatives": binary_metrics.get("false_negatives"),
                    "Precision": binary_metrics.get("precision"),
                    "Recall": binary_metrics.get("recall"),
                    "F1 Score": f1_score,
                    "F-Beta Score": f_beta_score,
                    "Accuracy": binary_metrics.get("accuracy"),
                    "NDCG": ndcg,
                    "Harmonic Mean (F1, NDCG)": harmonic_mean,
                    "Harmonic Mean (F-Beta, NDCG)": harmonic_mean_fbeta,
                }
                records.append(record)

    # ----------- BASELINE ADDITIONS -----------
    if baseline_paths:
        for baseline_path in baseline_paths:
            try:
                with open(baseline_path, 'r') as f:
                    data = json.load(f)
                binary_metrics = data.get("Evaluation Metrics", {}).get("binary_metrics", {})
                f1_score = binary_metrics.get("f1_score")
                f_beta_score = binary_metrics.get("f_beta_score")
                accuracy = binary_metrics.get("accuracy")
                precision = binary_metrics.get("precision")
                recall = binary_metrics.get("recall")
                total_predictions = binary_metrics.get("total_predictions")
                tp = binary_metrics.get("true_positives")
                fp = binary_metrics.get("false_positives")
                fn = binary_metrics.get("false_negatives")

                record = {
                    "Method": "Baseline All 1",
                    "Model": "-",
                    "Threshold": "-",
                    "Total Predictions": total_predictions,
                    "True Positives": tp,
                    "False Positives": fp,
                    "False Negatives": fn,
                    "Precision": precision,
                    "Recall": recall,
                    "F1 Score": f1_score,
                    "F-Beta Score": f_beta_score,
                    "Accuracy": accuracy,
                    "NDCG": "-",
                    "Harmonic Mean (F1, NDCG)": "-",
                    "Harmonic Mean (F-Beta, NDCG)": "-",
                }
                records.append(record)
            except Exception as e:
                print(f"Failed to load baseline from {baseline_path}: {e}")

    # ----------- SAVE TO CSV -----------
    df = pd.DataFrame(records)
    df.to_csv(output_csv_path, index=False)
    print(f"Saved evaluation summary to: {output_csv_path}")


def merge_catboost_classifier_evaluations(root_dir, output_csv_path):
    """
    Merge DEFAULT THRESHOLD evaluation metrics for CatBoost classifier across T0, T1, and T1_predicted cases.
    
    Args:
        root_dir (str): Root folder containing the eval_results
        output_csv_path (str): Path to save the final CSV summary
    """
    records = []
    
    # Define the paths we need to check
    cases = [
        ('T0', 'binary_classification/catboost_classifier'),
        ('T1', 'binary_classification/T1/catboost_classifier'),
        ('T1_predicted', 'binary_classification/T1_predicted/catboost_classifier')
    ]
    
    for case_name, rel_path in cases:
        case_dir = os.path.join(root_dir, rel_path)
        file_path = os.path.join(case_dir, 'evaluation_metrics_optimal_thrs.json')
        
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            binary_metrics = data.get("Evaluation Metrics", {}).get("binary_metrics", {})
            ranking_metrics = data.get("Evaluation Metrics", {}).get("ranking_metrics", {})
            
            f1_score = binary_metrics.get("f1_score")
            f_beta_score = binary_metrics.get("f_beta_score")
            ndcg = ranking_metrics.get("average_ndcg_with_probs")
            
            # Calculate harmonic mean of F1 and NDCG
            if f1_score and ndcg and (f1_score + ndcg) > 0:
                harmonic_mean = (2 * f1_score * ndcg) / (f1_score + ndcg)
            else:
                harmonic_mean = None

            # Harmonic mean of F-Beta and NDCG
            if f_beta_score and ndcg and (f_beta_score + ndcg) > 0:
                harmonic_mean_fbeta = (2 * f_beta_score * ndcg) / (f_beta_score + ndcg)
            else:
                harmonic_mean_fbeta = None
            
            record = {
                "Method": "Binary",
                "Model": "CatBoost Classifier",
                "Threshold": "Optimal Threshold",
                "Case": case_name,
                "Total Predictions": binary_metrics.get("total_predictions"),
                "True Positives": binary_metrics.get("true_positives"),
                "False Positives": binary_metrics.get("false_positives"),
                "False Negatives": binary_metrics.get("false_negatives"),
                "Precision": binary_metrics.get("precision"),
                "Recall": binary_metrics.get("recall"),
                "F1 Score": f1_score,
                "F-Beta Score": f_beta_score,
                "Accuracy": binary_metrics.get("accuracy"),
                "NDCG": ndcg,
                "Harmonic Mean (F1, NDCG)": harmonic_mean,
                "Harmonic Mean (F-Beta, NDCG)": harmonic_mean_fbeta,
            }
            records.append(record)
        else:
            print(f"Warning: Evaluation file not found at {file_path}")
    
    # Create DataFrame and save
    if records:
        df = pd.DataFrame(records)
        
        # Reorder columns to have Case after Model
        cols = df.columns.tolist()
        case_idx = cols.index("Case")
        cols.insert(3, cols.pop(case_idx))  # Move Case to position 3
        df = df[cols]
        
        df.to_csv(output_csv_path, index=False)
        print(f"Saved CatBoost classifier evaluation summary to: {output_csv_path}")
        return df
    else:
        print("No evaluation files found. No CSV generated.")
        return None
    
def merge_random_forests_classifier_evaluations(root_dir, output_csv_path):
    """
    Merge DEFAULT THRESHOLD evaluation metrics for random_forests classifier across T0, T1, and T1_predicted cases.
    
    Args:
        root_dir (str): Root folder containing the eval_results
        output_csv_path (str): Path to save the final CSV summary
    """
    records = []
    
    # Define the paths we need to check
    cases = [
        ('T0', 'binary_classification/random_forests_classifier'),
        ('T1', 'binary_classification/T1/random_forests_classifier'),
        ('T1_predicted', 'binary_classification/T1_predicted/random_forests_classifier')
    ]
    
    for case_name, rel_path in cases:
        case_dir = os.path.join(root_dir, rel_path)
        file_path = os.path.join(case_dir, 'evaluation_metrics_optimal_thrs.json')
        
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            binary_metrics = data.get("Evaluation Metrics", {}).get("binary_metrics", {})
            ranking_metrics = data.get("Evaluation Metrics", {}).get("ranking_metrics", {})
            
            f1_score = binary_metrics.get("f1_score")
            f_beta_score = binary_metrics.get("f_beta_score")
            ndcg = ranking_metrics.get("average_ndcg_with_probs")
            
            # Calculate harmonic mean of F1 and NDCG
            if f1_score and ndcg and (f1_score + ndcg) > 0:
                harmonic_mean = (2 * f1_score * ndcg) / (f1_score + ndcg)
            else:
                harmonic_mean = None

            # Harmonic mean of F-Beta and NDCG
            if f_beta_score and ndcg and (f_beta_score + ndcg) > 0:
                harmonic_mean_fbeta = (2 * f_beta_score * ndcg) / (f_beta_score + ndcg)
            else:
                harmonic_mean_fbeta = None
            
            record = {
                "Method": "Binary",
                "Model": "random_forests Classifier",
                "Threshold": "Optimal Threshold",
                "Case": case_name,
                "Total Predictions": binary_metrics.get("total_predictions"),
                "True Positives": binary_metrics.get("true_positives"),
                "False Positives": binary_metrics.get("false_positives"),
                "False Negatives": binary_metrics.get("false_negatives"),
                "Precision": binary_metrics.get("precision"),
                "Recall": binary_metrics.get("recall"),
                "F1 Score": f1_score,
                "F-Beta Score": f_beta_score,
                "Accuracy": binary_metrics.get("accuracy"),
                "NDCG": ndcg,
                "Harmonic Mean (F1, NDCG)": harmonic_mean,
                "Harmonic Mean (F-Beta, NDCG)": harmonic_mean_fbeta,
            }
            records.append(record)
        else:
            print(f"Warning: Evaluation file not found at {file_path}")
    
    # Create DataFrame and save
    if records:
        df = pd.DataFrame(records)
        
        # Reorder columns to have Case after Model
        cols = df.columns.tolist()
        case_idx = cols.index("Case")
        cols.insert(3, cols.pop(case_idx))  # Move Case to position 3
        df = df[cols]
        
        df.to_csv(output_csv_path, index=False)
        print(f"Saved random_forests classifier evaluation summary to: {output_csv_path}")
        return df
    else:
        print("No evaluation files found. No CSV generated.")
        return None
    

def merge_multi_nn_evaluations(root_dir, output_csv_path):
    """
    Merge DEFAULT THRESHOLD evaluation metrics for CatBoost classifier across T0, T1, and T1_predicted cases.
    
    Args:
        root_dir (str): Root folder containing the eval_results
        output_csv_path (str): Path to save the final CSV summary
    """
    records = []
    
    # Define the paths we need to check
    cases = [
        ('T0', 'multilabel/neural_network'),
        ('T1', 'multilabel/T1/neural_network'),
        ('T1_predicted', 'multilabel/T1_predicted/neural_network')
    ]
    
    for case_name, rel_path in cases:
        case_dir = os.path.join(root_dir, rel_path)
        file_path = os.path.join(case_dir, 'evaluation_metrics_optimal_thrs.json')
        
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            binary_metrics = data.get("Evaluation Metrics", {}).get("binary_metrics", {})
            ranking_metrics = data.get("Evaluation Metrics", {}).get("ranking_metrics", {})
            
            f1_score = binary_metrics.get("f1_score")
            f_beta_score = binary_metrics.get("f_beta_score")
            ndcg = ranking_metrics.get("average_ndcg_with_probs")
            
            # Calculate harmonic mean of F1 and NDCG
            if f1_score and ndcg and (f1_score + ndcg) > 0:
                harmonic_mean = (2 * f1_score * ndcg) / (f1_score + ndcg)
            else:
                harmonic_mean = None

            # Harmonic mean of F-Beta and NDCG
            if f_beta_score and ndcg and (f_beta_score + ndcg) > 0:
                harmonic_mean_fbeta = (2 * f_beta_score * ndcg) / (f_beta_score + ndcg)
            else:
                harmonic_mean_fbeta = None
            
            record = {
                "Method": "Multilabel",
                "Model": "Neural Network",
                "Threshold": "Optimal Threshold",
                "Case": case_name,
                "Total Predictions": binary_metrics.get("total_predictions"),
                "True Positives": binary_metrics.get("true_positives"),
                "False Positives": binary_metrics.get("false_positives"),
                "False Negatives": binary_metrics.get("false_negatives"),
                "Precision": binary_metrics.get("precision"),
                "Recall": binary_metrics.get("recall"),
                "F1 Score": f1_score,
                "F-Beta Score": f_beta_score,
                "Accuracy": binary_metrics.get("accuracy"),
                "NDCG": ndcg,
                "Harmonic Mean (F1, NDCG)": harmonic_mean,
                "Harmonic Mean (F-Beta, NDCG)": harmonic_mean_fbeta,
            }
            records.append(record)
        else:
            print(f"Warning: Evaluation file not found at {file_path}")
    
    # Create DataFrame and save
    if records:
        df = pd.DataFrame(records)
        
        # Reorder columns to have Case after Model
        cols = df.columns.tolist()
        case_idx = cols.index("Case")
        cols.insert(3, cols.pop(case_idx))  # Move Case to position 3
        df = df[cols]
        
        df.to_csv(output_csv_path, index=False)
        print(f"Saved Neural Network evaluation summary to: {output_csv_path}")
        return df
    else:
        print("No evaluation files found. No CSV generated.")
        return None
    

def combine_evaluation_results(output_path='src/recommendation/evaluation/eval_results/combined_catboost_regressor_evaluation_summary.csv'):
    # Define the files and their corresponding method labels
    eval_files = {
        'default_threshold': 'src/recommendation/evaluation/eval_results/catboost_regressor_default_threshold_summary.csv',
        'optimal_threshold_balance': 'src/recommendation/evaluation/eval_results/catboost_regressor_optimal_threshold_balance_summary.csv',
        'optimal_threshold_hrmnc_mean': 'src/recommendation/evaluation/eval_results/catboost_regressor_optimal_threshold_hrmnc_mean_summary.csv',
        'optimal_threshold_max_f1': 'src/recommendation/evaluation/eval_results/catboost_regressor_optimal_threshold_max_f1_summary.csv'
    }
    
    combined_df = pd.DataFrame()
    
    for method_name, file_path in eval_files.items():
        if os.path.exists(file_path):
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Add a column for the threshold method
            df['Threshold_Method'] = method_name
            
            # Concatenate with the combined dataframe
            combined_df = pd.concat([combined_df, df], ignore_index=True)
        else:
            print(f"Warning: File not found - {file_path}")
    
    # Reorder columns to have Threshold_Method early in the dataframe
    cols = combined_df.columns.tolist()
    cols.insert(3, cols.pop(cols.index('Threshold_Method')))
    combined_df = combined_df[cols]
    
    # Save the combined results
    combined_df.to_csv(output_path, index=False)
    print(f"Combined evaluation results saved to: {output_path}")
    
    return combined_df



if __name__ == "__main__":
    # extract_all_evaluation_metrics(
    #     root_dir='src/recommendation/evaluation/eval_results',
    #     output_csv_path='src/recommendation/evaluation/eval_results/all_evaluation_summary.csv',
    #     baseline_paths=[
    #         'src/recommendation/baseline/baseline_all_1/eval_results_grouped_catbased/evaluation_metrics.json'
    #     ]
    # )

    # merge_catboost_classifier_evaluations(
    #     root_dir='src/recommendation/evaluation/eval_results',
    #     output_csv_path='src/recommendation/evaluation/eval_results/catboost_classifier_summary.csv'
    # )

    # merge_random_forests_classifier_evaluations(
    #     root_dir='src/recommendation/evaluation/eval_results',
    #     output_csv_path='src/recommendation/evaluation/eval_results/random_forests_classifier_summary.csv'
    # )

    # merge_multi_nn_evaluations(
    #     root_dir='src/recommendation/evaluation/eval_results',
    #     output_csv_path='src/recommendation/evaluation/eval_results/neural_network_summary.csv'
    # )

    # combined_results = combine_evaluation_results()
    # print(combined_results.head())

    df = pd.read_csv('src/recommendation/evaluation/eval_results/all_evaluation_summary.csv')
    df = df[['Method', 'Model', 'Precision', 'Recall', 'F-Beta Score', 'NDCG', 'Harmonic Mean (F-Beta, NDCG)']]

    df.to_csv('src/recommendation/evaluation/eval_results/all_evaluation_summary_filtered.csv', index=False)