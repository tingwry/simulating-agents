import os
import json
import pandas as pd

def extract_all_evaluation_metrics(root_dir, output_csv_path):
    """
    Traverse the evaluation results folder and compile both default and optimal threshold 
    evaluation metrics into a single CSV file, including harmonic mean of F1 Score and NDCG.
    
    Args:
        root_dir (str): Root folder containing the eval_results.
        output_csv_path (str): Path to save the final CSV summary.
    """
    records = []

    for dirpath, _, filenames in os.walk(root_dir):
        for fname in ["evaluation_metrics.json", "evaluation_metrics_optimal_thrs.json"]:
            if fname in filenames:
                json_path = os.path.join(dirpath, fname)
                with open(json_path, 'r') as f:
                    data = json.load(f)

                # Relative path from root_dir to current file location
                rel_path = os.path.relpath(dirpath, root_dir)
                path_parts = rel_path.split(os.sep)

                # Handle normal case (e.g., binary_classification/catboost_classifier)
                if len(path_parts) >= 2:
                    task_type = path_parts[0].replace("_", " ").title()
                    model_name = path_parts[1].replace("_", " ").title()
                # Special case: task folders like 'rl' with no model subfolders
                elif len(path_parts) == 1:
                    task_type = path_parts[0].replace("_", " ").title()
                    model_name = '-'  # or ''
                else:
                    continue  # Skip malformed paths

                threshold_type = "Optimal Threshold" if "optimal_thrs" in fname else "Default Threshold"

                binary_metrics = data.get("Evaluation Metrics", {}).get("binary_metrics", {})
                ranking_metrics = data.get("Evaluation Metrics", {}).get("ranking_metrics", {})

                f1_score = binary_metrics.get("f1_score")
                ndcg = ranking_metrics.get("average_ndcg_with_probs")

                # Harmonic mean of F1 and NDCG
                if f1_score and ndcg and (f1_score + ndcg) > 0:
                    harmonic_mean = (2 * f1_score * ndcg) / (f1_score + ndcg)
                else:
                    harmonic_mean = None

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
                    "Accuracy": binary_metrics.get("accuracy"),
                    "NDCG": ndcg,
                    "Harmonic Mean (F1, NDCG)": harmonic_mean,
                }
                records.append(record)

    # Save to CSV
    df = pd.DataFrame(records)
    df.to_csv(output_csv_path, index=False)
    print(f"Saved evaluation summary to: {output_csv_path}")


if __name__ == "__main__":
    extract_all_evaluation_metrics(
        root_dir='src/recommendation/evaluation/eval_results',
        output_csv_path='src/recommendation/evaluation/eval_results/all_evaluation_summary.csv'
    )
