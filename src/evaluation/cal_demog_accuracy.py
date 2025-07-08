import pandas as pd
import numpy as np
import os
from tabulate import tabulate
from IPython.display import display, Markdown
from sklearn.metrics import f1_score


# all_predictions = pd.read_csv('src/prediction/pred_results/predictions_indiv_multi.csv')
# all_predictions = pd.read_csv('src/prediction/pred_results/predictions_indiv_single.csv')

# all_predictions = pd.read_csv('src/prediction/pred_results/predictions_cluster_multi_no_ca.csv')
# all_predictions = pd.read_csv('src/prediction/pred_results/predictions_cluster_single_ca.csv')
# all_predictions = pd.read_csv('src/prediction/pred_results/predictions_cluster_multi_ca.csv')
all_predictions = pd.read_csv('src/prediction/pred_results/predictions_cluster_multi_ca_const.csv')

# all_predictions = pd.read_csv('src/prediction/pred_results/predictions_rag_single.csv')
# all_predictions = pd.read_csv('src/prediction/pred_results/predictions_rag_multi.csv')


test_actual = pd.read_csv('src/data/T1/test_T1_actual_v3.csv')
output_dir = "src/evaluation/error_analysis/error_analysis_cluster_multi_ca_const"

def calculate_demographic_accuracy(predictions_df, actual_df, output_dir, is_cluster_method=False):    
    os.makedirs(output_dir, exist_ok=True)
    
    merged_df = pd.merge(
        predictions_df,
        actual_df,
        on='CUST_ID',
        suffixes=('_pred', '_actual')
    )
    
    demographic_fields = {
        'PRED_education': 'Education level_actual',
        'PRED_marital_status': 'Marital status_actual',
        'PRED_occupation': 'Occupation Group_actual',
        'PRED_num_children': 'Number of Children_actual',
        'PRED_region': 'Region_actual'
    }
    
    results = {
        'accuracy_metrics': {},
        'f1_scores': {},
        'analysis_files': {},
        'unmatched_data': {},
        'excluded_unknown': {}
    }
    
    # 1. Generate accuracy metrics (excluding 'Unknown' actual values)
    accuracy_metrics = []
    f1_scores = []

    for pred_col, actual_col in demographic_fields.items():
        field_name = pred_col.replace('PRED_', '')
        
        # Filter out rows where actual value is 'Unknown'
        filtered_df = merged_df[merged_df[actual_col] != 'Unknown'].copy()
        results['excluded_unknown'][field_name] = len(merged_df) - len(filtered_df)
        
        # Basic accuracy metrics
        correct = (filtered_df[pred_col] == filtered_df[actual_col]).sum()
        total = len(filtered_df)
        accuracy = correct / total if total > 0 else 0
        
        # Calculate F1 scores (macro and micro)
        y_true = filtered_df[actual_col]
        y_pred = filtered_df[pred_col]
        
        # Get all unique classes in both true and pred
        classes = np.unique(np.concatenate([y_true.unique(), y_pred.unique()]))
        print(classes)
        
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
        
        accuracy_metrics.append({
            'Field': field_name,
            'Correct': correct,
            'Total': total,
            'Accuracy': f"{accuracy:.1%}",
            'Error Rate': f"{(1-accuracy):.1%}",
            'Excluded (Unknown)': results['excluded_unknown'][field_name]
        })
        
        f1_scores.append({
            'Field': field_name,
            'F1 Macro': f"{f1_macro:.3f}",
            'F1 Micro': f"{f1_micro:.3f}",
            'Num Classes': len(classes)
        })
    
    accuracy_df = pd.DataFrame(accuracy_metrics)
    f1_df = pd.DataFrame(f1_scores)
    
    results['accuracy_metrics'] = accuracy_df.to_dict('records')
    results['f1_scores'] = f1_df.to_dict('records')
    
    # Save accuracy metrics
    accuracy_path = os.path.join(output_dir, "accuracy_metrics.csv")
    accuracy_df.to_csv(accuracy_path, index=False)
    results['analysis_files']['accuracy_metrics'] = accuracy_path
    
    # Save F1 scores
    f1_path = os.path.join(output_dir, "f1_scores.csv")
    f1_df.to_csv(f1_path, index=False)
    results['analysis_files']['f1_scores'] = f1_path
    
    # 2. Generate error analysis reports for each field
    for pred_col, actual_col in demographic_fields.items():
        field_name = pred_col.replace('PRED_', '')
        
        # Filter unmatched cases (excluding where actual is 'Unknown')
        unmatched = merged_df[
            (merged_df[pred_col] != merged_df[actual_col]) & 
            (merged_df[actual_col] != 'Unknown')
        ].copy()
        
        if len(unmatched) > 0:
            # Prepare error analysis dataframe
            if is_cluster_method:
                error_df = unmatched[[
                    'CUST_ID', 'cluster',
                    pred_col, actual_col,
                    f'CONFIDENCE_{field_name}',
                    f'REASONING_{field_name}'
                ]].rename(columns={
                    pred_col: 'Predicted',
                    actual_col: 'Actual',
                    f'CONFIDENCE_{field_name}': 'Confidence',
                    f'REASONING_{field_name}': 'Reasoning'
                })
            else:
                error_df = unmatched[[
                    'CUST_ID', 
                    pred_col, actual_col,
                    f'CONFIDENCE_{field_name}',
                    f'REASONING_{field_name}'
                ]].rename(columns={
                    pred_col: 'Predicted',
                    actual_col: 'Actual',
                    f'CONFIDENCE_{field_name}': 'Confidence',
                    f'REASONING_{field_name}': 'Reasoning'
                })
            
            error_df['Field'] = field_name
            results['unmatched_data'][field_name] = error_df
            
            # Save to Excel and CSV
            field_path = os.path.join(output_dir, f"errors_{field_name}.xlsx")
            with pd.ExcelWriter(field_path) as writer:
                error_df.to_excel(writer, index=False, sheet_name='Errors')
                
                # Add summary stats
                error_summary = error_df['Predicted'].value_counts().reset_index()
                error_summary.columns = ['Predicted Value', 'Count']
                error_summary['Percentage'] = error_summary['Count'] / len(error_df)
                error_summary.to_excel(writer, index=False, sheet_name='Summary')
            
            results['analysis_files'][field_name] = field_path
    
    # 3. Create combined error report
    if results['unmatched_data']:
        all_errors = pd.concat(results['unmatched_data'].values())
        combined_path = os.path.join(output_dir, "all_errors.xlsx")
        all_errors.to_excel(combined_path, index=False)
        results['analysis_files']['combined_errors'] = combined_path
    
    # 4. Generate visual reports (Markdown)
    md_report = []
    
    # Accuracy summary
    md_report.append("## Demographic Prediction Accuracy Summary\n")
    md_report.append("*(Excluding cases where actual value is 'Unknown')*\n")
    md_report.append(tabulate(accuracy_df, headers='keys', tablefmt='pipe', showindex=False))
    
    # F1 scores summary
    md_report.append("\n\n## F1 Scores Evaluation\n")
    md_report.append(tabulate(f1_df, headers='keys', tablefmt='pipe', showindex=False))
    md_report.append("\n- **F1 Macro**: Treats all classes equally (good for imbalanced data)")
    md_report.append("\n- **F1 Micro**: Aggregates across all classes (weights by class size)")

    # Error analysis by field
    md_report.append("\n\n## Error Analysis by Field\n")
    for field_name, error_df in results['unmatched_data'].items():
        md_report.append(f"\n### {field_name.title()} ({len(error_df)} errors)\n")
        
        # Top error patterns
        top_errors = error_df['Predicted'].value_counts().nlargest(3)
        md_report.append(f"**Most common prediction errors:**\n")
        for pred, count in top_errors.items():
            md_report.append(f"- Predicted '{pred}': {count} cases")
        
        # Example cases
        md_report.append("\n**Example cases:**")
        sample = error_df.head(3)
        for _, row in sample.iterrows():
            if is_cluster_method:
                md_report.append(
                    f"\n- **Customer {row['CUST_ID']}** (Cluster {row['cluster']}, Confidence: {row['Confidence']:.0%})"
                    f"\n  - Predicted: `{row['Predicted']}`"
                    f"\n  - Actual: `{row['Actual']}`"
                    f"\n  - Reasoning: {row['Reasoning']}"
                )
            else:
                md_report.append(
                    f"\n- **Customer {row['CUST_ID']}** (Confidence: {row['Confidence']:.0%})"
                    f"\n  - Predicted: `{row['Predicted']}`"
                    f"\n  - Actual: `{row['Actual']}`"
                    f"\n  - Reasoning: {row['Reasoning']}"
                )
    
    # Save and display Markdown report
    md_path = os.path.join(output_dir, "error_analysis.md")
    with open(md_path, 'w') as f:
        f.write("\n".join(md_report))
    results['analysis_files']['markdown_report'] = md_path
    
    # Display in notebook if available
    try:
        display(Markdown("\n".join(md_report)))
    except:
        print("Markdown report generated. View in Jupyter Notebook for better formatting.")
    
    return results



results = calculate_demographic_accuracy(all_predictions, test_actual, output_dir, is_cluster_method=True)

print("\nAnalysis files created:")
for name, path in results['analysis_files'].items():
    print(f"- {name}: {path}")