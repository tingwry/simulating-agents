import pandas as pd


all_predictions = pd.read_csv('src/demog_T1_pred/all_predictions_v3.csv')
test_actual = pd.read_csv('src/test_T1_actual/test_T1_actual.csv')



def calculate_demographic_accuracy(predictions_df, actual_df, output_dir="src/error_analysis_v3"):
    """
    Compare predicted vs actual demographics and generate error analysis reports in multiple formats.
    
    Args:
        predictions_df: DataFrame with predicted demographics
        actual_df: DataFrame with actual demographic values
        output_dir: Directory to save analysis files
        
    Returns:
        dict: Dictionary containing accuracy metrics and file paths to analysis reports
    """
    import os
    from tabulate import tabulate
    from IPython.display import display, Markdown
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Merge dataframes
    merged_df = pd.merge(
        predictions_df,
        actual_df,
        on='CUST_ID',
        suffixes=('_pred', '_actual')
    )
    
    # Define demographic fields to compare
    demographic_fields = {
        'PRED_education': 'Education level_actual',
        'PRED_marital_status': 'Marital status_actual',
        'PRED_occupation': 'Occupation Group_actual',
        'PRED_num_children': 'Number of Children_actual',
        'PRED_region': 'Region_actual'
    }
    
    # Initialize results
    results = {
        'accuracy_metrics': {},
        'analysis_files': {},
        'unmatched_data': {}
    }
    
    # 1. Generate accuracy metrics
    accuracy_metrics = []
    for pred_col, actual_col in demographic_fields.items():
        field_name = pred_col.replace('PRED_', '')
        correct = (merged_df[pred_col] == merged_df[actual_col]).sum()
        total = len(merged_df)
        accuracy = correct / total
        
        accuracy_metrics.append({
            'Field': field_name,
            'Correct': correct,
            'Total': total,
            'Accuracy': f"{accuracy:.1%}",
            'Error Rate': f"{(1-accuracy):.1%}"
        })
    
    accuracy_df = pd.DataFrame(accuracy_metrics)
    results['accuracy_metrics'] = accuracy_df.to_dict('records')
    
    # Save accuracy metrics
    accuracy_path = os.path.join(output_dir, "accuracy_metrics.csv")
    accuracy_df.to_csv(accuracy_path, index=False)
    results['analysis_files']['accuracy_metrics'] = accuracy_path
    
    # 2. Generate error analysis reports for each field
    for pred_col, actual_col in demographic_fields.items():
        field_name = pred_col.replace('PRED_', '')
        
        # Filter unmatched cases
        unmatched = merged_df[merged_df[pred_col] != merged_df[actual_col]].copy()
        if len(unmatched) > 0:
            # Prepare error analysis dataframe
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
    all_errors = pd.concat(results['unmatched_data'].values())
    combined_path = os.path.join(output_dir, "all_errors.xlsx")
    all_errors.to_excel(combined_path, index=False)
    results['analysis_files']['combined_errors'] = combined_path
    
    # 4. Generate visual reports (Markdown)
    md_report = []
    
    # Accuracy summary
    md_report.append("## Demographic Prediction Accuracy Summary\n")
    md_report.append(tabulate(accuracy_df, headers='keys', tablefmt='pipe', showindex=False))
    
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
            md_report.append(
                f"\n- **Customer {row['CUST_ID']}** (Cluster {row['cluster']}, Confidence: {row['Confidence']:.0%})"
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

# Example usage
results = calculate_demographic_accuracy(all_predictions, test_actual)

print("\nAnalysis files created:")
for name, path in results['analysis_files'].items():
    print(f"- {name}: {path}")