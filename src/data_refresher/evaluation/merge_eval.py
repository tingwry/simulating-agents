import pandas as pd

def process_data(acc_df, f1_df, experiment_name, constraints):
    merged = pd.merge(acc_df, f1_df, on='Field')
    
    merged['experiment'] = experiment_name
    merged['constraints'] = constraints
    
    merged = merged[['Field', 'experiment', 'constraints', 
                     'Correct', 'Total', 'Accuracy', 
                     'F1 Macro', 'F1 Micro', 'Excluded (Unknown)']]
    
    merged.columns = ['field', 'experiment', 'constraints', 
                      'correct', 'total', 'accuracy', 
                      'f1 macro', 'f1 micro', 'excluded (unknown)']
    
    return merged


if __name__ == "__main__":
    # acc
    acc_indiv_single = pd.read_csv('src/evaluation/error_analysis/error_analysis_indiv_single/accuracy_metrics.csv')
    acc_indiv_multi = pd.read_csv('src/evaluation/error_analysis/error_analysis_indiv_multi/accuracy_metrics.csv')
    acc_clus_single_ca = pd.read_csv('src/evaluation/error_analysis/error_analysis_cluster_single_ca/accuracy_metrics.csv')
    acc_clus_multi_ca = pd.read_csv('src/evaluation/error_analysis/error_analysis_cluster_multi_ca/accuracy_metrics.csv')
    acc_clus_multi_ca_const = pd.read_csv('src/evaluation/error_analysis/error_analysis_cluster_multi_ca_const/accuracy_metrics.csv')
    acc_rag_single = pd.read_csv('src/evaluation/error_analysis/error_analysis_rag_single/accuracy_metrics.csv')
    acc_rag_multi = pd.read_csv('src/evaluation/error_analysis/error_analysis_rag_multi/accuracy_metrics.csv')


    # f1
    f1_indiv_single = pd.read_csv('src/evaluation/error_analysis/error_analysis_indiv_single/f1_scores.csv')
    f1_indiv_multi = pd.read_csv('src/evaluation/error_analysis/error_analysis_indiv_multi/f1_scores.csv')
    f1_clus_single_ca = pd.read_csv('src/evaluation/error_analysis/error_analysis_cluster_single_ca/f1_scores.csv')
    f1_clus_multi_ca = pd.read_csv('src/evaluation/error_analysis/error_analysis_cluster_multi_ca/f1_scores.csv')
    f1_clus_multi_ca_const = pd.read_csv('src/evaluation/error_analysis/error_analysis_cluster_multi_ca_const/f1_scores.csv')
    f1_rag_single = pd.read_csv('src/evaluation/error_analysis/error_analysis_rag_single/f1_scores.csv')
    f1_rag_multi = pd.read_csv('src/evaluation/error_analysis/error_analysis_rag_multi/f1_scores.csv')


    indiv_single = process_data(acc_indiv_single, f1_indiv_single, 'indiv_single', 'no')
    indiv_multi = process_data(acc_indiv_multi, f1_indiv_multi, 'indiv_multi', 'no')
    clus_single_ca = process_data(acc_clus_single_ca, f1_clus_single_ca, 'clus_single', 'no')
    clus_multi_ca = process_data(acc_clus_multi_ca, f1_clus_multi_ca, 'clus_multi', 'no')
    clus_multi_ca_const = process_data(acc_clus_multi_ca_const, f1_clus_multi_ca_const, 'clus_multi', 'yes')
    rag_single = process_data(acc_rag_single, f1_rag_single, 'rag_single', 'no')
    rag_multi = process_data(acc_rag_multi, f1_rag_multi, 'rag_multi', 'no')


    combined_df = pd.concat([
        indiv_single,
        indiv_multi,
        clus_single_ca,
        clus_multi_ca,
        clus_multi_ca_const,
        rag_single,
        rag_multi
    ], ignore_index=True)


    if isinstance(combined_df['accuracy'].iloc[0], str):
        combined_df['accuracy'] = combined_df['accuracy'].str.rstrip('%').astype('float') / 100.0

    combined_df = combined_df.sort_values(['field', 'experiment'])

    combined_df = combined_df.reset_index(drop=True)

    print(combined_df)

    combined_df.to_csv('src/evaluation/error_analysis/combined_metrics.csv', index=False)