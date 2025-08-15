import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from matplotlib import rcParams

features = ['Number of Children', 'Age', 'Gender', 'Education level', 'Marital status', 'Region', 'Occupation Group']

def plot_features(df_T0, df_T1, df_T1_predicted, save_dir='plots', show_plots=False):
    # Create directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for feat in features:
        plt.figure(figsize=(10, 6))
        
        # Determine if the feature is numeric
        is_numeric = pd.api.types.is_numeric_dtype(df_T0[feat])
        
        if is_numeric:
            # For numeric features, create bins
            min_val = min(df_T0[feat].min(), df_T1[feat].min(), df_T1_predicted[feat].min())
            max_val = max(df_T0[feat].max(), df_T1[feat].max(), df_T1_predicted[feat].max())
            bins = np.linspace(min_val, max_val, 10)
            
            # Calculate histograms
            hist_T0, _ = np.histogram(df_T0[feat], bins=bins)
            hist_T1, _ = np.histogram(df_T1[feat], bins=bins)
            hist_T1_pred, _ = np.histogram(df_T1_predicted[feat], bins=bins)
            
            # Create x-axis labels as range strings
            labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins)-1)]
            x = np.arange(len(labels))
            
            # Plot bars
            width = 0.25
            plt.bar(x - width, hist_T0, width, label='T0', alpha=0.7)
            plt.bar(x, hist_T1, width, label='T1', alpha=0.7)
            plt.bar(x + width, hist_T1_pred, width, label='T1 Predicted', alpha=0.7)
            
            plt.xticks(x, labels, rotation=45)
            
        else:
            # For categorical features, convert all values to strings
            all_categories = list(
                set(df_T0[feat].astype(str).unique()) | 
                set(df_T1[feat].astype(str).unique()) | 
                set(df_T1_predicted[feat].astype(str).unique())
            )
            
            # Sort as strings
            all_categories = sorted(all_categories, key=lambda x: str(x))
            
            # Count values for each category (converting to string for counting)
            counts_T0 = df_T0[feat].astype(str).value_counts().reindex(all_categories, fill_value=0)
            counts_T1 = df_T1[feat].astype(str).value_counts().reindex(all_categories, fill_value=0)
            counts_T1_pred = df_T1_predicted[feat].astype(str).value_counts().reindex(all_categories, fill_value=0)
            
            x = np.arange(len(all_categories))
            width = 0.25
            
            plt.bar(x - width, counts_T0, width, label='T0', alpha=0.7)
            plt.bar(x, counts_T1, width, label='T1', alpha=0.7)
            plt.bar(x + width, counts_T1_pred, width, label='T1 Predicted', alpha=0.7)
            
            plt.xticks(x, all_categories, rotation=45)
        
        plt.title(f'Distribution of {feat}')
        plt.xlabel(feat)
        plt.ylabel('Number of rows')
        plt.legend()
        plt.tight_layout()
        
        # Save the plot
        filename = os.path.join(save_dir, f'{feat.lower().replace(" ", "_")}_distribution.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f'Saved plot to {filename}')
        
        if show_plots:
            plt.show()
        else:
            plt.close()  # Close the figure to free memory



def plot_txn_category_comparison(df_T0, df_T1, df_T1_predicted, save_path=None):
    # Set style
    rcParams['figure.figsize'] = (12, 6)
    rcParams['axes.spines.top'] = False
    rcParams['axes.spines.right'] = False
    
    # Prepare data - extract and count TPs and FPs for each category
    def process_df(df, prefix):
        # Process True Positives
        tp = df['classes_of_true_positives'].str.split(',').explode().str.strip()
        tp_counts = tp.value_counts().to_dict()
        
        # Process False Positives
        fp = df['classes_of_false_positives'].str.split(',').explode().str.strip()
        fp_counts = fp.value_counts().to_dict()
        
        # Combine into a DataFrame
        all_categories = set(tp_counts.keys()).union(set(fp_counts.keys()))
        result = pd.DataFrame(index=sorted(all_categories))
        result[f'{prefix}_TP'] = result.index.map(tp_counts).fillna(0)
        result[f'{prefix}_FP'] = result.index.map(fp_counts).fillna(0)
        
        return result
    
    # Process all DataFrames
    df_T0_processed = process_df(df_T0, 'T0')
    df_T1_processed = process_df(df_T1, 'T1')
    df_T1_pred_processed = process_df(df_T1_predicted, 'T1_pred')
    
    # Combine all results
    combined = pd.concat([df_T0_processed, df_T1_processed, df_T1_pred_processed], axis=1)
    combined = combined.fillna(0)
    
    # Get categories and set up plot
    categories = combined.index
    x = np.arange(len(categories))
    width = 0.25  # Width of each group
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot TP and FP bars for each time period
    bar_handles = []
    bar_labels = []
    
    for i, prefix in enumerate(['T0', 'T1', 'T1_pred']):
        offset = width * i
        tp_col = f'{prefix}_TP'
        fp_col = f'{prefix}_FP'
        
        # Plot True Positives (darker color)
        tp_bars = ax.bar(x + offset, combined[tp_col], width, 
                        label=f'{prefix} True Positives',
                        color=plt.cm.tab10(i), alpha=0.8)
        bar_handles.append(tp_bars)
        bar_labels.append(f'{prefix} True Positives')
        
        # Only plot FP if there are any
        if combined[fp_col].sum() > 0:
            fp_bars = ax.bar(x + offset, combined[fp_col], width, 
                            bottom=combined[tp_col],
                            label=f'{prefix} False Positives',
                            color=plt.cm.tab10(i), alpha=0.4)
            bar_handles.append(fp_bars)
            bar_labels.append(f'{prefix} False Positives')
    
    # Customize the plot
    ax.set_xticks(x + width)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.set_ylabel('Count')
    ax.set_title('True Positives and False Positives by Transaction Category')
    
    # Create legend using only the handles we actually created
    ax.legend(bar_handles, bar_labels, 
              title='Category Type', bbox_to_anchor=(1.05, 1), 
              loc='upper left')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()

### Plot features ###
# train
# df_T0 = pd.read_csv('src/recommendation/data/T0/demog_ranking_grouped_catbased.csv')
# df_T1 = pd.read_csv('src/recommendation/data/T1/demog_ranking_grouped_catbased.csv')
# df_T1_predicted = pd.read_csv('src/recommendation/data/T1_predicted/demog_ranking_grouped_catbased.csv')

# df_T0 = pd.read_csv('src/recommendation/data/T0/demog_ranking_grouped_catbased_no_norm.csv')
# df_T1 = pd.read_csv('src/recommendation/data/T1/demog_ranking_grouped_catbased_no_norm.csv')
# df_T1_predicted = pd.read_csv('src/recommendation/data/T1_predicted/demog_ranking_grouped_catbased_no_norm.csv')

# df_T0 = pd.read_csv('src/recommendation/data/T0/demog_grouped_catbased.csv')
# df_T1 = pd.read_csv('src/recommendation/data/T1/demog_grouped_catbased.csv')
# df_T1_predicted = pd.read_csv('src/recommendation/data/T1_predicted/demog_grouped_catbased.csv')

# test
# df_T0 = pd.read_csv('src/recommendation/data/T0/test_with_lifestyle.csv')
# df_T1 = pd.read_csv('src/recommendation/data/T1/test_with_lifestyle.csv')
# df_T1_predicted = pd.read_csv('src/recommendation/data/T1_predicted/test_with_lifestyle.csv')

# plot_features(df_T0, df_T1, df_T1_predicted, save_dir='src/recommendation/data_prep/eda/feature_distributions', show_plots=False)
# plot_features(df_T0, df_T1, df_T1_predicted, save_dir='src/recommendation/data_prep/eda/feature_distributions_no_norm', show_plots=False)
# plot_features(df_T0, df_T1, df_T1_predicted, save_dir='src/recommendation/data_prep/eda/feature_distributions_no_rank', show_plots=False)
# plot_features(df_T0, df_T1, df_T1_predicted, save_dir='src/recommendation/data_prep/eda/feature_distributions_test', show_plots=False)

### Plot Txn Categories ###
df_T0 = pd.read_csv('src/recommendation/evaluation/eval_results/binary_classification/random_forests_classifier/detailed_evaluation_results_optimal_thrs.csv')
df_T1 = pd.read_csv('src/recommendation/evaluation/eval_results/binary_classification/T1/random_forests_classifier/detailed_evaluation_results_optimal_thrs.csv')
df_T1_predicted = pd.read_csv('src/recommendation/evaluation/eval_results/binary_classification/T1_predicted/random_forests_classifier/detailed_evaluation_results_optimal_thrs.csv')

plot_txn_category_comparison(
    df_T0, 
    df_T1, 
    df_T1_predicted,
    save_path='src/recommendation/data_prep/eda/transaction_category_comparison.png'
)