import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

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



# train
df_T0 = pd.read_csv('src/recommendation/data/T0/demog_ranking_grouped_catbased.csv')
df_T1 = pd.read_csv('src/recommendation/data/T1/demog_ranking_grouped_catbased.csv')
df_T1_predicted = pd.read_csv('src/recommendation/data/T1_predicted/demog_ranking_grouped_catbased.csv')

# test
# df_T0 = pd.read_csv('src/recommendation/data/T0/test_with_lifestyle.csv')
# df_T1 = pd.read_csv('src/recommendation/data/T1/test_with_lifestyle.csv')
# df_T1_predicted = pd.read_csv('src/recommendation/data/T1_predicted/test_with_lifestyle.csv')

plot_features(df_T0, df_T1, df_T1_predicted, save_dir='src/recommendation/data_prep/eda/feature_distributions', show_plots=False)
