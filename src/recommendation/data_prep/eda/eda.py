import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from matplotlib import rcParams
import json
from scipy.stats import hmean

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

def merge_train_test(train, test):
    return pd.concat([train, test], axis=0, ignore_index=True)


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter

# Set style for better-looking plots
plt.style.use('default')
sns.set_palette("husl")

def create_demographics_dashboard(csv_path, save_path="demographics_dashboard.png"):
    """
    Create a comprehensive demographics dashboard from customer CSV
    
    Args:
        csv_path (str): Path to the customer CSV file
        save_path (str): Path to save the PNG file
    """
    
    # Load the data
    df = pd.read_csv(csv_path)
    total_customers = len(df)
    
    # Create the figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    # fig.suptitle(f'Customer Demographics Overview (N={total_customers})', 
    #              fontsize=20, fontweight='bold', y=0.95)
    
    # Define colors
    # colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
    colors = ['#0f3741', '#0f3741', '#0f3741', '#3e809d', '#0f3741', '#0f3741']
    
    # 1. Age Distribution (Top Left)
    ax1 = axes[0, 0]
    ax1.hist(df['Age'], bins=15, color=colors[0], alpha=1, edgecolor='black', linewidth=0.5)
    ax1.set_title('Age', fontsize=18, fontweight='bold')
    ax1.set_xlabel('Age', fontsize=14)
    ax1.set_ylabel('Count', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Add statistics
    mean_age = df['Age'].mean()
    ax1.axvline(mean_age, color='black', linestyle='--', linewidth=2, label=f'Mean: {mean_age:.1f}')
    ax1.legend()
    
    # 2. Education Level (Top Center)
    ax2 = axes[0, 1]
    education_counts = df['Education level'].value_counts()
    y_pos = np.arange(len(education_counts))
    bars = ax2.barh(y_pos, education_counts.values, color=colors[1], alpha=1)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(education_counts.index, fontsize=12)
    ax2.set_title('Education Level', fontsize=18, fontweight='bold')
    ax2.set_xlabel('Count', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Add count labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax2.text(width + 1, bar.get_y() + bar.get_height()/2, 
                f'{int(width)}', ha='left', va='center', fontsize=12)
    
    # 3. Gender Distribution (Top Right)
    ax3 = axes[0, 2]
    gender_counts = df['Gender'].value_counts()
    wedges, texts, autotexts = ax3.pie(gender_counts.values, labels=gender_counts.index, 
                                       autopct='%1.1f%%', colors=colors[2:4], 
                                       startangle=90, textprops={'fontsize': 14})
    ax3.set_title('Gender', fontsize=18, fontweight='bold')
    
    # Make percentage text bold
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    # 4. Marital Status (Bottom Left)
    ax4 = axes[1, 0]
    marital_counts = df['Marital status'].value_counts()
    bars = ax4.bar(range(len(marital_counts)), marital_counts.values, 
                   color=colors[4], alpha=1, edgecolor='black', linewidth=0.5)
    ax4.set_xticks(range(len(marital_counts)))
    ax4.set_xticklabels(marital_counts.index, rotation=45, ha='right', fontsize=12)
    ax4.set_title('Marital Status', fontsize=18, fontweight='bold')
    ax4.set_ylabel('Count', fontsize=14)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add count labels on bars
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{int(height)}', ha='center', va='bottom', fontsize=12)
    
    # 5. Occupation Group (Bottom Center)
    ax5 = axes[1, 1]
    occupation_counts = df['Occupation Group'].value_counts()
    y_pos = np.arange(len(occupation_counts))
    bars = ax5.barh(y_pos, occupation_counts.values, color=colors[5], alpha=1)
    ax5.set_yticks(y_pos)
    ax5.set_yticklabels(occupation_counts.index, fontsize=12)
    ax5.set_title('Occupation Group', fontsize=18, fontweight='bold')
    ax5.set_xlabel('Count', fontsize=14)
    ax5.grid(True, alpha=0.3, axis='x')
    
    # Add count labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax5.text(width + 1, bar.get_y() + bar.get_height()/2, 
                f'{int(width)}', ha='left', va='center', fontsize=12)
    
    # 6. Region Distribution (Bottom Right)
    ax6 = axes[1, 2]
    region_counts = df['Region'].value_counts()
    bars = ax6.bar(range(len(region_counts)), region_counts.values, 
                   color=colors[0], alpha=1, 
                   edgecolor='black', linewidth=0.5)
    ax6.set_xticks(range(len(region_counts)))
    ax6.set_xticklabels(region_counts.index, rotation=45, ha='right', fontsize=12)
    ax6.set_title('Region', fontsize=18, fontweight='bold')
    ax6.set_ylabel('Count', fontsize=14)
    ax6.grid(True, alpha=0.3, axis='y')
    
    # Add count labels on bars
    for bar in bars:
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{int(height)}', ha='center', va='bottom', fontsize=12)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.3, wspace=0.5)
    
    # Save the figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Demographics dashboard saved as: {save_path}")
    
    # Display the plot
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*50)
    print("DEMOGRAPHICS SUMMARY")
    print("="*50)
    print(f"Total Customers: {total_customers}")
    print(f"Age Range: {df['Age'].min():.0f} - {df['Age'].max():.0f} years")
    print(f"Average Age: {df['Age'].mean():.1f} years")
    print(f"Average Children: {df['Number of Children'].mean():.1f}")
    
    print(f"\nGender Split:")
    for gender, count in df['Gender'].value_counts().items():
        print(f"  {gender}: {count} ({count/total_customers*100:.1f}%)")
    
    print(f"\nTop Education Level: {df['Education level'].mode().iloc[0]}")
    print(f"Top Occupation: {df['Occupation Group'].mode().iloc[0]}")
    print(f"Top Region: {df['Region'].mode().iloc[0]}")
    
    return fig






def create_harmonic_mean_chart(rf_file, zeroshot_file, rag_file, output_file='harmonic_mean_comparison.png'):
    """
    Create a bar chart comparing harmonic mean of NDCG and F-beta scores
    
    Parameters:
    rf_file (str): Path to Random Forest JSON file
    zeroshot_file (str): Path to Zero-shot LLM JSON file  
    rag_file (str): Path to RAG LLM JSON file
    output_file (str): Output PNG filename
    """
    
    # Load JSON files
    with open(rf_file, 'r') as f:
        rf_data = json.load(f)
    with open(zeroshot_file, 'r') as f:
        zeroshot_data = json.load(f)
    with open(rag_file, 'r') as f:
        rag_data = json.load(f)
    
    # Extract metrics and calculate harmonic means
    approaches = []
    harmonic_means = []
    
    # Random Forest
    rf_fbeta = rf_data['Evaluation Metrics']['binary_metrics']['f_beta_score']
    rf_ndcg = rf_data['Evaluation Metrics']['ranking_metrics']['average_ndcg_with_probs']
    rf_harmonic = hmean([rf_fbeta, rf_ndcg])
    approaches.append('Random Forest')
    harmonic_means.append(rf_harmonic)
    
    # Zero-shot LLM
    zs_fbeta = zeroshot_data['Evaluation Metrics']['binary_metrics']['f_beta_score']
    zs_ndcg = zeroshot_data['Evaluation Metrics']['ranking_metrics']['average_ndcg_with_probs']
    zs_harmonic = hmean([zs_fbeta, zs_ndcg])
    approaches.append('Zero-shot LLM')
    harmonic_means.append(zs_harmonic)
    
    # RAG LLM
    rag_fbeta = rag_data['Evaluation Metrics']['binary_metrics']['f_beta_score']
    rag_ndcg = rag_data['Evaluation Metrics']['ranking_metrics']['average_ndcg_with_probs']
    rag_harmonic = hmean([rag_fbeta, rag_ndcg])
    approaches.append('RAG LLM')
    harmonic_means.append(rag_harmonic)
    
    # Color scheme matching your palette
    colors = ['#8B5A8C', '#5B9BD5', '#2E8B8B']  # Purple, Blue, Teal
    
    # Create the bar chart
    plt.figure(figsize=(10, 6))
    bars = plt.bar(approaches, harmonic_means, color=colors, alpha=0.8, edgecolor='white', linewidth=1.5)
    
    # Customize the chart
    plt.title('Harmonic Mean of NDCG and F-beta Scores by Approach', 
              fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Harmonic Mean Score', fontsize=12, fontweight='bold')
    plt.xlabel('Approach', fontsize=12, fontweight='bold')
    
    # Add value labels on bars
    for bar, value in zip(bars, harmonic_means):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # Customize appearance
    plt.ylim(0, max(harmonic_means) * 1.15)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.gca().set_axisbelow(True)
    
    # Style the axes
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_color('#666666')
    plt.gca().spines['bottom'].set_color('#666666')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()
    
    # Print results
    print("\nHarmonic Mean Results:")
    print("-" * 30)
    for approach, score in zip(approaches, harmonic_means):
        print(f"{approach:<15}: {score:.4f}")
    
    return harmonic_means

def create_t0_t1_comparison_chart(t0_file, t1_file, output_file='t0_t1_harmonic_comparison.png'):
    """
    Create a bar chart comparing harmonic mean of NDCG and F-beta scores for T0 vs T1
    
    Parameters:
    t0_file (str): Path to Direct T0 JSON file
    t1_file (str): Path to Refreshed T1 JSON file
    output_file (str): Output PNG filename
    """
    
    # Load JSON files
    with open(t0_file, 'r') as f:
        t0_data = json.load(f)
    with open(t1_file, 'r') as f:
        t1_data = json.load(f)
    
    # Extract metrics and calculate harmonic means
    approaches = []
    harmonic_means = []
    
    # Direct T0
    t0_fbeta = t0_data['Evaluation Metrics']['binary_metrics']['f_beta_score']
    t0_ndcg = t0_data['Evaluation Metrics']['ranking_metrics']['average_ndcg_with_probs']
    t0_harmonic = hmean([t0_fbeta, t0_ndcg])
    approaches.append('Direct T0')
    harmonic_means.append(t0_harmonic)
    
    # Refreshed T1
    t1_fbeta = t1_data['Evaluation Metrics']['binary_metrics']['f_beta_score']
    t1_ndcg = t1_data['Evaluation Metrics']['ranking_metrics']['average_ndcg_with_probs']
    t1_harmonic = hmean([t1_fbeta, t1_ndcg])
    approaches.append('Refreshed T1')
    harmonic_means.append(t1_harmonic)
    
    # Color scheme - using two contrasting colors from your palette
    colors = ['#B19CD9', '#7FB3B3']  # Light purple for T0, Light teal for T1
    
    # Create the horizontal bar chart
    plt.figure(figsize=(8, 5))
    bars = plt.barh(approaches, harmonic_means, color=colors, alpha=0.8, edgecolor='white', linewidth=1.5)
    
    # Customize the chart
    plt.title('Harmonic Mean of NDCG and F-beta Scores: T0 vs T1', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Harmonic Mean Score', fontsize=12, fontweight='bold')
    plt.ylabel('Demographics Data', fontsize=12, fontweight='bold')
    
    # Add value labels on bars
    for bar, value in zip(bars, harmonic_means):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{value:.4f}', ha='left', va='center', fontweight='bold')
    
    # Customize appearance
    plt.xlim(0, max(harmonic_means) * 1.15)
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    plt.gca().set_axisbelow(True)
    
    # Style the axes
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_color('#666666')
    plt.gca().spines['bottom'].set_color('#666666')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()
    
    # Print results
    print("\nT0 vs T1 Harmonic Mean Results:")
    print("-" * 35)
    for approach, score in zip(approaches, harmonic_means):
        print(f"{approach:<12}: {score:.4f}")
    
    # Calculate and show improvement
    improvement = ((t1_harmonic - t0_harmonic) / t0_harmonic) * 100
    print(f"\nImprovement from T0 to T1: {improvement:+.2f}%")
    
    return harmonic_means


if __name__ == "__main__":
    harmonic_means = create_harmonic_mean_chart(
        'src/recommendation/evaluation/eval_results/binary_classification/random_forests_classifier/evaluation_metrics_optimal_thrs.json', 
        'src/recommendation/evaluation/eval_results/llm/evaluation_metrics_optimal_thrs.json', 
        'src/recommendation/evaluation/eval_results/llm/rag/evaluation_metrics_optimal_thrs.json', 
        output_file='src/recommendation/data_prep/eda/harmonic_mean_comparison.png'
        )
    

    t0_t1_means = create_t0_t1_comparison_chart(
        'src/recommendation/evaluation/eval_results/binary_classification/random_forests_classifier/evaluation_metrics_optimal_thrs.json', 
        'src/recommendation/evaluation/eval_results/binary_classification/T1_predicted/random_forests_classifier_single/evaluation_metrics_optimal_thrs.json',
        output_file='src/recommendation/data_prep/eda/t0_t1_harmonic_comparison.png'
        )

    ### Plot features ###
    # train
    df_T0 = pd.read_csv('src/recommendation/data/T0/demog_ranking_grouped_catbased.csv')
    df_T1 = pd.read_csv('src/recommendation/data/T1/demog_ranking_grouped_catbased.csv')
    df_T1_predicted = pd.read_csv('src/recommendation/data/T1_predicted/demog_ranking_grouped_catbased.csv')

    plot_features(df_T0, df_T1, df_T1_predicted, save_dir='src/recommendation/data_prep/eda/feature_distributions_train', show_plots=False)

    # test
    df_T0 = pd.read_csv('src/recommendation/data/T0/test_with_lifestyle.csv')
    df_T1 = pd.read_csv('src/recommendation/data/T1/test_with_lifestyle.csv')
    df_T1_predicted = pd.read_csv('src/recommendation/data/T1_predicted/test_with_lifestyle.csv')
    
    plot_features(df_T0, df_T1, df_T1_predicted, save_dir='src/recommendation/data_prep/eda/feature_distributions_test', show_plots=False)

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


    # Merge train test
    train_df = pd.read_csv('src/recommendation/data/T0/demog_grouped_catbased.csv')
    train_df = train_df[['CUST_ID','Number of Children','Age','Gender','Education level','Marital status','Region','Occupation Group']]
    test_df = pd.read_csv('src/recommendation/data/T0/test_with_lifestyle.csv')

    merged_df = merge_train_test(train_df, test_df)
    print(merged_df)
    merged_df.to_csv('src/recommendation/data/T0/merged_train_test.csv', index=False)

    # Create demographics dashboard
    csv_file_path = "src/recommendation/data/T0/merged_train_test.csv"
    fig1 = create_demographics_dashboard(csv_file_path, "src/recommendation/data/T0/demographics_dashboard.png")




