import pandas as pd
import os


# df = pd.read_csv('src/mock_filtered.csv')
# DIR = 'src/'

# # df_filtered = df[df['NO_OF_CHLD'] != df['NO_OF_CHLD_prev']][['NO_OF_CHLD', 'NO_OF_CHLD_prev']]
# df_filtered = df[(df['EDU_DESC'] != df['EDU_DESC_prev']) & (~df['EDU_DESC'].isna())][['EDU_DESC', 'EDU_DESC_prev']]

# print(df_filtered)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def create_data_refreshing_visualization(csv_path, save_path="data_refreshing_results.png"):
    """
    Create visualization showing data refreshing results with cluster multi-stage as best approach
    
    Args:
        csv_path (str): Path to the results CSV file
        save_path (str): Path to save the PNG file
    """
    
    # Load and filter data (exclude constraints=yes)
    df = pd.read_csv(csv_path)
    df_filtered = df[df['constraints'] == 'no'].copy()
    
    # Create method labels mapping
    method_mapping = {
        'clus_multi': 'Cluster Multi-stage',
        'clus_single': 'Cluster Single-stage', 
        'indiv_multi': 'Individual Multi-stage',
        'indiv_single': 'Individual Single-stage',
        'rag_multi': 'RAG Multi-stage',
        'rag_single': 'RAG Single-stage'
    }
    
    df_filtered['method_label'] = df_filtered['experiment'].map(method_mapping)
    
    # Create the visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Data Refreshing Results: LLM Approaches for Demographic Prediction\n(T0 → T1 Demographics)', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Define colors for each method
    colors = {
        'Cluster Multi-stage': '#FF6B6B',      # Red - Best performer
        'Cluster Single-stage': '#FF9999',     # Light red
        'Individual Multi-stage': '#4ECDC4',   # Teal
        'Individual Single-stage': '#7FDBDA',  # Light teal
        'RAG Multi-stage': '#45B7D1',          # Blue
        'RAG Single-stage': '#87CEEB'          # Light blue
    }
    
    # 1. Overall Accuracy by Method (Top Left)
    ax1 = axes[0, 0]
    accuracy_by_method = df_filtered.groupby('method_label')['accuracy'].mean().sort_values(ascending=True)
    
    bars = ax1.barh(range(len(accuracy_by_method)), accuracy_by_method.values, 
                    color=[colors[method] for method in accuracy_by_method.index])
    ax1.set_yticks(range(len(accuracy_by_method)))
    ax1.set_yticklabels(accuracy_by_method.index, fontsize=11)
    ax1.set_xlabel('Average Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Overall Accuracy by Method', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, v in enumerate(accuracy_by_method.values):
        ax1.text(v + 0.005, i, f'{v:.3f}', va='center', fontweight='bold')
    
    # Highlight best performer
    best_idx = len(accuracy_by_method) - 1
    ax1.text(accuracy_by_method.iloc[best_idx] + 0.03, best_idx, '🏆 BEST', 
             va='center', fontweight='bold', fontsize=12, color='red')
    
    # 2. Accuracy by Demographic Field (Top Right)
    ax2 = axes[0, 1]
    
    # Pivot data for heatmap
    pivot_data = df_filtered.pivot(index='method_label', columns='field', values='accuracy')
    
    # Reorder methods to put best at top
    method_order = accuracy_by_method.index[::-1]  # Reverse to put best at top
    pivot_data = pivot_data.reindex(method_order)
    
    # Create heatmap
    sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlGn', center=0.75,
                ax=ax2, cbar_kws={'label': 'Accuracy'})
    ax2.set_title('Accuracy by Demographic Field', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Demographic Field', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Method', fontsize=12, fontweight='bold')
    
    # 3. F1 Macro Performance (Bottom Left)
    ax3 = axes[1, 0]
    f1_by_method = df_filtered.groupby('method_label')['f1 macro'].mean().sort_values(ascending=True)
    
    bars = ax3.barh(range(len(f1_by_method)), f1_by_method.values,
                    color=[colors[method] for method in f1_by_method.index])
    ax3.set_yticks(range(len(f1_by_method)))
    ax3.set_yticklabels(f1_by_method.index, fontsize=11)
    ax3.set_xlabel('Average F1 Macro Score', fontsize=12, fontweight='bold')
    ax3.set_title('F1 Macro Score by Method', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, v in enumerate(f1_by_method.values):
        ax3.text(v + 0.01, i, f'{v:.3f}', va='center', fontweight='bold')
    
    # 4. Method Performance by Field (Bottom Right)
    ax4 = axes[1, 1]
    
    # Focus on cluster multi-stage vs others
    cluster_multi_data = df_filtered[df_filtered['experiment'] == 'clus_multi']
    
    # Get best performer for each field
    best_per_field = df_filtered.loc[df_filtered.groupby('field')['accuracy'].idxmax()]
    
    # Create comparison chart
    fields = cluster_multi_data['field'].values
    cluster_multi_acc = cluster_multi_data['accuracy'].values
    best_acc = best_per_field['accuracy'].values
    
    x = np.arange(len(fields))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, cluster_multi_acc, width, label='Cluster Multi-stage', 
                    color=colors['Cluster Multi-stage'], alpha=0.8)
    bars2 = ax4.bar(x + width/2, best_acc, width, label='Best Method per Field', 
                    color='lightgreen', alpha=0.8)
    
    ax4.set_xlabel('Demographic Field', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax4.set_title('Cluster Multi-stage vs Best Method per Field', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(fields, rotation=45, ha='right')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, hspace=0.3, wspace=0.3)
    
    # Save the figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Data refreshing visualization saved as: {save_path}")
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("DATA REFRESHING RESULTS SUMMARY")
    print("="*60)
    
    print(f"\n🏆 BEST OVERALL METHOD: {accuracy_by_method.index[-1]}")
    print(f"   Average Accuracy: {accuracy_by_method.iloc[-1]:.3f}")
    
    print(f"\n📊 METHOD RANKINGS (by Average Accuracy):")
    for i, (method, acc) in enumerate(accuracy_by_method.iloc[::-1].items(), 1):
        print(f"   {i}. {method}: {acc:.3f}")
    
    print(f"\n📈 CLUSTER MULTI-STAGE PERFORMANCE BY FIELD:")
    for _, row in cluster_multi_data.iterrows():
        print(f"   {row['field'].title()}: {row['accuracy']:.3f} accuracy, {row['f1 macro']:.3f} F1")
    
    print(f"\n🎯 BEST PERFORMANCE PER FIELD:")
    for _, row in best_per_field.iterrows():
        print(f"   {row['field'].title()}: {row['accuracy']:.3f} ({row['method_label']})")
    
    # Calculate how often cluster multi-stage is best
    is_best = (best_per_field['experiment'] == 'clus_multi').sum()
    total_fields = len(best_per_field)
    print(f"\n✨ Cluster Multi-stage is BEST in {is_best}/{total_fields} fields ({is_best/total_fields*100:.0f}%)")
    
    return fig

def create_summary_comparison_chart(csv_path, save_path="method_comparison_summary.png"):
    """
    Create a summary chart comparing all methods
    """
    df = pd.read_csv(csv_path)
    df_filtered = df[df['constraints'] == 'no'].copy()
    
    # Method mapping
    method_mapping = {
        'clus_multi': 'Cluster\nMulti-stage',
        'clus_single': 'Cluster\nSingle-stage', 
        'indiv_multi': 'Individual\nMulti-stage',
        'indiv_single': 'Individual\nSingle-stage',
        'rag_multi': 'RAG\nMulti-stage',
        'rag_single': 'RAG\nSingle-stage'
    }
    
    df_filtered['method_label'] = df_filtered['experiment'].map(method_mapping)
    
    # Calculate average metrics
    summary = df_filtered.groupby('method_label').agg({
        'accuracy': 'mean',
        'f1 macro': 'mean',
        'f1 micro': 'mean'
    }).round(3)
    
    # Sort by accuracy
    summary = summary.sort_values('accuracy', ascending=False)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(summary))
    width = 0.25
    
    bars1 = ax.bar(x - width, summary['accuracy'], width, label='Accuracy', alpha=0.8, color='#FF6B6B')
    bars2 = ax.bar(x, summary['f1 macro'], width, label='F1 Macro', alpha=0.8, color='#4ECDC4')
    bars3 = ax.bar(x + width, summary['f1 micro'], width, label='F1 Micro', alpha=0.8, color='#45B7D1')
    
    ax.set_xlabel('Method', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score', fontsize=14, fontweight='bold')
    ax.set_title('Data Refreshing Methods Comparison\n(Average Performance Across All Demographic Fields)', 
                 fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(summary.index, fontsize=11)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1)
    
    # Add crown to best performer
    ax.text(0, summary.iloc[0]['accuracy'] + 0.03, '👑', ha='center', fontsize=20)
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Summary comparison chart saved as: {save_path}")
    plt.show()
    
    return fig

# Example usage
if __name__ == "__main__":
    csv_file_path = "src/data_refresher/evaluation/error_analysis/combined_metrics.csv"  # Replace with your actual file path
    
    # Create main visualization
    fig1 = create_data_refreshing_visualization(csv_file_path, "src/data_refresher/data_prep/eda/data_refreshing_results.png")
    
    # Create summary comparison
    fig2 = create_summary_comparison_chart(csv_file_path, "src/data_refresher/data_prep/eda/method_comparison_summary.png")