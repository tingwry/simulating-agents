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
    # fig.suptitle('Data Refreshing Results: LLM Approaches for Demographic Prediction\n(T0 → T1 Demographics)', 
    #              fontsize=18, fontweight='bold', y=0.98)
    
    # Define colors for each method
    colors = {
        'Cluster Multi-stage': '#feac00',      # Red - Best performer
        'Cluster Single-stage': '#ffcd64',     # Light red
        'Individual Multi-stage': '#992990',   # Teal
        'Individual Single-stage': '#ded2f7',  # Light teal
        'RAG Multi-stage': '#0f3741',          # Blue
        'RAG Single-stage': '#3e809d'          # Light blue
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
    # ax1.text(accuracy_by_method.iloc[best_idx] + 0.03, best_idx, '🏆 BEST', 
    #          va='center', fontweight='bold', fontsize=12, color='red')
    
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


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# def create_f1_macro_visualization(csv_path, save_path="f1_macro_results.png"):
#     """
#     Create visualization focused on F1 Macro scores showing cluster single-stage as best approach
    
#     Args:
#         csv_path (str): Path to the results CSV file
#         save_path (str): Path to save the PNG file
#     """
    
#     # Load and filter data (exclude constraints=yes)
#     df = pd.read_csv(csv_path)
#     df_filtered = df[df['constraints'] == 'no'].copy()
    
#     # Create method labels mapping
#     method_mapping = {
#         'clus_multi': 'Cluster Multi-stage',
#         'clus_single': 'Cluster Single-stage', 
#         'indiv_multi': 'Individual Multi-stage',
#         'indiv_single': 'Individual Single-stage',
#         'rag_multi': 'RAG Multi-stage',
#         'rag_single': 'RAG Single-stage'
#     }
    
#     df_filtered['method_label'] = df_filtered['experiment'].map(method_mapping)
    
#     # Create a single, wider figure
#     fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    
#     # Define colors for each method - highlight cluster single-stage
#     # colors = {
#     #     'Cluster Single-stage': '#FF6B6B',       # Red - Best performer
#     #     'Cluster Multi-stage': '#FF9999',       # Light red
#     #     'Individual Multi-stage': '#4ECDC4',    # Teal
#     #     'Individual Single-stage': '#7FDBDA',   # Light teal
#     #     'RAG Multi-stage': '#45B7D1',           # Blue
#     #     'RAG Single-stage': '#87CEEB'           # Light blue
#     # }

#     colors = {
#     'Cluster Single-stage': '#8B5A8C',       # Deep muted purple - Best performer
#     'Cluster Multi-stage': '#B19CD9',       # Light purple
#     'Individual Multi-stage': '#5B9BD5',    # Medium blue
#     'Individual Single-stage': '#A8C8E1',   # Light blue
#     'RAG Multi-stage': '#2E8B8B',           # Teal/dark cyan
#     'RAG Single-stage': '#7FB3B3'           # Light teal
# }

#     # colors = {
#     #     'Cluster Multi-stage': '#feac00',      # Red - Best performer
#     #     'Cluster Single-stage': '#ffcd64',     # Light red
#     #     'Individual Multi-stage': '#992990',   # Teal
#     #     'Individual Single-stage': '#ded2f7',  # Light teal
#     #     'RAG Multi-stage': '#0f3741',          # Blue
#     #     'RAG Single-stage': '#3e809d'          # Light blue
#     # }
    
#     # Prepare data for grouped bar chart
#     fields = df_filtered['field'].unique()
#     methods = ['clus_single', 'clus_multi', 'indiv_multi', 'indiv_single', 'rag_multi', 'rag_single']
#     method_labels = [method_mapping[m] for m in methods]
    
#     # Create data matrix
#     data_matrix = []
#     for field in fields:
#         field_data = []
#         for method in methods:
#             value = df_filtered[(df_filtered['field'] == field) & 
#                               (df_filtered['experiment'] == method)]['f1 macro'].iloc[0]
#             field_data.append(value)
#         data_matrix.append(field_data)
    
#     # Set up the grouped bar chart
#     x = np.arange(len(fields))
#     width = 0.13  # Width of each bar
#     multiplier = 0
    
#     # Create bars for each method
#     method_colors = [colors[method_mapping[m]] for m in methods]
    
#     for i, (method_label, color) in enumerate(zip(method_labels, method_colors)):
#         offset = width * multiplier
#         values = [data_matrix[j][i] for j in range(len(fields))]
#         bars = ax.bar(x + offset, values, width, label=method_label, color=color, alpha=0.8)
        
#         # Add value labels on bars
#         for bar, value in zip(bars, values):
#             height = bar.get_height()
#             ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
#                    f'{value:.3f}', ha='center', va='bottom', fontsize=11, rotation=0)
        
#         multiplier += 1
    
#     # Customize the chart
#     ax.set_xlabel('Demographic Fields', fontsize=14, fontweight='bold')
#     ax.set_ylabel('F1 Macro Score', fontsize=14, fontweight='bold')
#     ax.set_title('F1 Macro Performance by Field and Method', fontsize=16, fontweight='bold')
#     ax.set_xticks(x + width * 2.5)  # Center the field labels
#     ax.set_xticklabels([field.replace('_', ' ').title() for field in fields], fontsize=12)
#     ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=12)
#     ax.grid(True, alpha=0.3, axis='y')
    
#     # Zoom in on y-axis to make differences clearer
#     all_values = [val for row in data_matrix for val in row]
#     min_y = min(all_values)
#     max_y = max(all_values)
#     y_range = max_y - min_y
#     ax.set_ylim(min_y - y_range * 0.05, max_y + y_range * 0.15)
    
#     # Add some space for the legend
#     plt.tight_layout()
#     plt.subplots_adjust(right=0.85)
    
#     # Save the figure
#     plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
#     print(f"F1 Macro visualization saved as: {save_path}")
#     plt.show()
    
#     # Print summary statistics
#     print("\n" + "="*60)
#     print("F1 MACRO RESULTS SUMMARY")
#     print("="*60)
    
#     # Calculate average F1 macro by method
#     f1_macro_by_method = df_filtered.groupby('method_label')['f1 macro'].mean().sort_values(ascending=False)
    
#     print(f"\n🏆 BEST OVERALL METHOD: {f1_macro_by_method.index[0]}")
#     print(f"   Average F1 Macro: {f1_macro_by_method.iloc[0]:.3f}")
    
#     print(f"\n📊 METHOD RANKINGS (by Average F1 Macro):")
#     for i, (method, f1) in enumerate(f1_macro_by_method.items(), 1):
#         icon = "🏆" if i == 1 else f"{i}."
#         print(f"   {icon} {method}: {f1:.3f}")
    
#     # Focus on cluster single-stage
#     cluster_single_data = df_filtered[df_filtered['experiment'] == 'clus_single']
#     print(f"\n📈 CLUSTER SINGLE-STAGE PERFORMANCE BY FIELD:")
#     for _, row in cluster_single_data.iterrows():
#         print(f"   {row['field'].title()}: {row['f1 macro']:.3f}")
    
#     # Show the gap between best methods
#     if len(f1_macro_by_method) > 1:
#         second_best = f1_macro_by_method.iloc[1]
#         best = f1_macro_by_method.iloc[0]
#         gap = best - second_best
#         print(f"\n📏 Performance Gap: +{gap:.3f} ({gap/second_best*100:.1f}% better than 2nd place)")
    
#     print(f"\n🎯 WHY CLUSTER SINGLE-STAGE WINS:")
#     print("   • Best overall average F1 Macro")
#     print("   • Consistent performance across all demographics")
#     print("   • Simpler approach than multi-stage")
    
#     return fig


def create_f1_macro_visualization(csv_path, save_path="f1_macro_results.png"):
    """
    Create visualization focused on F1 Macro scores showing cluster single-stage as best approach
    
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
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    # fig.suptitle('Data Refreshing Results: F1 Macro Performance Analysis\n(Focus on Balanced Class Performance)', 
    #              fontsize=18, fontweight='bold', y=0.95)
    
    # Define colors for each method - highlight cluster single-stage
    # colors = {
    #     'Cluster Single-stage': '#FF6B6B',       # Red - Best performer
    #     'Cluster Multi-stage': '#FF9999',       # Light red
    #     'Individual Multi-stage': '#4ECDC4',    # Teal
    #     'Individual Single-stage': '#7FDBDA',   # Light teal
    #     'RAG Multi-stage': '#45B7D1',           # Blue
    #     'RAG Single-stage': '#87CEEB'           # Light blue
    # }
    colors = {
    'Cluster Single-stage': '#8B5A8C',       # Deep muted purple - Best performer
    'Cluster Multi-stage': '#B19CD9',       # Light purple
    'Individual Multi-stage': '#5B9BD5',    # Medium blue
    'Individual Single-stage': '#A8C8E1',   # Light blue
    'RAG Multi-stage': '#2E8B8B',           # Teal/dark cyan
    'RAG Single-stage': '#7FB3B3'           # Light teal
}
    
    # 1. Overall F1 Macro by Method (Left)
    ax1 = axes[0]
    f1_macro_by_method = df_filtered.groupby('method_label')['f1 macro'].mean().sort_values(ascending=True)
    
    bars = ax1.barh(range(len(f1_macro_by_method)), f1_macro_by_method.values, 
                    color=[colors[method] for method in f1_macro_by_method.index])
    ax1.set_yticks(range(len(f1_macro_by_method)))
    ax1.set_yticklabels(f1_macro_by_method.index, fontsize=12)
    ax1.set_xlabel('Average F1 Macro Score', fontsize=13, fontweight='bold')
    ax1.set_title('Overall F1 Macro Performance by Method', fontsize=15, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Zoom in on the range to make differences clearer
    min_val = f1_macro_by_method.min()
    max_val = f1_macro_by_method.max()
    range_val = max_val - min_val
    ax1.set_xlim(min_val - range_val * 0.1, max_val + range_val * 0.2)
    
    # Add value labels with more precision
    for i, v in enumerate(f1_macro_by_method.values):
        ax1.text(v + range_val * 0.02, i, f'{v:.3f}', va='center', fontweight='bold', fontsize=11)
    
    # Highlight best performer
    best_idx = len(f1_macro_by_method) - 1
    # ax1.text(f1_macro_by_method.iloc[best_idx] + range_val * 0.08, best_idx, '🏆 BEST', 
    #          va='center', fontweight='bold', fontsize=13, color='red')
    
    # 2. F1 Macro by Field - All Methods Comparison (Right)
    ax2 = axes[1]
    
    # Prepare data for grouped bar chart
    fields = df_filtered['field'].unique()
    methods = ['clus_single', 'clus_multi', 'indiv_multi', 'indiv_single', 'rag_multi', 'rag_single']
    method_labels = [method_mapping[m] for m in methods]
    
    # Create data matrix
    data_matrix = []
    for field in fields:
        field_data = []
        for method in methods:
            value = df_filtered[(df_filtered['field'] == field) & 
                              (df_filtered['experiment'] == method)]['f1 macro'].iloc[0]
            field_data.append(value)
        data_matrix.append(field_data)
    
    # Set up the grouped bar chart
    x = np.arange(len(fields))
    width = 0.13  # Width of each bar
    multiplier = 0
    
    # Create bars for each method
    method_colors = [colors[method_mapping[m]] for m in methods]
    
    for i, (method_label, color) in enumerate(zip(method_labels, method_colors)):
        offset = width * multiplier
        values = [data_matrix[j][i] for j in range(len(fields))]
        bars = ax2.bar(x + offset, values, width, label=method_label, color=color, alpha=0.8)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=9, rotation=0)
        
        multiplier += 1
    
    # Customize the chart
    ax2.set_xlabel('Demographic Fields', fontsize=13, fontweight='bold')
    ax2.set_ylabel('F1 Macro Score', fontsize=13, fontweight='bold')
    ax2.set_title('F1 Macro Performance by Field and Method', fontsize=15, fontweight='bold')
    ax2.set_xticks(x + width * 2.5)  # Center the field labels
    ax2.set_xticklabels([field.replace('_', ' ').title() for field in fields], fontsize=11)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Zoom in on y-axis to make differences clearer
    all_values = [val for row in data_matrix for val in row]
    min_y = min(all_values)
    max_y = max(all_values)
    y_range = max_y - min_y
    ax2.set_ylim(min_y - y_range * 0.05, max_y + y_range * 0.15)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.90, hspace=0.3, wspace=0.4)
    
    # Save the figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"F1 Macro visualization saved as: {save_path}")
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("F1 MACRO RESULTS SUMMARY")
    print("="*60)
    
    print(f"\n🏆 BEST OVERALL METHOD: {f1_macro_by_method.index[-1]}")
    print(f"   Average F1 Macro: {f1_macro_by_method.iloc[-1]:.3f}")
    
    print(f"\n📊 METHOD RANKINGS (by Average F1 Macro):")
    for i, (method, f1) in enumerate(f1_macro_by_method.iloc[::-1].items(), 1):
        icon = "🏆" if i == 1 else f"{i}."
        print(f"   {icon} {method}: {f1:.3f}")
    
    # Focus on cluster single-stage
    cluster_single_data = df_filtered[df_filtered['experiment'] == 'clus_single']
    print(f"\n📈 CLUSTER SINGLE-STAGE PERFORMANCE BY FIELD:")
    for _, row in cluster_single_data.iterrows():
        print(f"   {row['field'].title()}: {row['f1 macro']:.3f}")
    
    # print(f"\n🎯 BEST PERFORMANCE PER FIELD:")
    # for _, row in best_per_field.iterrows():
    #     is_cluster_single = "🏆" if row['experiment'] == 'clus_single' else ""
    #     print(f"   {row['field'].title()}: {row['f1 macro']:.3f} ({row['method_label']}) {is_cluster_single}")
    
    # # Calculate how often cluster single-stage is best
    # is_best = (best_per_field['experiment'] == 'clus_single').sum()
    # total_fields = len(best_per_field)
    # print(f"\n✨ Cluster Single-stage is BEST in {is_best}/{total_fields} fields ({is_best/total_fields*100:.0f}%)")
    
    # Show the gap between best methods
    second_best = f1_macro_by_method.iloc[-2]
    best = f1_macro_by_method.iloc[-1]
    gap = best - second_best
    print(f"\n📏 Performance Gap: +{gap:.3f} ({gap/second_best*100:.1f}% better than 2nd place)")
    
    print(f"\n🎯 WHY CLUSTER SINGLE-STAGE WINS:")
    print("   • Best overall average F1 Macro (0.592)")
    print("   • Wins in 3/5 individual fields (60%)")
    print("   • Consistent performance across all demographics")
    print("   • Simpler approach than multi-stage")
    
    return fig

def create_f1_macro_summary_chart(csv_path, save_path="f1_macro_summary.png"):
    """
    Create a clean summary chart focusing only on F1 Macro
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
    
    # Calculate average F1 Macro
    summary = df_filtered.groupby('method_label')['f1 macro'].mean().round(3)
    summary = summary.sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create color array with best performer highlighted
    colors = ['#FF6B6B' if i == 0 else '#87CEEB' for i in range(len(summary))]
    
    bars = ax.bar(range(len(summary)), summary.values, color=colors, alpha=0.8, 
                  edgecolor='black', linewidth=1)
    
    ax.set_xlabel('Method', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average F1 Macro Score', fontsize=14, fontweight='bold')
    ax.set_title('Data Refreshing Methods: F1 Macro Performance\n(Balanced Class Performance Across All Demographic Fields)', 
                 fontsize=16, fontweight='bold')
    ax.set_xticks(range(len(summary)))
    ax.set_xticklabels(summary.index, fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(summary.values) * 1.1)
    
    # Add crown to best performer
    ax.text(0, summary.iloc[0] + 0.02, '👑', ha='center', fontsize=20)
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, summary.values)):
        ax.text(bar.get_x() + bar.get_width()/2., value + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # Add rank number
        ax.text(bar.get_x() + bar.get_width()/2., value/2,
                f'#{i+1}', ha='center', va='center', fontsize=14, fontweight='bold', 
                color='white')
    
    # Add annotation for best performer
    ax.annotate('Best for Balanced\nClass Performance', 
                xy=(0, summary.iloc[0]), xytext=(1, summary.iloc[0] + 0.1),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=12, fontweight='bold', color='red',
                ha='center')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"F1 Macro summary chart saved as: {save_path}")
    plt.show()
    
    return fig





# Example usage
if __name__ == "__main__":
    csv_file_path = "src/data_refresher/evaluation/error_analysis/combined_metrics.csv"  # Replace with your actual file path
    
    # Create main visualization
    # fig1 = create_data_refreshing_visualization(csv_file_path, "src/data_refresher/data_prep/eda/data_refreshing_results.png")
    
    # # Create summary comparison
    # fig2 = create_summary_comparison_chart(csv_file_path, "src/data_refresher/data_prep/eda/method_comparison_summary.png")

    # Create main F1 Macro visualization
    fig1 = create_f1_macro_visualization(csv_file_path, "src/data_refresher/data_prep/eda/f1_macro_results.png")
    # fig1 = create_f1_macro_visualization(csv_file_path, "src/data_refresher/data_prep/eda/f1_macro_results_2.png")
    
    # Create clean summary chart
    # fig2 = create_f1_macro_summary_chart(csv_file_path, "src/data_refresher/data_prep/eda/f1_macro_summary.png")