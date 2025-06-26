import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler  # to standardize the features
from sklearn.decomposition import PCA  # to apply PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

from pyod.models.ecod import ECOD
from yellowbrick.cluster import KElbowVisualizer

import os
import json
import joblib

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from src.client.llm import *

# from langchain.chains import LLMChain
# from langchain.prompts import PromptTemplate

# data visualization
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# import plotly.express as px
# import plotly.graph_objects as go
# import seaborn as sns
# import shap

# sklearn 
# from sklearn.preprocessing import PowerTransformer, OrdinalEncoder
# from sklearn.pipeline import Pipeline
# from sklearn.manifold import TSNE
# from sklearn.metrics import silhouette_score, silhouette_samples, accuracy_score, classification_report

# import lightgbm as lgb
# import prince


def preprocess_num(df):
    """
    Preprocess numeric features
    """

    df_processed = df.copy()

    numeric_feature_columns = [
        'Number of Children', 'Number of Vehicles', 'Savings Account', 
        'Savings Account Subgroup', 'Health Insurance', 'Lending', 
        'Payment', 'Service', 'Business Lending', 'Deposit Account',
        'Deposit Account Balance', 'Deposit Account Transactions',
        'Deposit Account Transactions AVG', 'Deposit Account Transactions MIN',
        'Deposit Account Transactions MAX', 'Deposit Account Inflow',
        'Deposit Account Inflow MIN', 'Deposit Account Inflow MAX',
        'Deposit Account Outflow', 'Deposit Account Outflow MIN',
        'Deposit Account Outflow MAX', 'Deposit Account Inflow Amount', 
        'Deposit Account Outflow Amount', 'Age'
    ]

    for col in numeric_feature_columns:
        if col in df_processed.columns:
            df_processed[col] = pd.to_numeric(df_processed[col].replace('Unknown', np.nan), errors='coerce')

    X = df_processed[numeric_feature_columns].copy()

    for col in numeric_feature_columns:
        if col in X.columns:
            X[col] = X[col].fillna(X[col].median())

    return X, numeric_feature_columns

def embedding_creation(df):

    def compile_text(x):

        text = f"""
        - Gender: {x['Gender']},
        - Education: {x['Education level']},
        - Marital Status: {x['Marital status']},
        - Occupation Group: {x['Occupation Group']},
        - Region: {x['Region']}
        """

        return text

    sentences = df.apply(lambda x: compile_text(x), axis=1).tolist()

    model = SentenceTransformer(r"sentence-transformers/paraphrase-MiniLM-L6-v2")

    output = model.encode(sentences=sentences, show_progress_bar= True, normalize_embeddings  = True)

    df_embedding = pd.DataFrame(output)

    return df_embedding


def dim_reduction(df_embedding, emb_dim):
    scalar = StandardScaler() 
    scaled_data = pd.DataFrame(scalar.fit_transform(df_embedding)) #scaling the data

    pca = PCA(n_components = emb_dim)
    pca.fit(scaled_data)


    data_pca = pca.transform(scaled_data)
    data_pca = pd.DataFrame(data_pca)

    # sns.heatmap(data_pca.corr())
    return data_pca

def clustering(train_df, df_combined, MODEL_DIR, n_clusters):
    # Preserve customer IDs before any processing
    cust_ids = train_df['CUST_ID'].copy()
    train_df = train_df.drop('CUST_ID', axis=1)

    # Outliers
    clf = ECOD()
    clf.fit(df_combined)
    out = clf.predict(df_combined) 
    
    # df_combined["outliers"] = out

    df = train_df.copy()
    # df["outliers"] = out

    # df_combined_no_out = df_combined[df_combined["outliers"] == 0]
    # df_combined_no_out = df_combined_no_out.drop(["outliers"], axis = 1)

    df_combined_with_out = df_combined.copy()
    # df_combined_with_out = df_combined_with_out.drop(["outliers"], axis = 1)

    # Create outlier flags while preserving indices
    outlier_mask = pd.Series(out, index=df_combined.index, name='outliers')
    
    # Apply outlier filtering to both feature and original data
    df_combined_no_out = df_combined[outlier_mask == 0].copy()
    df_no_outliers = train_df[outlier_mask == 0].copy()


    # Elbow
    if n_clusters == None:
        km = KMeans(init="k-means++", random_state=0, n_init="auto")
        visualizer = KElbowVisualizer(km, k=(2,20))
        
        visualizer.fit(df_combined_with_out)        # Fit the data to the visualizer
        # visualizer.show()    

        n_clusters = visualizer.elbow_value_

        print(f"✅ Suggested optimal number of clusters: {n_clusters}")

    # Modeling
    clusters = KMeans(n_clusters=n_clusters, init = "k-means++").fit(df_combined_no_out)
    print(f"Clusters inertia: {clusters.inertia_}")
    clusters_predict = clusters.predict(df_combined_no_out)
    

    # Save Model
    os.makedirs(MODEL_DIR, exist_ok=True)

    existing_models = [f for f in os.listdir(MODEL_DIR) if f.startswith('model_app3_v') and f.endswith('.pkl')]
    if existing_models:
        # Extract version numbers and find the max
        versions = [int(f.split('_v')[1].split('.pkl')[0]) for f in existing_models]
        next_version = max(versions) + 1
    else:
        next_version = 1  # Start with v1 if no

    model_filename = f"model_app3_v{next_version}.pkl"
    model_path = os.path.join(MODEL_DIR, model_filename)

    joblib.dump(clusters, model_path)
    print(f"Model saved as {model_path}")

    # df_no_outliers = df[df["outliers"] == 0]
    # df_no_outliers = df_no_outliers.drop("outliers", axis = 1)
    df_no_outliers = df_no_outliers.replace('Unknown', np.nan)

    # Add back customer IDs to the non-outlier results
    df_no_outliers['CUST_ID'] = cust_ids[df_no_outliers.index]  # This preserves the correct IDs

    categorical_columns = ['Gender', 'Education level', 'Marital status', 'Region', 'Occupation Group']
    for col in df_no_outliers.columns:
        if col not in categorical_columns and col != 'CUST_ID':
            df_no_outliers[col] = df_no_outliers[col].astype(float)
        else:
            df_no_outliers[col] = df_no_outliers[col].astype('category')

    df_no_outliers["cluster"] = clusters_predict

    return df_no_outliers, n_clusters, clusters, next_version, df_combined_no_out

def cluster_stats_info(df_no_outliers, col):
    df_group = df_no_outliers.groupby(col).agg(
        {
            # Original columns (for backward compatibility)
            'Age': ['mean', 'median'],
            'Gender': lambda x: x.value_counts().index[0],
            'Education level': lambda x: x.value_counts().index[0],
            'Marital status': lambda x: x.value_counts().index[0],
            'Occupation Group': lambda x: x.value_counts().index[0],
            'Region': lambda x: x.value_counts().index[0],
            'Number of Children': ['mean', 'median'],
            'Number of Vehicles': ['mean', 'median'],
            'Savings Account': ['mean', 'median'],
            'Savings Account Subgroup': ['mean', 'median'],
            'Health Insurance': ['mean', 'median'],
            'Lending': ['mean', 'median'],
            'Payment': ['mean', 'median'],
            'Service': ['mean', 'median'],
            'Business Lending': ['mean', 'median'],
            'Deposit Account': ['mean', 'median'],
            'Deposit Account Balance': ['mean', 'median'],
            'Deposit Account Transactions': ['mean', 'median'],
            'Deposit Account Transactions AVG': ['mean', 'median'],
            'Deposit Account Transactions MIN': ['mean', 'median'],
            'Deposit Account Transactions MAX': ['mean', 'median'],
            'Deposit Account Inflow': ['mean', 'median'],
            'Deposit Account Inflow MIN': ['mean', 'median'],
            'Deposit Account Inflow MAX': ['mean', 'median'],
            'Deposit Account Outflow': ['mean', 'median'],
            'Deposit Account Outflow MIN': ['mean', 'median'],
            'Deposit Account Outflow MAX': ['mean', 'median'],
            'Deposit Account Inflow Amount': ['mean', 'median'],
            'Deposit Account Outflow Amount': ['mean', 'median'],
        }
    )

    df_group.columns = ['_'.join(col).strip('_') for col in df_group.columns.values]

    df_group = df_group.reset_index()
    df_group.columns = df_group.columns.str.replace('_<lambda>', '', regex=False)

    # cluster_size
    cluster_counts = df_no_outliers[col].value_counts().reset_index()
    cluster_counts.columns = [col, 'cluster_size']

    df_group = pd.merge(df_group, cluster_counts, on=col)

    cols = [col, 'cluster_size'] + [c for c in df_group.columns if c not in [col, 'cluster_size']]
    df_group = df_group[cols]
    
    return df_group


def should_merge_clusters(df_group, size_threshold_ratio=1):
    """
    Determine if any clusters should be merged based on size threshold.
    
    Args:
        df_group: DataFrame with cluster statistics
        size_threshold_ratio: Minimum size ratio compared to median cluster size
        
    Returns:
        bool: True if merging should occur, False otherwise
        list: Clusters to be merged (empty if no merging)
    """
    # Calculate cluster size statistics
    median_size = df_group['cluster_size'].median()
    threshold = median_size * size_threshold_ratio
    
    # Identify small clusters
    small_clusters = df_group[df_group['cluster_size'] < threshold]
    
    if len(small_clusters) == 0:
        return False, []
    
    # Only merge if we have at least one small cluster AND 
    # the small clusters represent less than 20% of total data
    # total_small_size = small_clusters['cluster_size'].sum()
    # total_size = df_group['cluster_size'].sum()
    
    # if total_small_size / total_size < 0.2:
    return True, small_clusters['cluster'].tolist()


def adjust_clusters(df_group, clusters_to_merge):
    # find similar clusters
    df_for_similarity = df_group.copy()
    if 'cluster' in df_for_similarity.columns:
        df_for_similarity = df_for_similarity.drop(columns=['cluster'])
    if 'cluster_size' in df_for_similarity.columns:
        df_for_similarity = df_for_similarity.drop(columns=['cluster_size'])
    if 'description' in df_for_similarity.columns:
        df_for_similarity = df_for_similarity.drop(columns=['description'])
    if 'outliers' in df_for_similarity.columns:
        df_for_similarity = df_for_similarity.drop(columns=['outliers'])

    numerical_df = df_for_similarity.select_dtypes(include=np.number)
    distance_matrix = euclidean_distances(numerical_df)
    distance_df = pd.DataFrame(distance_matrix, index=numerical_df.index, columns=numerical_df.index)

    most_similar_row_index = []
    min_distance = []

    for i in range(len(distance_df)):
        distances_for_row = distance_df.iloc[i].drop(distance_df.index[i])

        min_dist_index = distances_for_row.idxmin()

        most_similar_row_index.append(min_dist_index)
        min_distance.append(distances_for_row.min())

    similarity_results = pd.DataFrame({
        'Original_Row_Index': numerical_df.index,
        'Most_Similar_Row_Index': most_similar_row_index,
        'Minimum_Distance': min_distance
    })

    print("Most similar row for each row based on numerical features:")
    print(similarity_results)

    # Create cluster mapping
    distance_df = pd.DataFrame(distance_matrix, index=df_group['cluster'], columns=df_group['cluster'])
    
    cluster_mapping = {}
    small_clusters = set(clusters_to_merge)
    all_clusters = set(df_group['cluster'])
    large_clusters = all_clusters - small_clusters
    
    for small_cluster in small_clusters:
        distances_to_large = distance_df.loc[small_cluster, list(large_clusters)]
        # Find the closest large cluster
        closest_large = distances_to_large.idxmin()
        cluster_mapping[small_cluster] = closest_large
    
    # Large clusters map to themselves
    for large_cluster in large_clusters:
        cluster_mapping[large_cluster] = large_cluster
    
    print("Cluster mapping:")
    print(cluster_mapping)
    
    return cluster_mapping

def apply_cluster_mapping(df_no_outliers, cluster_mapping):
    """
    Apply the cluster mapping to the original dataframe
    
    Args:
        df_no_outliers: Original dataframe with 'cluster' column
        cluster_mapping: Dictionary mapping old clusters to new clusters
        
    Returns:
        DataFrame with new 'merged_cluster' column
    """
    df = df_no_outliers.copy()
    df['merged_cluster'] = df['cluster'].map(cluster_mapping)
    
    # For any unmapped clusters (shouldn't happen if mapping is complete)
    df['merged_cluster'] = df['merged_cluster'].fillna(df['cluster'])
    
    return df





# LLM

def format_cluster_stats_for_analysis(df_group, cluster_id, col):
    """Format cluster statistics into readable text for LLM analysis"""
    cluster_data = df_group[df_group[col] == cluster_id].iloc[0]
    
    formatted_text = f"CLUSTER {cluster_id} PROFILE:\n"
    formatted_text += "=" * 40 + "\n\n"
    
    # Demographics
    formatted_text += "DEMOGRAPHICS:\n"
    formatted_text += f"• Average Age: {cluster_data['Age_mean']:.1f} years\n"
    formatted_text += f"• Median Age: {cluster_data['Age_median']:.1f} years\n"
    formatted_text += f"• Most common Gender: {cluster_data['Gender']}\n"
    formatted_text += f"• Most common Education: {cluster_data['Education level']}\n"
    formatted_text += f"• Most common Marital Status: {cluster_data['Marital status']}\n"
    formatted_text += f"• Most common Occupation: {cluster_data['Occupation Group']}\n"
    formatted_text += f"• Most common Region: {cluster_data['Region']}\n"
    formatted_text += f"• Average Number of Children: {cluster_data['Number of Children_mean']:.1f}\n"
    formatted_text += f"• Median Number of Children: {cluster_data['Number of Children_median']:.1f}\n"
    formatted_text += f"• Average Number of Vehicles: {cluster_data['Number of Vehicles_mean']:.1f}\n"
    formatted_text += f"• Median Number of Vehicles: {cluster_data['Number of Vehicles_median']:.1f}\n\n"
    
    # Financial Products & Services
    formatted_text += "FINANCIAL PRODUCTS & SERVICES:\n"
    formatted_text += f"• Average Savings Account: {cluster_data['Savings Account_mean']:.1%}\n"
    formatted_text += f"• Median Savings Account: {cluster_data['Savings Account_median']:.1%}\n"
    formatted_text += f"• Average Savings Account Subgroup: {cluster_data['Savings Account Subgroup_mean']:.1%}\n"
    formatted_text += f"• Median Savings Account Subgroup: {cluster_data['Savings Account Subgroup_median']:.1%}\n"
    formatted_text += f"• Average Health Insurance: {cluster_data['Health Insurance_mean']:.1%}\n"
    formatted_text += f"• Median Health Insurance: {cluster_data['Health Insurance_median']:.1%}\n"
    formatted_text += f"• Average Lending: {cluster_data['Lending_mean']:.1%}\n"
    formatted_text += f"• Median Lending: {cluster_data['Lending_median']:.1%}\n"
    formatted_text += f"• Average Payment Services: {cluster_data['Payment_mean']:.1%}\n"
    formatted_text += f"• Median Payment Services: {cluster_data['Payment_median']:.1%}\n"
    formatted_text += f"• Average General Services: {cluster_data['Service_mean']:.1%}\n"
    formatted_text += f"• Median General Services: {cluster_data['Service_median']:.1%}\n"
    formatted_text += f"• Average Business Lending: {cluster_data['Business Lending_mean']:.1%}\n"
    formatted_text += f"• Median Business Lending: {cluster_data['Business Lending_median']:.1%}\n\n"
    
    # Deposit Account Behavior
    formatted_text += "DEPOSIT ACCOUNT BEHAVIOR:\n"
    formatted_text += f"• Average Deposit Account: {cluster_data['Deposit Account_mean']:.1%}\n"
    formatted_text += f"• Median Deposit Account: {cluster_data['Deposit Account_median']:.1%}\n"
    formatted_text += f"• Average Balance: ${cluster_data['Deposit Account Balance_mean']:,.0f}\n"
    formatted_text += f"• Median Balance: ${cluster_data['Deposit Account Balance_median']:,.0f}\n"
    formatted_text += f"• Average Monthly Transactions: {cluster_data['Deposit Account Transactions_mean']:.1f}\n"
    formatted_text += f"• Median Monthly Transactions: {cluster_data['Deposit Account Transactions_median']:.1f}\n"
    formatted_text += f"• Average Transaction Amount: ${cluster_data['Deposit Account Transactions AVG_mean']:,.0f}\n"
    formatted_text += f"• Median Transaction Amount: ${cluster_data['Deposit Account Transactions AVG_median']:,.0f}\n"
    formatted_text += f"• Average Min Transaction: ${cluster_data['Deposit Account Transactions MIN_mean']:,.0f}\n"
    formatted_text += f"• Median Min Transaction: ${cluster_data['Deposit Account Transactions MIN_median']:,.0f}\n"
    formatted_text += f"• Average Max Transaction: ${cluster_data['Deposit Account Transactions MAX_mean']:,.0f}\n"
    formatted_text += f"• Median Max Transaction: ${cluster_data['Deposit Account Transactions MAX_median']:,.0f}\n\n"
    
    # Cash Flow Patterns
    formatted_text += "CASH FLOW PATTERNS:\n"
    formatted_text += f"• Average Monthly Inflows: {cluster_data['Deposit Account Inflow_mean']:.1f}\n"
    formatted_text += f"• Median Monthly Inflows: {cluster_data['Deposit Account Inflow_median']:.1f}\n"
    formatted_text += f"• Average Min Inflow: ${cluster_data['Deposit Account Inflow MIN_mean']:,.0f}\n"
    formatted_text += f"• Median Min Inflow: ${cluster_data['Deposit Account Inflow MIN_median']:,.0f}\n"
    formatted_text += f"• Average Max Inflow: ${cluster_data['Deposit Account Inflow MAX_mean']:,.0f}\n"
    formatted_text += f"• Median Max Inflow: ${cluster_data['Deposit Account Inflow MAX_median']:,.0f}\n"
    formatted_text += f"• Average Monthly Outflows: {cluster_data['Deposit Account Outflow_mean']:.1f}\n"
    formatted_text += f"• Median Monthly Outflows: {cluster_data['Deposit Account Outflow_median']:.1f}\n"
    formatted_text += f"• Average Min Outflow: ${cluster_data['Deposit Account Outflow MIN_mean']:,.0f}\n"
    formatted_text += f"• Median Min Outflow: ${cluster_data['Deposit Account Outflow MIN_median']:,.0f}\n"
    formatted_text += f"• Average Max Outflow: ${cluster_data['Deposit Account Outflow MAX_mean']:,.0f}\n"
    formatted_text += f"• Median Max Outflow: ${cluster_data['Deposit Account Outflow MAX_median']:,.0f}\n"
    formatted_text += f"• Average Total Inflow Amount: ${cluster_data['Deposit Account Inflow Amount_mean']:,.0f}\n"
    formatted_text += f"• Median Total Inflow Amount: ${cluster_data['Deposit Account Inflow Amount_median']:,.0f}\n"
    formatted_text += f"• Average Total Outflow Amount: ${cluster_data['Deposit Account Outflow Amount_mean']:,.0f}\n"
    formatted_text += f"• Median Total Outflow Amount: ${cluster_data['Deposit Account Outflow Amount_median']:,.0f}\n\n"
    
    return formatted_text

def analyze_clusters_with_llm(df_group, col):
    """Use LLM to analyze clusters and generate descriptions"""
    client = get_aoi_client()
    
    # Get all cluster statistics
    all_cluster_stats = []
    cluster_ids = sorted(df_group[col].unique())
    
    for cluster_id in cluster_ids:
        formatted_stats = format_cluster_stats_for_analysis(df_group, cluster_id, col)
        all_cluster_stats.append(formatted_stats)
    
    # First, identify key differentiating factors
    comparison_text = "\n".join(all_cluster_stats)
    
    differentiator_prompt = """Analyze these customer cluster profiles and identify the 2-3 main factors that differentiate these customer segments:

{cluster_stats}

Focus on the most significant differences in:
- Demographics (age, education, occupation, family size)
- Financial behavior (account balances, transaction patterns, cash flows)
- Product usage (savings, lending, insurance, services)

Provide a brief summary of the key differentiating factors (2-3 sentences).""".format(cluster_stats=comparison_text)

    # Use CHAT COMPLETIONS API for GPT-4o
    differentiators_response = client.chat.completions.create(
        model="gpt-4o",  # Your GPT-4o deployment name
        messages=[
            {"role": "system", "content": "You are a data analyst who identifies key differences between customer segments."},
            {"role": "user", "content": differentiator_prompt}
        ],
        temperature=0.3,
        max_tokens=200
    )
    differentiators = differentiators_response.choices[0].message.content
    print(f"Key differentiating factors: {differentiators}\n")
    
    # Analyze each cluster individually
    cluster_prompt_template = """Based on this customer profile, provide a concise business description of this customer segment:

{cluster_stats}

Key differentiating factors to consider: {differentiators}

Provide a 2-3 sentence description that captures:
1. The demographic profile of this segment
2. Their key financial behaviors and product usage patterns
3. What makes them distinct from other segments

Write in a business-friendly tone suitable for marketing and customer strategy."""

    print("Generating cluster descriptions...")
    descriptions = []
    
    for cluster_id in cluster_ids:
        print(f"Analyzing Cluster {cluster_id}...")
        formatted_stats = format_cluster_stats_for_analysis(df_group, cluster_id, col)
        
        prompt = cluster_prompt_template.format(
            cluster_stats=formatted_stats,
            differentiators=differentiators
        )
        
        response = client.chat.completions.create(
            model="gpt-4o",  # Your GPT-4o deployment name
            messages=[
                {"role": "system", "content": "You are a marketing analyst describing customer segments."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=300
        )
        
        descriptions.append(response.choices[0].message.content.strip())
    
    return descriptions, differentiators

def add_llm_descriptions_to_clusters(df_group, col):
    """Add LLM-generated descriptions to the cluster DataFrame"""
    # Generate descriptions using LLM
    descriptions, differentiators = analyze_clusters_with_llm(df_group, col)
    
    # Create a mapping of cluster to description
    cluster_ids = sorted(df_group[col].unique())
    description_mapping = dict(zip(cluster_ids, descriptions))
    
    # Add description column to dataframe
    df_group_with_descriptions = df_group.copy()
    df_group_with_descriptions['description'] = df_group_with_descriptions[col].map(description_mapping)
    
    return df_group_with_descriptions, differentiators

def add_descriptions_simple(df_group, col):
    """Simplified version to just add descriptions to existing df_group"""
    try:
        descriptions, differentiators = analyze_clusters_with_llm(df_group, col)
        df_group['description'] = df_group[col].map(dict(zip(sorted(df_group[col].unique()), descriptions)))
        print("✅ Descriptions added successfully!")
        print(f"\nKey differentiators: {differentiators}")
        return df_group
    except Exception as e:
        print(f"❌ Error: {e}")
        return df_group
    

def calculate_cluster_distances(df_no_outliers, df_combined_no_out, clusters, cluster_col='cluster'):
    """
    Calculate average distance of points to their cluster centroids
    
    Args:
        df_no_outliers: DataFrame with cluster assignments
        df_combined_no_out: Original feature space data (without outliers)
        clusters: Fitted KMeans model
        cluster_col: Column name containing cluster assignments
        
    Returns:
        DataFrame with cluster numbers and average distances
    """
    # Get centroids and cluster assignments
    centroids = clusters.cluster_centers_
    cluster_labels = df_no_outliers[cluster_col].values
    
    # Calculate distances from each point to its centroid
    distances = []
    for i, point in enumerate(df_combined_no_out.values):
        cluster_idx = cluster_labels[i]
        centroid = centroids[cluster_idx]
        distance = euclidean_distances([point], [centroid])[0][0]
        distances.append(distance)
    
    # Create DataFrame with distances
    distance_df = pd.DataFrame({
        'point_index': df_no_outliers.index,
        cluster_col: cluster_labels,
        'distance_to_centroid': distances
    })
    
    # Calculate average distance per cluster
    cluster_stats = distance_df.groupby(cluster_col)['distance_to_centroid'].agg(
        ['mean', 'median', 'std', 'count']
    ).reset_index()
    
    cluster_stats.columns = [
        cluster_col, 
        'avg_distance_to_centroid', 
        'median_distance_to_centroid',
        'std_distance_to_centroid',
        'n_points'
    ]
    
    return cluster_stats


def calculate_overall_distance(df_clus_level_eval):
    """
    Calculate overall average distance from centroids across all clusters
    
    Args:
        df_clus_level_eval: DataFrame from calculate_cluster_distances()
        
    Returns:
        DataFrame with single row containing overall evaluation metrics
    """
    # Calculate weighted average (weighted by cluster size)
    total_distance = (df_clus_level_eval['avg_distance_to_centroid'] * 
                     df_clus_level_eval['n_points']).sum()
    total_points = df_clus_level_eval['n_points'].sum()
    overall_avg_distance = total_distance / total_points
    
    # Create single-row DataFrame
    overall_eval = pd.DataFrame({
        'approach': [2],
        'overall_avg_distance_from_centroids': [overall_avg_distance],
        'total_clusters': [len(df_clus_level_eval)],
        'total_points': [total_points]
    })
    
    return overall_eval


def save_csv_file(DIR, df, file_name, next_version):
    os.makedirs(DIR, exist_ok=True)
    # existing_results = [f for f in os.listdir(DIR) 
    #                   if f.startswith(f'{file_name}_v') and f.endswith('.csv')]
    
    # if existing_results:
    #     versions = [int(f.split('_v')[1].split('.csv')[0]) for f in existing_results]
    #     next_version = max(versions) + 1
    # else:
    #     next_version = 1

    result_filename = f"{file_name}_v{next_version}.csv"
    result_path = os.path.join(DIR, result_filename)
    
    df.to_csv(result_path, index=False)
    
    print(f"\n✅ Results saved to: {result_path}")



def save_json_file(DIR, cluster_map, file_name, next_version):
    os.makedirs(DIR, exist_ok=True)

    serializable_map = {k: int(v) if isinstance(v, (np.integer, np.int32, np.int64)) else v 
                       for k, v in cluster_map.items()}
    
    # existing_files = [f for f in os.listdir(DIR) 
    #                  if f.startswith(f'{file_name}_v') and f.endswith('.json')]
    
    # if existing_files:
    #     versions = [int(f.split('_v')[1].split('.json')[0]) for f in existing_files]
    #     next_version = max(versions) + 1
    # else:
    #     next_version = 1

    result_filename = f"{file_name}_v{next_version}.json"
    result_path = os.path.join(DIR, result_filename)

    with open(result_path, 'w') as f:
        json.dump(serializable_map, f, indent=4)
    
    print(f"\n✅ Cluster mapping saved to: {result_path}")
    return result_path


