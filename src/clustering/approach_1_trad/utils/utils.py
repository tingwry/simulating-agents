from yellowbrick.cluster import SilhouetteVisualizer, KElbowVisualizer
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

# from sklearn.metrics import pairwise_distances

# from langchain_openai import ChatOpenAI
# from langchain.chains import LLMChain
# from langchain.prompts import PromptTemplate
import json
import joblib
import os
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

from src.client.llm import *

def preprocess_customer_data_improved(df):
    df_processed = df.copy()

    # Label encoding for ordinal features
    label_columns = ['Gender', 'Education level']
    label_encoders = {}
    for col in label_columns:
        if col in df_processed.columns:
            le = LabelEncoder()
            df_processed[col + '_encoded'] = le.fit_transform(df_processed[col].astype(str))
            label_encoders[col] = le

    # Binary encoding for categorical features with many categories
    binary_columns = ['Marital status', 'Region', 'Occupation Group']
    binary_encoders = {}
    for col in binary_columns:
        if col in df_processed.columns:
            le = LabelEncoder()
            encoded = le.fit_transform(df_processed[col].astype(str))
            binary_encoders[col] = le

            # Convert to binary representation
            binary_strings = [format(val, 'b') for val in encoded]
            max_len = max(len(b) for b in binary_strings)
            padded_binary = [b.zfill(max_len) for b in binary_strings]

            # Create binary columns
            for i in range(max_len):
                df_processed[f'{col}_bin_{i}'] = [int(b[i]) for b in padded_binary]

    # Define numeric columns
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

    # Filter columns that exist in dataframe
    numeric_feature_columns = [col for col in numeric_feature_columns if col in df_processed.columns]

    # Create list of binary encoded columns
    binary_encoded_columns = []
    for col in binary_columns:
        if col in df_processed.columns:
            max_val = df_processed[col].nunique() - 1
            max_len = len(format(max_val, 'b'))
            binary_encoded_columns.extend([f"{col}_bin_{i}" for i in range(max_len)])

    # Create list of label encoded columns
    label_encoded_columns = [col + '_encoded' for col in label_columns if col in df_processed.columns]

    # Combine all feature columns
    feature_columns = numeric_feature_columns + label_encoded_columns + binary_encoded_columns

    # Handle numeric columns - convert to numeric and handle missing values
    for col in numeric_feature_columns:
        if col in df_processed.columns:
            df_processed[col] = pd.to_numeric(df_processed[col].replace('Unknown', np.nan), errors='coerce')

    # Create feature matrix
    X = df_processed[feature_columns].copy()

    # Handle missing values
    for col in numeric_feature_columns:
        if col in X.columns:
            X[col] = X[col].fillna(X[col].median())

    for col in label_encoded_columns:
        if col in X.columns:
            X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 0)

    return X, label_encoders, binary_encoders, feature_columns

def evaluate_optimal_clusters(X_scaled, max_clusters=20):
    """Evaluate optimal number of clusters using elbow method and silhouette analysis"""
    model = KMeans(random_state=42)

    print("🔍 Running Elbow Method...")
    elbow = KElbowVisualizer(model, k=(2, max_clusters), timings=False)
    elbow.fit(X_scaled)
    # elbow.show()

    print("🔍 Running Silhouette Analysis...")
    silhouette = SilhouetteVisualizer(KMeans(n_clusters=elbow.elbow_value_, random_state=42))
    silhouette.fit(X_scaled)
    # silhouette.show()

    print(f"✅ Suggested optimal number of clusters: {elbow.elbow_value_}")
    return elbow.elbow_value_

def decode_binary_features(df_with_clusters, binary_encoders):
    """Decode binary encoded features back to their original text values"""
    decoded_df = df_with_clusters.copy()

    for col, encoder in binary_encoders.items():
        # if col in df_with_clusters.columns:
        # Find all binary columns for this feature
        max_val = len(encoder.classes_) - 1
        max_len = len(format(max_val, 'b'))
        binary_cols = [f"{col}_bin_{i}" for i in range(max_len)]

        # Check if all binary columns exist
        if all(bc in df_with_clusters.columns for bc in binary_cols):
            # Reconstruct the integer values from binary columns
            decoded_values = []
            for idx in df_with_clusters.index:
                binary_str = ''.join([str(int(df_with_clusters.loc[idx, bc])) for bc in binary_cols])
                integer_val = int(binary_str, 2) if binary_str else 0
                # Convert back to original text using inverse_transform
                try:
                    original_text = encoder.inverse_transform([integer_val])[0]
                    decoded_values.append(original_text)
                except ValueError:
                    # Handle case where integer_val is out of range
                    decoded_values.append('Unknown')

            decoded_df[f'{col}_decoded'] = decoded_values

    return decoded_df

def clustering(train_df, X_scaled, MODEL_DIR, n_clusters):
    # Preserve customer IDs before any processing
    # cust_ids = train_df['CUST_ID'].copy()
    # train_df = train_df.drop('CUST_ID', axis=1)

    df = train_df.copy()

    # Modeling
    clusters = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = clusters.fit_predict(X_scaled)


    # Save Model
    os.makedirs(MODEL_DIR, exist_ok=True)

    existing_models = [f for f in os.listdir(MODEL_DIR) if f.startswith('model_app1_v') and f.endswith('.pkl')]
    if existing_models:
        # Extract version numbers and find the max
        versions = [int(f.split('_v')[1].split('.pkl')[0]) for f in existing_models]
        next_version = max(versions) + 1
    else:
        next_version = 1  # Start with v1 if no

    model_filename = f"model_app1_v{next_version}.pkl"
    model_path = os.path.join(MODEL_DIR, model_filename)

    joblib.dump(clusters, model_path)
    print(f"Model saved as {model_path}")


    df = df.replace('Unknown', np.nan)

    categorical_columns = ['Gender', 'Education level', 'Marital status', 'Region', 'Occupation Group']
    for col in df.columns:
        if col not in categorical_columns and col != 'CUST_ID':
            df[col] = df[col].astype(float)
        else:
            df[col] = df[col].astype('category')

    df["cluster"] = cluster_labels

    return df, n_clusters, clusters, next_version
    

def calculate_cluster_statistics(train_df, clustering, label_encoders, binary_encoders):
    """Calculate comprehensive statistics with proper binary decoding and text mapping"""
    # Add cluster labels to dataframe
    df_with_clusters = train_df.copy()
    df_with_clusters['cluster'] = clustering.labels_

    # Decode binary features back to original text
    df_decoded = decode_binary_features(df_with_clusters, binary_encoders)

    # Define feature groups
    numeric_features = [
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

    label_encoded_features = ['Gender', 'Education level']
    binary_encoded_features = list(binary_encoders.keys())

    # Filter features that exist in dataframe
    numeric_features = [f for f in numeric_features if f in df_decoded.columns]
    label_encoded_features = [f for f in label_encoded_features if f in df_decoded.columns]

    stats_summary = {}

    # Calculate statistics for each cluster
    for cluster_id in sorted(df_decoded['cluster'].unique()):
        cluster_data = df_decoded[df_decoded['cluster'] == cluster_id]
        cluster_stats = {
            'cluster_size': len(cluster_data),
            'numeric_stats': {},
            'label_encoded_stats': {},
            'binary_encoded_stats': {}
        }

        # Numeric feature statistics
        for feature in numeric_features:
            if feature in cluster_data.columns:
                numeric_data = pd.to_numeric(cluster_data[feature].replace('Unknown', np.nan), errors='coerce')

                unique_vals = numeric_data.dropna().unique()
                is_binary = len(unique_vals) <= 2 and all(val in [0, 1] for val in unique_vals)

                stats = {
                    'mean': numeric_data.mean(),
                    'median': numeric_data.median(),
                    'std': numeric_data.std(),
                    'min': numeric_data.min(),
                    'max': numeric_data.max(),
                    'q25': numeric_data.quantile(0.25),
                    'q75': numeric_data.quantile(0.75),
                    'missing_count': numeric_data.isna().sum(),
                    'missing_pct': (numeric_data.isna().sum() / len(cluster_data)) * 100,
                    'is_binary': is_binary
                }

                if is_binary:
                    stats.update({
                        'count_1': (numeric_data == 1).sum(),
                        'count_0': (numeric_data == 0).sum(),
                        'percentage_1': (numeric_data == 1).sum() / len(cluster_data) * 100,
                        'percentage_0': (numeric_data == 0).sum() / len(cluster_data) * 100
                    })

                cluster_stats['numeric_stats'][feature] = stats

        # Label encoded categorical feature statistics
        for feature in label_encoded_features:
            if feature in cluster_data.columns:
                value_counts = cluster_data[feature].value_counts()
                cluster_stats['label_encoded_stats'][feature] = {
                    'mode': cluster_data[feature].mode().iloc[0] if not cluster_data[feature].mode().empty else 'N/A',
                    'distribution': (value_counts / len(cluster_data) * 100).to_dict(),
                    'unique_count': cluster_data[feature].nunique()
                }

        # Binary encoded categorical features (using decoded values)
        for feature in binary_encoded_features:
            decoded_col = f'{feature}_decoded'
            if decoded_col in cluster_data.columns:
                value_counts = cluster_data[decoded_col].value_counts()

                # Show the mapping between integer codes and text
                encoder = binary_encoders[feature]
                code_to_text_mapping = {i: text for i, text in enumerate(encoder.classes_)}

                cluster_stats['binary_encoded_stats'][feature] = {
                    'mode': cluster_data[decoded_col].mode().iloc[0] if not cluster_data[decoded_col].mode().empty else 'N/A',
                    'distribution': (value_counts / len(cluster_data) * 100).to_dict(),
                    'unique_count': cluster_data[decoded_col].nunique(),
                    'code_mapping': code_to_text_mapping,
                    'raw_distribution': value_counts.to_dict()
                }

        stats_summary[f'Cluster_{cluster_id}'] = cluster_stats

    return stats_summary, df_decoded


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


def format_cluster_stats_for_analysis(df_group, cluster_id, col):
    """Format cluster statistics into readable text for LLM analysis (Approach 1 version)"""
    cluster_data = df_group[df_group[col] == cluster_id].iloc[0]
    
    formatted_text = f"CLUSTER {cluster_id} PROFILE:\n"
    formatted_text += "=" * 40 + "\n\n"
    
    # Numeric Features
    formatted_text += "NUMERIC FEATURES:\n"
    for feature in ['Age', 'Number of Children', 'Number of Vehicles']:
        if f'{feature}_mean' in cluster_data:
            formatted_text += f"• Average {feature}: {cluster_data[f'{feature}_mean']:.1f}\n"
            formatted_text += f"• Median {feature}: {cluster_data[f'{feature}_median']:.1f}\n"
    
    # Binary Features
    formatted_text += "\nBINARY FEATURES:\n"
    for feature in ['Savings Account', 'Health Insurance', 'Lending']:
        if f'{feature}_mean' in cluster_data:
            formatted_text += f"• {feature} penetration: {cluster_data[f'{feature}_mean']:.1%}\n"
    
    # Categorical Features
    formatted_text += "\nDEMOGRAPHICS:\n"
    for feature in ['Gender', 'Education level', 'Marital status']:
        if feature in cluster_data:
            formatted_text += f"• Most common {feature}: {cluster_data[feature]}\n"
    
    return formatted_text

def analyze_clusters_with_llm(df_group, col):
    """Use LLM to analyze clusters and generate descriptions (Approach 1 version)"""
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
- Demographics (age, education, marital status)
- Financial product usage (savings, lending, insurance)
- Family characteristics (number of children, vehicles)

Provide a brief summary of the key differentiating factors (2-3 sentences).""".format(cluster_stats=comparison_text)

    differentiators_response = client.chat.completions.create(
        model="gpt-4o",
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
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a marketing analyst describing customer segments."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=300
        )
        
        descriptions.append(response.choices[0].message.content.strip())
    
    return descriptions, differentiators

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
    for i, point in enumerate(df_combined_no_out):
        cluster_idx = cluster_labels[i]
        centroid = centroids[cluster_idx]
        distance = euclidean_distances([point], [centroid])[0][0]
        distances.append(distance)
    
    # Create DataFrame with distances
    distance_df = pd.DataFrame({
        'point_index': list(range(len(df_no_outliers))),
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
        'approach': [1],
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


