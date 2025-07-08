from src.clustering.approach_1.utils.utils import *
import pandas as pd
from sklearn.preprocessing import StandardScaler

train_df = pd.read_csv('src/clustering/data_v2/train_df.csv')
MODEL_DIR = "src/clustering/approach_1/model"
RESULTS_DIR = "src/clustering/approach_1/result"


def train_approach_1(train_df, n_clusters=None):
    """Full pipeline with LLM analysis matching Approach 3's style"""
    print("Step 1: Preprocessing and scaling...")
    X, label_encoders, binary_encoders, feature_columns = preprocess_customer_data_improved(train_df)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Step 2: Determining optimal number of clusters...")
    if n_clusters is None:
        n_clusters = evaluate_optimal_clusters(X_scaled, max_clusters=20)

    print(f"Step 3: Clustering customers into {n_clusters} clusters...")
    df, n_clusters, clusters, next_version = clustering(train_df, X_scaled, MODEL_DIR, n_clusters)

    

    print("Step 4: Calculating comprehensive feature statistics...")
    stats_summary, df_decoded = calculate_cluster_statistics(
        train_df, clusters, label_encoders, binary_encoders
    )


    df_group = pd.DataFrame()
    for cluster_id in range(n_clusters):
        cluster_key = f'Cluster_{cluster_id}'
        stats = stats_summary[cluster_key]
        
        # Create a row for this cluster
        row = {
            'cluster': cluster_id,
            'cluster_size': stats['cluster_size']
        }
        
        # Add numeric stats
        for feature, vals in stats['numeric_stats'].items():
            row[f'{feature}_mean'] = vals['mean']
            row[f'{feature}_median'] = vals['median']
        
        # Add categorical stats
        for feature, vals in stats['label_encoded_stats'].items():
            row[feature] = vals['mode']
        
        df_group = pd.concat([df_group, pd.DataFrame([row])], ignore_index=True)


    col = 'cluster'

    # Check if clusters need merging
    should_merge, clusters_to_merge = should_merge_clusters(df_group)
    if should_merge:
        col = 'merged_cluster'
        print(f"Merging small clusters: {clusters_to_merge}")
        cluster_mapping = adjust_clusters(df_group, clusters_to_merge)
        df = apply_cluster_mapping(df, cluster_mapping)
        df_group = df.groupby(col).agg({
            'cluster': 'count'
        }).rename(columns={'cluster': 'cluster_size'}).reset_index()
    else:
        cluster_mapping = None



    clus_level_eval = calculate_cluster_distances(
        df, 
        X_scaled,
        clusters,  # You'll need to return this from your clustering function
        col
    )
    # Calculate overall evaluation metrics
    overall_eval = calculate_overall_distance(clus_level_eval)

    # Add LLM descriptions
    df_group = add_descriptions_simple(df_group, col)

    clus_explain = df_group[[col, 'cluster_size', 'description']].copy()

    # Save results with versioning
    CLUS_EXPLAIN_DIR = RESULTS_DIR + '/clus_explain'
    FULL_DATA_WITH_CLUSTER_DIR = RESULTS_DIR + '/full_data_with_cluster'
    CLUS_LEVEL_EVAL_DIR = RESULTS_DIR + '/clus_level_eval'
    OVERALL_EVAL_DIR = RESULTS_DIR + '/overall_eval'
    CLUSTER_MAPPING_DIR = RESULTS_DIR + '/cluster_mapping'

    # clus_explain
    save_csv_file(CLUS_EXPLAIN_DIR, clus_explain, 'clus_explain', next_version)
    # full_data_with_cluster
    save_csv_file(FULL_DATA_WITH_CLUSTER_DIR, df, 'full_data_with_cluster', next_version)
    # clus_level_eval
    save_csv_file(CLUS_LEVEL_EVAL_DIR, clus_level_eval, 'clus_level_eval', next_version)
    # overall_eval
    save_csv_file(OVERALL_EVAL_DIR, overall_eval, 'overall_eval', next_version)
    # cluster_mapping
    if cluster_mapping:
        save_json_file(CLUSTER_MAPPING_DIR, cluster_mapping, 'cluster_mapping', next_version)

    return clus_explain, df, clus_level_eval, overall_eval, cluster_mapping

clus_explain, full_data_with_cluster, clus_level_eval, overall_eval, cluster_mapping = train_approach_1(train_df, None)

print(clus_explain)
print(full_data_with_cluster)
print(clus_level_eval)
print(overall_eval)
print(cluster_mapping)
