from src.data_refresher.clustering.approach_3_embed_num.utils.utils import *
import pandas as pd

def train_approach_3(train_df, emb_dim=None, n_clusters=None):
    df_num, numeric_feature_col = preprocess_num(train_df)
    df_embedding = embedding_creation(train_df)

    if emb_dim == None:
        emb_dim = df_num.shape[1]
    
    data_pca = dim_reduction(df_embedding, emb_dim)

    df_num = df_num.reset_index(drop=True)
    data_pca = data_pca.reset_index(drop=True)

    # concat
    df_combined = pd.concat([df_num, data_pca], axis=1)
    df_combined.columns = df_combined.columns.astype(str)

    df_no_outliers, n_clusters, clusters, next_version, df_combined_no_out = clustering(train_df, df_combined, MODEL_DIR, n_clusters)

    col = 'cluster'

    df_group = cluster_stats_info(df_no_outliers, col)
    print(df_group[[col, 'cluster_size']])

    # Check if clusters need merging
    should_merge, clusters_to_merge = should_merge_clusters(df_group)
    if should_merge:
        col = 'merged_cluster'
        print(f"Merging small clusters: {clusters_to_merge}")
        cluster_mapping = adjust_clusters(df_group, clusters_to_merge)
        # df_group = apply_cluster_mapping(df_group, cluster_mapping)
        df_no_outliers = apply_cluster_mapping(df_no_outliers, cluster_mapping)
        df_group = cluster_stats_info(df_no_outliers, col)
        print(df_group)

    
    # Calculate cluster distances
    # df_combined_no_out = df_combined[df_combined["outliers"] == 0]
    # df_combined_no_out = df_combined_no_out.drop(["outliers"], axis=1)
    clus_level_eval = calculate_cluster_distances(
        df_no_outliers, 
        df_combined_no_out,
        clusters,  # You'll need to return this from your clustering function
        col
    )
    # Calculate overall evaluation metrics
    overall_eval = calculate_overall_distance(clus_level_eval)


    df_group = add_descriptions_simple(df_group, col)
    print(df_group[[col, 'cluster_size', 'description']])

    clus_explain = df_group[[col, 'cluster_size', 'description']].copy()
    # full_data_with_cluster = df_no_outliers.copy()


    # Save results with versioning
    CLUS_EXPLAIN_DIR = RESULTS_DIR + '/clus_explain'
    FULL_DATA_WITH_CLUSTER_DIR = RESULTS_DIR + '/full_data_with_cluster'
    CLUS_LEVEL_EVAL_DIR = RESULTS_DIR + '/clus_level_eval'
    OVERALL_EVAL_DIR = RESULTS_DIR + '/overall_eval'
    CLUSTER_MAPPING_DIR = RESULTS_DIR + '/cluster_mapping'

    # clus_explain
    save_csv_file(CLUS_EXPLAIN_DIR, clus_explain, 'clus_explain', next_version)
    # full_data_with_cluster
    save_csv_file(FULL_DATA_WITH_CLUSTER_DIR, df_no_outliers, 'full_data_with_cluster', next_version)
    # clus_level_eval
    save_csv_file(CLUS_LEVEL_EVAL_DIR, clus_level_eval, 'clus_level_eval', next_version)
    # overall_eval
    save_csv_file(OVERALL_EVAL_DIR, overall_eval, 'overall_eval', next_version)
    # cluster_mapping
    if cluster_mapping:
        save_json_file(CLUSTER_MAPPING_DIR, cluster_mapping, 'cluster_mapping', next_version)

    return clus_explain, df_no_outliers, clus_level_eval, overall_eval, cluster_mapping



if __name__ == "__main__":
    train_df = pd.read_csv('src/data_refresher/data/T0/train_df.csv')
    MODEL_DIR = "src/data_refresher/clustering/approach_3_embed_num/model"
    RESULTS_DIR = "src/data_refresher/clustering/approach_3_embed_num/result"


    clus_explain, full_data_with_cluster, clus_level_eval, overall_eval, cluster_mapping = train_approach_3(train_df, None, 6)

    print(clus_explain)
    print(full_data_with_cluster)
    print(clus_level_eval)
    print(overall_eval)
    print(cluster_mapping)