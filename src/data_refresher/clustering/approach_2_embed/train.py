from src.data_refresher.clustering.approach_2_embed.utils.utils import *
import pandas as pd


def train_approach_2(train_df, n_clusters=None):
    df_embedding = embedding_creation(train_df)

    df_no_outliers, n_clusters, clusters, next_version, df_combined_no_out = clustering(train_df, df_embedding, MODEL_DIR, n_clusters)

    col = 'cluster'

    df_group = cluster_stats_info(df_no_outliers, col)
    print(df_group[[col, 'cluster_size']])

    cluster_mapping = None

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

    # clus_explain = df_group[[col, 'cluster_size', 'description']].copy()
    clus_explain = df_group.copy()

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
    MODEL_DIR = "src/data_refresher/clustering/approach_2_embed/model"
    RESULTS_DIR = "src/data_refresher/clustering/approach_2_embed/result"


    clus_explain, full_data_with_cluster, clus_level_eval, overall_eval, cluster_mapping = train_approach_2(train_df, None)

    print(clus_explain)
    print(full_data_with_cluster)
    print(clus_level_eval)
    print(overall_eval)
    print(cluster_mapping)