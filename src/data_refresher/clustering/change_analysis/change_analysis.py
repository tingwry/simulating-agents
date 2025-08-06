from src.data_refresher.clustering.change_analysis.utils.utils import *
import pandas as pd
import numpy as np


def change_analysis(train_T1, full_data_with_cluster, train_T0_df_group, save_path):
    print(full_data_with_cluster.head())
    print(full_data_with_cluster['cluster'].dtype)
    original_dtype = full_data_with_cluster['cluster'].dtype
    cluster_mapping = full_data_with_cluster.dropna(subset=['cluster']) \
                                         .set_index('CUST_ID')['cluster'] \
                                         .to_dict()
    
    train_T1['cluster'] = train_T1['CUST_ID'].map(cluster_mapping)
    train_T1 = train_T1.dropna(subset=['cluster'])
    train_T1['cluster'] = train_T1['cluster'].astype(original_dtype)
    
    # preprocess
    train_T1 = train_T1.replace('Unknown', np.nan)
    categorical_columns = ['Gender', 'Education level', 'Marital status', 'Region', 'Occupation Group']
    for col in train_T1.columns:
        if col not in categorical_columns and col != 'CUST_ID' and col != 'cluster':
            train_T1[col] = train_T1[col].astype(float)
        else:
            train_T1[col] = train_T1[col].astype('category')
    print(train_T1.head())


    col = 'cluster'

    train_T1_df_group = cluster_stats_info(train_T1, col)
    print(train_T1_df_group)

    change_analysis_df, differentiators = add_llm_descriptions_to_clusters(train_T1_df_group, train_T0_df_group, col)

    
    save_csv_file(save_path, change_analysis_df, 'change_analysis')

    return change_analysis_df


if __name__ == "__main__":
    train_T1 = pd.read_csv('src/data_refresher/data/T1/train_T1_v3.csv')
    full_data_with_cluster = pd.read_csv('src/data_refresher/clustering/approach_2_embed/result/full_data_with_cluster/full_data_with_cluster_v4.csv')
    train_T0_df_group = pd.read_csv('src/data_refresher/clustering/approach_2_embed/result/clus_explain/clus_explain_v4.csv')
    save_path = 'src/data_refresher/clustering/change_analysis/result'

    print(change_analysis(train_T1, full_data_with_cluster, train_T0_df_group, save_path).head())