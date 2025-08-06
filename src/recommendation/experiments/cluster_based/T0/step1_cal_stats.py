import pandas as pd

warm_cust_df = pd.read_csv('src/clustering/approach_2_embed/result/full_data_with_cluster/full_data_with_cluster_v4.csv')
lifestyle = pd.read_csv('src/data/preprocessed_data/lifestyle.csv')

def cluster_stats_info(lifestyle, full_data_with_cluster, clus_id, txn_cat):
    """
    Calculate demographic statistics for customers in a specific cluster who have performed transactions in a given category.
    """
    
    filtered_clus = full_data_with_cluster[full_data_with_cluster['cluster'] == clus_id]
    all_cust_in_clus = filtered_clus['CUST_ID'].unique()
    
    filtered_lifestyle = lifestyle[(lifestyle['txn_catg'] == txn_cat) & 
                                 (lifestyle['cust_id'].isin(all_cust_in_clus))]
    
    grouped_lifestyle = filtered_lifestyle.groupby('cust_id').agg({
        'dpst_txn_amt_tot': 'sum',
        'dpst_txn_cnt': 'sum'
    }).reset_index()
    
    merged_df = pd.merge(
        grouped_lifestyle,
        filtered_clus,
        left_on='cust_id',
        right_on='CUST_ID',
        how='inner'
    )

    stats_data = {
        'cluster_id': [clus_id],
        'transaction_category': [txn_cat],
        'customer_count': [len(merged_df)],
        'total_transaction_amount': [merged_df['dpst_txn_amt_tot'].sum()],
        'total_transaction_count': [merged_df['dpst_txn_cnt'].sum()],
        'avg_transaction_amount': [merged_df['dpst_txn_amt_tot'].mean()],
        'avg_transaction_count': [merged_df['dpst_txn_cnt'].mean()],
        'avg_age': [merged_df['Age'].mean()],
        'avg_number_of_children': [merged_df['Number of Children'].mean()],
        'avg_transactions': [merged_df['Deposit Account Transactions'].mean()],
    }
    
    # Add value counts for categorical variables (top 3 for each)
    categorical_cols = ['Education level', 'Marital status', 'Region', 'Occupation Group']
    for col in categorical_cols:
        # value_counts = merged_df[col].value_counts(normalize=True)
        value_counts = merged_df[col].value_counts(normalize=True).head(3)
        for i, (val, pct) in enumerate(value_counts.items(), 1):
            stats_data[f'{col.lower().replace(" ", "_")}_top_{i}'] = [val]
            stats_data[f'{col.lower().replace(" ", "_")}_top_{i}_pct'] = [pct]
    
    stats_df = pd.DataFrame(stats_data)
    
    return stats_df

def generate_all_cluster_stats(lifestyle, full_data_with_cluster, txn_cat_list):
    """
    Generate statistics for all combinations of clusters and transaction categories.
    """
    cluster_sizes = full_data_with_cluster['cluster'].value_counts().to_dict()
    cluster_ids = full_data_with_cluster['cluster'].unique()
    
    all_stats = []
    
    for clus_id in cluster_ids:
        for txn_cat in txn_cat_list:
            try:
                stats_df = cluster_stats_info(lifestyle, full_data_with_cluster, clus_id, txn_cat)
                # Add total customers in cluster to the stats
                stats_df['total_in_cluster'] = cluster_sizes.get(clus_id, 0)
                all_stats.append(stats_df)
            except Exception as e:
                print(f"Error processing cluster {clus_id}, category {txn_cat}: {str(e)}")
    
    return pd.concat(all_stats, ignore_index=True)



txn_cat = ['charity', 'loan', 'utility', 'investment', 'finance', 'shopping',
           'personal_care', 'medical', 'home_and_living', 'insurance', 'automotive',
           'restaurant', 'business', 'entertainment', 'bank', 'education', 'pet_care',
           'government', 'travel', 'transportation', 'visit', 'system_dpst']

all_cluster_stats = generate_all_cluster_stats(lifestyle, warm_cust_df, txn_cat)
all_cluster_stats.to_csv('src/recommendation/cluster_based/T0/data/cluster_transaction_stats.csv', index=False)

print(all_cluster_stats.head())