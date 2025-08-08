import pandas as pd

# train_with_lifestyle = pd.read_csv('src/data/T0/train_with_lifestyle.csv')
test_predicted_T1 = pd.read_csv('src/recommendation/data/T1_predicted/test_predicted_T1.csv')

# ans_key = pd.read_csv('src/data/cf_demog_summary/user_item_matrix/user_item_matrix.csv')
# ans_key_grouped = pd.read_csv('src/data/cf_demog_summary/user_item_matrix/user_item_matrix_grouped.csv')
ans_key_grouped = pd.read_csv('src/recommendation/data/ans_key/grouped_catbased.csv')


def merge_demog_with_labels(demog, labels):
    merged_df = pd.merge(
        demog,
        labels,
        left_on='CUST_ID',
        right_on='cust_id',
        how='inner'
    )
    return merged_df


merged_data = merge_demog_with_labels(test_predicted_T1, ans_key_grouped)

print(f"Merged DataFrame shape: {merged_data.shape}")
print(merged_data.head())

merged_data.to_csv('src/recommendation/data/T1_predicted/pred_demog_ranking_grouped_catbased.csv', index=False)