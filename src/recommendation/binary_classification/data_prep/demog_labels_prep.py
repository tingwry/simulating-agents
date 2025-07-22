import pandas as pd

train_with_lifestyle = pd.read_csv('src/data/T0/train_with_lifestyle.csv')
ans_key = pd.read_csv('src/recommendation/cluster_based/eval/ans_key.csv')


def merge_demog_with_labels(demog, labels):
    merged_df = pd.merge(
        demog,
        labels,
        left_on='CUST_ID',
        right_on='cust_id',
        how='inner'
    )
    return merged_df


merged_data = merge_demog_with_labels(train_with_lifestyle, ans_key)

print(f"Merged DataFrame shape: {merged_data.shape}")
print(merged_data.head())

merged_data.to_csv('src/recommendation/binary_classification/data/demog_labels.csv', index=False)