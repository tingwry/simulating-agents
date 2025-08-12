import pandas as pd

# train_with_lifestyle_T0 = pd.read_csv('src/data/T0/train_with_lifestyle.csv')
train_with_lifestyle_T1 = pd.read_csv('src/recommendation/data/T1/train_with_lifestyle.csv')
# train_with_lifestyle_pred_T1 = pd.read_csv('src/recommendation/data/T1_predicted/train_with_lifestyle.csv')

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


merged_data = merge_demog_with_labels(train_with_lifestyle_T1, ans_key_grouped)

print(f"Merged DataFrame shape: {merged_data.shape}")
print(merged_data.head())

merged_data.to_csv('src/recommendation/data/T1/T1_demog_ranking_grouped_catbased.csv', index=False)