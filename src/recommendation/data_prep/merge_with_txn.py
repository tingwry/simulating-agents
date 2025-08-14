import pandas as pd

# train_with_lifestyle_T0 = pd.read_csv('src/recommendation/data/T0/train_with_lifestyle.csv')
# train_with_lifestyle_T1 = pd.read_csv('src/recommendation/data/T1/train_with_lifestyle.csv')
train_with_lifestyle_pred_T1 = pd.read_csv('src/recommendation/data/T1_predicted/train_with_lifestyle.csv')

# ans_key_grouped = pd.read_csv('src/recommendation/data/ans_key/grouped_catbased.csv')
ans_key_grouped_no_norm = pd.read_csv('src/recommendation/data/ans_key/grouped_catbased_no_norm.csv')


def merge_demog_with_labels(demog, labels):
    merged_df = pd.merge(
        demog,
        labels,
        left_on='CUST_ID',
        right_on='cust_id',
        how='inner'
    )

    features = ['CUST_ID', 'Number of Children', 'Age', 'Gender', 'Education level', 'Marital status', 'Region', 'Occupation Group', 
                'loan','utility','finance','shopping','financial_services','health_and_care','home_lifestyle','transport_travel','leisure','public_services']
    
    merged_df = merged_df[features]

    return merged_df


merged_data = merge_demog_with_labels(train_with_lifestyle_pred_T1, ans_key_grouped_no_norm)

print(f"Merged DataFrame shape: {merged_data.shape}")
print(merged_data.head())

# merged_data.to_csv('src/recommendation/data/T0/demog_ranking_grouped_catbased_no_norm.csv', index=False)
# merged_data.to_csv('src/recommendation/data/T1/demog_ranking_grouped_catbased_no_norm.csv', index=False)
merged_data.to_csv('src/recommendation/data/T1_predicted/demog_ranking_grouped_catbased_no_norm.csv', index=False)