import pandas as pd
import numpy as np
from sklearn.utils import resample

# binary (0/1)
def binary_ans_key_prep():
    user_item_matrix = pd.read_csv('src/data/cf_demog_summary/user_item_matrix/user_item_matrix.csv')

    user_item_matrix_binary = user_item_matrix.copy()
    user_item_matrix_binary.iloc[:, 1:] = np.where(user_item_matrix.iloc[:, 1:] > 0, 1, 0)


    output_path = 'src/recommendation/cluster_based/eval/ans_key.csv'
    user_item_matrix_binary.to_csv(output_path, index=False)




lifestyle = pd.read_csv('src/data/preprocessed_data/lifestyle.csv')
train_T0_demog_summ = pd.read_csv('src/data/cf_demog_summary/train_T0_demog_summ.csv/train_T0_demog_summ.csv')


def group_and_normalize_categories(df, group_into_other=True):
    # Categories to group into 'other' (if enabled)
    other_categories = [
        'charity', 'investment', 'personal_care', 'medical', 'home_and_living',
        'insurance', 'automotive', 'restaurant', 'business', 'entertainment',
        'bank', 'education', 'pet_care', 'government', 'travel',
        'transportation', 'visit', 'system_dpst'
    ]
    
    # Create a copy to avoid modifying the original DataFrame
    df = df.copy()
    
    if group_into_other:
        # Create 'other' column by summing counts for specified categories
        df['txn_catg'] = df['txn_catg'].apply(
            lambda x: 'other' if x in other_categories else x
        )
    
    # Group by customer and category, sum transaction counts
    grouped = df.groupby(['cust_id', 'txn_catg'])['dpst_txn_cnt'].sum().unstack(fill_value=0)
    
    # Normalize each row (customer) so transaction counts sum to 1
    normalized = grouped.div(grouped.sum(axis=1), axis=0)
    
    # Columns we want to keep in the final output
    final_columns = [
        'loan', 'utility', 'finance', 'shopping', 'other'
    ] if group_into_other else [
        'charity', 'loan', 'utility', 'investment', 'finance', 'shopping',
        'personal_care', 'medical', 'home_and_living', 'insurance', 'automotive',
        'restaurant', 'business', 'entertainment', 'bank', 'education', 'pet_care',
        'government', 'travel', 'transportation', 'visit', 'system_dpst'
    ]
    
    # Ensure all required columns are present (fill missing with 0)
    for col in final_columns:
        if col not in normalized.columns:
            normalized[col] = 0.0
    
    # Select only the columns we want to keep
    normalized = normalized[final_columns]
    
    # Reset index to make cust_id a column
    normalized.reset_index(inplace=True)
    
    return normalized


user_item_matrix = group_and_normalize_categories(lifestyle)
print(user_item_matrix)
user_item_matrix.to_csv('src/data/cf_demog_summary/user_item_matrix/user_item_matrix_grouped.csv', index=False)


def see_col_freq():
    ans_key = pd.read_csv('src/recommendation/cluster_based/eval/ans_key.csv')
    columns = ans_key.columns
    print(columns)
    for col in columns:
        print(col, ans_key[ans_key[col] == 1][col].count(), ":", ans_key[ans_key[col] == 0][col].count())


def group_cat():
    ans_key = pd.read_csv('src/recommendation/cluster_based/eval/ans_key.csv')
    columns = ['charity', 'investment', 'personal_care', 
       'automotive', 'business', 'entertainment', 'bank',
       'education', 'pet_care', 'government', 'travel', 'transportation',
       'visit', 'system_dpst']
    df = ans_key.copy()
    df['severe'] = df['charity']
    for col in columns:
        df['severe'] += df[col]


        # df_1 = ans_key[ans_key['label'] == 1]
        # df_0 = ans_key[ans_key['label'] == 0]

    #     df_1_upsampled = resample(df_1, replace=True, n_samples=len(df_0), random_state=42)
    # df_balanced = pd.concat([df_1_upsampled, df_0])

