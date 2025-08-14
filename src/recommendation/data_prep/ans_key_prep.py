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


def group_and_normalize_ratio(df, group_into_other=True):
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


def group_and_normalize_categories(df, group_custom=True, normalize_by='column'):
    """
    Groups categories according to custom rules and normalizes transaction counts.
    
    Parameters:
    - df: Input DataFrame with transaction data
    - group_custom: Whether to apply custom grouping (default True)
    - normalize_by: 'row' (per customer) or 'column' (per category) normalization
    
    Returns:
    - DataFrame with normalized transaction counts
    """
    # Create a copy to avoid modifying the original DataFrame
    df = df.copy()
    
    if group_custom:
        # Define category mapping for custom grouping
        category_mapping = {
            # Financial services group
            'investment': 'financial_services',
            'insurance': 'financial_services',
            'bank': 'financial_services',
            'business': 'financial_services',
            
            # Health and care group
            'personal_care': 'health_and_care',
            'medical': 'health_and_care',
            'pet_care': 'health_and_care',
            
            # Home lifestyle group
            'home_and_living': 'home_lifestyle',
            'education': 'home_lifestyle',
            'visit': 'home_lifestyle',
            
            # Transport and travel group
            'automotive': 'transport_travel',
            'transportation': 'transport_travel',
            'travel': 'transport_travel',
            
            # Leisure group
            'restaurant': 'leisure',
            'entertainment': 'leisure',
            
            # Public services group
            'charity': 'public_services',
            'government': 'public_services',
            
            # System deposits (keep as-is)
            'system_dpst': 'system_dpst'
        }
        
        # Apply the category mapping
        df['txn_catg'] = df['txn_catg'].map(category_mapping).fillna(df['txn_catg'])
    
    # Group by customer and category, sum transaction counts
    # grouped = df.groupby(['cust_id', 'txn_catg'])['dpst_txn_cnt'].sum().unstack(fill_value=0)
    grouped = df.groupby(['cust_id', 'txn_catg'])['dpst_txn_amt_tot'].sum().unstack(fill_value=0)
    
    # Apply normalization
    if normalize_by == 'row':
        # Normalize each row (customer) so transaction counts sum to 1
        normalized = grouped.div(grouped.sum(axis=1), axis=0)
    elif normalize_by == 'column':
        # Normalize each column (category) so values sum to 1
        normalized = grouped.div(grouped.sum(axis=0), axis=1)
    else:
        raise ValueError("normalize_by must be either 'row' or 'column'")
    
    # Define final columns (core categories + new grouped categories)
    core_categories = ['loan', 'utility', 'finance', 'shopping']
    new_grouped_categories = [
        'financial_services', 'health_and_care', 'home_lifestyle',
        'transport_travel', 'leisure', 'public_services', 'system_dpst'
    ]
    
    final_columns = core_categories + new_grouped_categories if group_custom else [
        'charity', 'loan', 'utility', 'investment', 'finance', 'shopping',
        'personal_care', 'medical', 'home_and_living', 'insurance', 'automotive',
        'restaurant', 'business', 'entertainment', 'bank', 'education', 'pet_care',
        'government', 'travel', 'transportation', 'visit', 'system_dpst'
    ]
    
    # Ensure all required columns are present (fill missing with 0)
    for col in final_columns:
        if col not in normalized.columns:
            normalized[col] = 0.0
    
    # Select and order columns
    normalized = normalized[final_columns]
    
    # Reset index to make cust_id a column
    normalized.reset_index(inplace=True)
    
    return normalized

def group_categories_without_normalization(df, group_custom=True):
    """
    Groups categories according to custom rules without normalizing transaction counts.
    
    Parameters:
    - df: Input DataFrame with transaction data
    - group_custom: Whether to apply custom grouping (default True)
    
    Returns:
    - DataFrame with raw transaction counts per customer and category
    """
    # Create a copy to avoid modifying the original DataFrame
    df = df.copy()
    
    if group_custom:
        # Define category mapping for custom grouping
        category_mapping = {
            # Financial services group
            'investment': 'financial_services',
            'insurance': 'financial_services',
            'bank': 'financial_services',
            'business': 'financial_services',
            
            # Health and care group
            'personal_care': 'health_and_care',
            'medical': 'health_and_care',
            'pet_care': 'health_and_care',
            
            # Home lifestyle group
            'home_and_living': 'home_lifestyle',
            'education': 'home_lifestyle',
            'visit': 'home_lifestyle',
            
            # Transport and travel group
            'automotive': 'transport_travel',
            'transportation': 'transport_travel',
            'travel': 'transport_travel',
            
            # Leisure group
            'restaurant': 'leisure',
            'entertainment': 'leisure',
            
            # Public services group
            'charity': 'public_services',
            'government': 'public_services',
            
            # System deposits (keep as-is)
            'system_dpst': 'system_dpst'
        }
        
        # Apply the category mapping
        df['txn_catg'] = df['txn_catg'].map(category_mapping).fillna(df['txn_catg'])
    
    # Group by customer and category, sum transaction counts (no normalization)
    # grouped = df.groupby(['cust_id', 'txn_catg'])['dpst_txn_cnt'].sum().unstack(fill_value=0)
    grouped = df.groupby(['cust_id', 'txn_catg'])['dpst_txn_amt_tot'].sum().unstack(fill_value=0)
    
    # Define final columns (core categories + new grouped categories)
    core_categories = ['loan', 'utility', 'finance', 'shopping']
    new_grouped_categories = [
        'financial_services', 'health_and_care', 'home_lifestyle',
        'transport_travel', 'leisure', 'public_services', 'system_dpst'
    ]
    
    final_columns = core_categories + new_grouped_categories if group_custom else [
        'charity', 'loan', 'utility', 'investment', 'finance', 'shopping',
        'personal_care', 'medical', 'home_and_living', 'insurance', 'automotive',
        'restaurant', 'business', 'entertainment', 'bank', 'education', 'pet_care',
        'government', 'travel', 'transportation', 'visit', 'system_dpst'
    ]
    
    # Ensure all required columns are present (fill missing with 0)
    for col in final_columns:
        if col not in grouped.columns:
            grouped[col] = 0
    
    # Select and order columns
    grouped = grouped[final_columns]
    
    # Reset index to make cust_id a column
    grouped.reset_index(inplace=True)
    
    return grouped



def see_col_freq():
    ans_key = pd.read_csv('src/recommendation/cluster_based/eval/ans_key.csv')
    columns = ans_key.columns
    print(columns)
    for col in columns:
        print(col, ans_key[ans_key[col] == 1][col].count(), ":", ans_key[ans_key[col] == 0][col].count())


if __name__ == "__main__":
    lifestyle = pd.read_csv('src/data_refresher/data/preprocessed_data/lifestyle.csv')
    # train_T0_demog_summ = pd.read_csv('src/data/cf_demog_summary/train_T0_demog_summ.csv/train_T0_demog_summ.csv')

    # user_item_matrix = group_and_normalize_categories(lifestyle, group_custom=True, normalize_by='row') 
    # print(user_item_matrix)
    # user_item_matrix.to_csv('src/recommendation/data/ans_key/grouped_catbased_amt.csv', index=False)

    # user_item_matrix = group_and_normalize_categories(lifestyle, group_custom=True, normalize_by='column') 
    # print(user_item_matrix)
    # user_item_matrix.to_csv('src/recommendation/data/ans_key/grouped_catbased_amt_by_column.csv', index=False)

    # user_item_matrix = group_categories_without_normalization(lifestyle)
    # print(user_item_matrix)
    # user_item_matrix.to_csv('src/recommendation/data/ans_key/grouped_catbased_amt_no_norm.csv', index=False)


    # Assuming df is your DataFrame
    df = pd.read_csv('src/recommendation/data/T1_predicted/test_with_lifestyle.csv')
    # df = df.sort_values(by='CUST_ID', ascending=True)
    # print(df.head(10))

    # If you want to reset the index after sorting (optional)
    df = df.sort_values(by='CUST_ID', ascending=True).reset_index(drop=True)
    print(df.head(10))
    df.to_csv('src/recommendation/data/T1_predicted/test_with_lifestyle.csv', index=False)