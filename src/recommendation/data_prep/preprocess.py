import pandas as pd



def dropna(lifestyle, outputpath):
    print(len(lifestyle['cust_id'].unique()))

    lifestyle = lifestyle.dropna()
    lifestyle.to_csv(outputpath, index=False)
    print(lifestyle.head())

    print(len(lifestyle['cust_id'].unique()))

def df_with_lifestyle(df, lifestyle, filename):
    result = df.copy()
    cust_id_ls = lifestyle['cust_id'].unique()
    result = result[result['CUST_ID'].isin(cust_id_ls)]

    result.to_csv(filename, index=False)
    return result



def filter_matching_customers(df_t0_path, df_t1_path, df_t1_predicted_path):
    """
    Returns versions of the input DataFrames containing only rows with matching CUST_ID across all three.
    
    Parameters:
    df_t0 (pd.DataFrame): DataFrame for time period T0
    df_t1 (pd.DataFrame): DataFrame for time period T1
    df_t1_predicted (pd.DataFrame): Predicted values for time period T1
    
    Returns:
    tuple: (df_t0_filtered, df_t1_filtered, df_t1_predicted_filtered)
    """
    df_t0 = pd.read_csv(df_t0_path)
    df_t1 = pd.read_csv(df_t1_path)
    df_t1_predicted = pd.read_csv(df_t1_predicted_path)
    # Find intersection of CUST_IDs across all three DataFrames
    common_ids = set(df_t0['CUST_ID'])\
                .intersection(set(df_t1['CUST_ID']))\
                .intersection(set(df_t1_predicted['CUST_ID']))
    
    # Filter each DataFrame to only include these common IDs
    df_t0_filtered = df_t0[df_t0['CUST_ID'].isin(common_ids)]
    df_t1_filtered = df_t1[df_t1['CUST_ID'].isin(common_ids)]
    df_t1_predicted_filtered = df_t1_predicted[df_t1_predicted['CUST_ID'].isin(common_ids)]
    
    df_t0_filtered = df_t0_filtered.reset_index(drop=True)
    df_t1_filtered = df_t1_filtered.reset_index(drop=True)
    df_t1_predicted_filtered = df_t1_predicted_filtered.reset_index(drop=True)

    df_t0_filtered.to_csv(df_t0_path, index=False)
    df_t1_filtered.to_csv(df_t1_path, index=False)
    df_t1_predicted_filtered.to_csv(df_t1_predicted_path, index=False)
    
    return df_t0_filtered, df_t1_filtered, df_t1_predicted_filtered

def add_transaction_categories_to_demographics(lifestyle_df, demog_df, category_mapping=None):
    """
    Group transaction categories and add them as columns to demographic data
    
    Args:
        lifestyle_df: DataFrame from lifestyle_t0.csv
        demog_df: DataFrame from demog_ranking_group.csv
        category_mapping: Dictionary mapping subcategories to main categories
    
    Returns:
        DataFrame: Updated demographic DataFrame with transaction category columns
    """
    
    # Default category mapping
    if category_mapping is None:
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
    
    # Define all the expected final categories we want to keep
    expected_categories = [
        'loan', 'utility', 'finance', 'shopping', 'financial_services',
        'health_and_care', 'home_lifestyle', 'transport_travel',
        'leisure', 'public_services'
    ]
    
    # Create a copy to avoid modifying original data
    lifestyle_processed = lifestyle_df.copy()
    
    # Map subcategories to main categories
    lifestyle_processed['mapped_category'] = lifestyle_processed['txn_subcatg'].map(category_mapping)
    
    # For subcategories not in mapping, use the main category
    lifestyle_processed['mapped_category'] = lifestyle_processed['mapped_category'].fillna(
        lifestyle_processed['txn_catg']
    )
    
    # Group by customer ID and mapped category, sum the transaction amounts
    grouped_transactions = lifestyle_processed.groupby(['cust_id', 'mapped_category'])['dpst_txn_amt_tot'].sum().reset_index()
    
    # Pivot the data to have categories as columns
    pivot_df = grouped_transactions.pivot_table(
        index='cust_id', 
        columns='mapped_category', 
        values='dpst_txn_amt_tot', 
        fill_value=0
    ).reset_index()
    
    # Ensure all expected categories are present in the pivot table (add them with NaN values if missing)
    for category in expected_categories:
        if category not in pivot_df.columns:
            pivot_df[category] = pd.NA
    
    # Rename columns to add _t0 suffix - use the mapped category names
    pivot_df.columns = [f"{col}_t0" if col != 'cust_id' else col for col in pivot_df.columns]
    
    # Merge with demographic data
    result_df = demog_df.merge(pivot_df, left_on='CUST_ID', right_on='cust_id', how='left')
    
    # Drop the cust_id column from merge
    result_df = result_df.drop('cust_id', axis=1)
    
    # Now handle the original columns that might exist in demog_df
    # We need to sum them into the grouped categories before dropping them
    
    # Create a reverse mapping to know which original columns belong to which grouped category
    reverse_mapping = {}
    for original_cat, grouped_cat in category_mapping.items():
        if grouped_cat in expected_categories:
            original_col = f"{original_cat}_t0"
            grouped_col = f"{grouped_cat}_t0"
            if original_col not in reverse_mapping:
                reverse_mapping[original_col] = []
            reverse_mapping[original_col].append(grouped_col)
    
    # For each original column that exists in the result_df, add its value to the grouped column
    for original_col, grouped_cols in reverse_mapping.items():
        if original_col in result_df.columns:
            for grouped_col in grouped_cols:
                if grouped_col in result_df.columns:
                    # Only add if both values are not null
                    mask = result_df[original_col].notna() & result_df[grouped_col].notna()
                    result_df.loc[mask, grouped_col] = result_df.loc[mask, grouped_col] + result_df.loc[mask, original_col]
                    
                    # If grouped column is null but original column has value, use original column value
                    mask_grouped_null = result_df[grouped_col].isna() & result_df[original_col].notna()
                    result_df.loc[mask_grouped_null, grouped_col] = result_df.loc[mask_grouped_null, original_col]
    
    # Now drop all the original columns that were grouped
    columns_to_drop = []
    for original_cat, grouped_cat in category_mapping.items():
        original_col = f"{original_cat}_t0"
        if original_col in result_df.columns and original_col not in [f"{cat}_t0" for cat in expected_categories]:
            columns_to_drop.append(original_col)
    
    result_df = result_df.drop(columns=columns_to_drop, errors='ignore')
    
    # Ensure all expected _t0 columns are present (add them with NaN if missing)
    expected_t0_cols = [f"{cat}_t0" for cat in expected_categories]
    for col in expected_t0_cols:
        if col not in result_df.columns:
            result_df[col] = pd.NA
    
    # Identify transaction columns (those ending with _t0 from our expected list)
    transaction_cols = [col for col in result_df.columns if col.endswith('_t0') and col in expected_t0_cols]
    
    # Drop rows where ANY transaction column has null values
    result_df = result_df.dropna(subset=transaction_cols, how='any')
    
    return result_df

def create_binary_t0_columns(demog_grouped_path, demog_t0_path, output_path):
    """
    Create a new file with binary _t0 columns from demog_t0.csv, keeping only matching IDs
    
    Args:
        demog_grouped_path: Path to demog_grouped_catbased.csv
        demog_t0_path: Path to demog_t0.csv
        output_path: Path to save the output file
    """
    
    # Load the data
    demog_grouped = pd.read_csv(demog_grouped_path)
    demog_t0 = pd.read_csv(demog_t0_path)
    
    # Get the list of expected _t0 columns (all columns ending with _t0)
    t0_columns = [col for col in demog_t0.columns if col.endswith('_t0')]
    
    # Filter demog_t0 to only include matching CUST_IDs
    matching_ids = demog_t0['CUST_ID'].isin(demog_grouped['CUST_ID'])
    demog_t0_filtered = demog_t0[matching_ids].copy()
    
    # Create binary columns (1 if value > 0, else 0)
    for t0_col in t0_columns:
        binary_col = t0_col  # Keep the same column name
        demog_t0_filtered[binary_col] = (demog_t0_filtered[t0_col] > 0).astype(int)
    
    # Merge with the original demog_grouped data to ensure we keep all its columns
    result_df = demog_grouped.merge(
        demog_t0_filtered[['CUST_ID'] + t0_columns], 
        on='CUST_ID', 
        how='inner'
    )
    
    # Save the result
    result_df.to_csv(output_path, index=False)
    
    print(f"Processing completed!")
    print(f"Original demog_grouped rows: {len(demog_grouped)}")
    print(f"Matching rows in demog_t0: {len(demog_t0_filtered)}")
    print(f"Final result rows: {len(result_df)}")
    print(f"New binary columns added: {t0_columns}")
    
    return result_df
    

if __name__ == "__main__":
    raw_lifestyle = pd.read_csv('src/recommendation/data/raw_data/customerllm_lifestyle.csv')
    raw_lifestyle_t0 = pd.read_csv('src/recommendation/data/raw_data/customerllm_y_t0.csv')
    dropna(raw_lifestyle, 'src/recommendation/data/preprocessed_data/lifestyle.csv')
    dropna(raw_lifestyle_t0, 'src/recommendation/data/preprocessed_data/lifestyle_t0.csv')

    lifestyle = pd.read_csv('src/recommendation/data/preprocessed_data/lifestyle.csv')

    # T0
    train_t0 = pd.read_csv('src/data_refresher/data/T0/train_df.csv')
    test_t0 = pd.read_csv('src/data_refresher/data/T0/test_df.csv')

    train_with_lifestyle_t0 = df_with_lifestyle(train_t0, lifestyle, 'src/recommendation/data/T0/train_with_lifestyle.csv')
    test_with_lifestyle_t0 = df_with_lifestyle(test_t0, lifestyle, 'src/recommendation/data/T0/test_with_lifestyle.csv')

    # T1
    train_t1 = pd.read_csv('src/data_refresher/data/T1/train_T1.csv')
    test_t1 = pd.read_csv('src/data_refresher/data/T1/test_T1_actual.csv')

    train_with_lifestyle_t1 = df_with_lifestyle(train_t1, lifestyle, 'src/recommendation/data/T1/train_with_lifestyle.csv')
    test_with_lifestyle_t1 = df_with_lifestyle(test_t1, lifestyle, 'src/recommendation/data/T1/test_with_lifestyle.csv')

    # test_with_lifestyle_t1['Age'] = test_with_lifestyle_t1['Age'] + 2
    # test_with_lifestyle_t1.to_csv('src/recommendation/data/T1/test_with_lifestyle.csv', index=False)


    # train

    train_t0_path = 'src/recommendation/data/T0/demog_ranking_grouped_catbased_no_norm.csv'
    train_t1_path = 'src/recommendation/data/T1/demog_ranking_grouped_catbased_no_norm.csv'
    train_t1_predicted_path = 'src/recommendation/data/T1_predicted/demog_ranking_grouped_catbased_no_norm_single.csv'
    # train_t0_path = 'src/recommendation/data/T0/demog_grouped_catbased.csv'
    # train_t1_path = 'src/recommendation/data/T1/demog_grouped_catbased.csv'
    # train_t1_predicted_path = 'src/recommendation/data/T1_predicted/demog_grouped_catbased.csv'

    df_t0_filtered, df_t1_filtered, df_t1_predicted_filtered = filter_matching_customers(train_t0_path, train_t1_path, train_t1_predicted_path)


    # test
    test_t0_path = 'src/recommendation/data/T0/test_with_lifestyle.csv'
    test_t1_path = 'src/recommendation/data/T1/test_with_lifestyle.csv'
    test_t1_predicted_path = 'src/recommendation/data/T1_predicted/test_with_lifestyle_single.csv'

    df_t0_filtered, df_t1_filtered, df_t1_predicted_filtered = filter_matching_customers(test_t0_path, test_t1_path, test_t1_predicted_path)


    # Add txn at T0
    lifestyle_df = pd.read_csv('src/recommendation/data/preprocessed_data/lifestyle_t0.csv')

    train_demog_df = pd.read_csv('src/recommendation/data/T0/demog_ranking_grouped_catbased_no_norm.csv')
    result_df = add_transaction_categories_to_demographics(lifestyle_df, train_demog_df)
    result_df.to_csv('src/recommendation/data/T0/demog_ranking_grouped_catbased_no_norm_t0.csv', index=False)
    print("Processing completed!")
    print(f"Original demographic rows: {len(train_demog_df)}")
    print(f"Result rows after processing: {len(result_df)}")
    print(f"Rows dropped due to null transaction values: {len(train_demog_df) - len(result_df)}")

    train_demog_df_t1 = pd.read_csv('src/recommendation/data/T1/demog_ranking_grouped_catbased_no_norm.csv')
    result_df = add_transaction_categories_to_demographics(lifestyle_df, train_demog_df_t1)
    result_df.to_csv('src/recommendation/data/T1/demog_ranking_grouped_catbased_no_norm_t0.csv', index=False)
    print("Processing completed!")
    print(f"Original demographic rows: {len(train_demog_df_t1)}")
    print(f"Result rows after processing: {len(result_df)}")
    print(f"Rows dropped due to null transaction values: {len(train_demog_df_t1) - len(result_df)}")

    test = pd.read_csv('src/recommendation/data/T0/test_with_lifestyle.csv')
    result_df = add_transaction_categories_to_demographics(lifestyle_df, test)
    result_df.to_csv('src/recommendation/data/T0/test_with_lifestyle_t0.csv', index=False)
    print("Processing completed!")
    print(f"Original demographic rows: {len(test)}")
    print(f"Result rows after processing: {len(result_df)}")
    print(f"Rows dropped due to null transaction values: {len(test) - len(result_df)}")



    # Define file paths
    demog_grouped_path = 'src/recommendation/data/T0/demog_grouped_catbased.csv'
    demog_t0_path = 'src/recommendation/data/T0/demog_ranking_grouped_catbased_no_norm_t0.csv'
    output_path = 'src/recommendation/data/T0/demog_grouped_catbased_t0.csv'
    
    # Create the new file
    result_df = create_binary_t0_columns(demog_grouped_path, demog_t0_path, output_path)