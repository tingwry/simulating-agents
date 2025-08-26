import pandas as pd

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

# Usage example:
def main():
    # Load your data
    lifestyle_df = pd.read_csv('src/data_refresher/data/preprocessed_data/lifestyle_t0.csv')
    # demog_df = pd.read_csv('src/recommendation/data/T0/demog_ranking_grouped_catbased_no_norm.csv')
    # demog_df_t1 = pd.read_csv('src/recommendation/data/T1/demog_ranking_grouped_catbased_no_norm.csv')
    test = pd.read_csv('src/recommendation/data/T0/test_with_lifestyle.csv')
    
    # Process the data
    result_df = add_transaction_categories_to_demographics(lifestyle_df, test)
    
    # Save or use the result
    # result_df.to_csv('src/recommendation/data/T0/demog_ranking_grouped_catbased_no_norm_t0.csv', index=False)
    # result_df.to_csv('src/recommendation/data/T1/demog_ranking_grouped_catbased_no_norm_t0.csv', index=False)
    result_df.to_csv('src/recommendation/data/T0/test_with_lifestyle_t0.csv', index=False)
    print("Processing completed!")
    print(f"Original demographic rows: {len(test)}")
    print(f"Result rows after processing: {len(result_df)}")
    print(f"Rows dropped due to null transaction values: {len(test) - len(result_df)}")
    
    # Show the new column names
    transaction_cols = [col for col in result_df.columns if col.endswith('_t0')]
    print(f"\nTransaction columns after processing: {transaction_cols}")
    
    # Show some sample data to verify the summing worked
    print("\nSample data (first 5 rows):")
    print(result_df[transaction_cols].head())
    
    return result_df

if __name__ == "__main__":
    main()