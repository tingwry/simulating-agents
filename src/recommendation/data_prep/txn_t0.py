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
    
    # Rename columns to add _t0 suffix
    pivot_df.columns = [f"{col}_t0" if col != 'cust_id' else col for col in pivot_df.columns]
    
    # Merge with demographic data
    result_df = demog_df.merge(pivot_df, left_on='CUST_ID', right_on='cust_id', how='left')
    
    # Drop the cust_id column from merge
    result_df = result_df.drop('cust_id', axis=1)
    
    # Identify transaction columns (those ending with _t0)
    transaction_cols = [col for col in result_df.columns if col.endswith('_t0')]
    
    # Drop rows where ANY transaction column has null values
    result_df = result_df.dropna(subset=transaction_cols, how='any')
    
    return result_df

# Usage example:
def main():
    # Load your data
    lifestyle_df = pd.read_csv('src/data_refresher/data/preprocessed_data/lifestyle_t0.csv')
    # demog_df = pd.read_csv('src/recommendation/data/T0/demog_ranking_grouped_catbased_no_norm.csv')
    demog_df_t1 = pd.read_csv('src/recommendation/data/T1/demog_ranking_grouped_catbased_no_norm.csv')
    
    # Process the data
    result_df = add_transaction_categories_to_demographics(lifestyle_df, demog_df_t1)
    
    # Save or use the result
    result_df.to_csv('src/recommendation/data/T1/demog_ranking_grouped_catbased_no_norm_t0.csv', index=False)
    print("Processing completed!")
    print(f"Original demographic rows: {len(demog_df_t1)}")
    print(f"Result rows after processing: {len(result_df)}")
    print(f"Rows dropped due to null transaction values: {len(demog_df_t1) - len(result_df)}")
    
    return result_df

if __name__ == "__main__":
    main()