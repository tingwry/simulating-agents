import pandas as pd

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

# Usage example:
def main():
    # Define file paths
    demog_grouped_path = 'src/recommendation/data/T0/demog_grouped_catbased.csv'
    demog_t0_path = 'src/recommendation/data/T0/demog_ranking_grouped_catbased_no_norm_t0.csv'
    output_path = 'src/recommendation/data/T0/demog_grouped_catbased_t0.csv'
    
    # Create the new file
    result_df = create_binary_t0_columns(demog_grouped_path, demog_t0_path, output_path)
    
    # Show sample of the result
    print("\nSample of the result:")
    t0_columns = [col for col in result_df.columns if col.endswith('_t0')]
    print(result_df[['CUST_ID'] + t0_columns].head())
    
    return result_df

if __name__ == "__main__":
    main()