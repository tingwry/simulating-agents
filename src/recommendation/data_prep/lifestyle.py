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

def keep_demog_col(path):
    df = pd.read_csv(path)
    features = ['CUST_ID', 'Number of Children', 'Age', 'Gender', 'Education level', 'Marital status', 'Region', 'Occupation Group']
                # 'loan','utility','finance','shopping','financial_services','health_and_care','home_lifestyle','transport_travel','leisure','public_services']
    
    df = df[features]
    df.to_csv(path, index=False)
    return df

def clean(path):
    df = pd.read_csv(path)

    df['Education level'] = df['Education level'].replace(
        'vocational certificate/ diploma',
        'vocational certificate'
    )

    df['Marital status'] = df['Marital status'].replace(
        'married - non registered',
        'married'
    )
    df['Marital status'] = df['Marital status'].replace(
        'married - registered',
        'married'
    )

    df['Occupation Group'] = df['Occupation Group'].replace(
        'retire',
        'Retired'
    )

    df.to_csv(path, index=False)

    return df

def remove_nan(path):
    df = pd.read_csv(path)

    print(len(df))

    # Remove rows with 'Unknown' or NaN values
    for column in df.columns:
        # Remove rows with 'Unknown' (case insensitive)
        df = df[~df[column].astype(str).str.lower().eq('unknown')]
        # Remove NaN values
        df = df.dropna(subset=[column])

    df.to_csv(path, index=False)
    print(len(df))

    return df

# def remove_nan(path, show_removed_rows=False):
#     df = pd.read_csv(path)
    
#     # Store original row count
#     original_rows = len(df)
    
#     # Identify rows to remove
#     mask = pd.Series(False, index=df.index)
#     reasons = {}
    
#     for column in df.columns:
#         # Find 'Unknown' values (case insensitive)
#         unknown_mask = df[column].astype(str).str.lower().eq('unknown')
#         # Find NaN values
#         nan_mask = df[column].isna()
#         # Combine masks
#         col_mask = unknown_mask | nan_mask
#         mask = mask | col_mask
        
#         # Record reasons for removal
#         if col_mask.any():
#             # Get the first 3 indices where the mask is True
#             example_indices = col_mask[col_mask].index[:3]
#             examples = df.loc[example_indices, column].tolist()
            
#             reasons[column] = {
#                 'unknown': unknown_mask.sum(),
#                 'nan': nan_mask.sum(),
#                 'examples': examples
#             }

#     # Show removal statistics and examples
#     print(f"\nOriginal row count: {original_rows}")
#     print(f"Rows to remove: {mask.sum()} ({mask.mean():.1%})")
    
#     if show_removed_rows and mask.any():
#         print("\nSample of rows being removed:")
#         print(df[mask].head())
    
#     print("\nBreakdown by column:")
#     for col, stats in reasons.items():
#         print(f"\nColumn: {col}")
#         print(f"- 'Unknown' values: {stats['unknown']}")
#         print(f"- NaN values: {stats['nan']}")
#         print(f"- Example values: {stats['examples']}")

#     # Remove the identified rows
#     cleaned_df = df[~mask].copy()
#     rows_removed = original_rows - len(cleaned_df)
    
#     print(f"\nFinal row count: {len(cleaned_df)}")
#     print(f"Rows removed: {rows_removed}")

#     return cleaned_df


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
    
    # Reset indices (optional)
    df_t0_filtered = df_t0_filtered.reset_index(drop=True)
    df_t1_filtered = df_t1_filtered.reset_index(drop=True)
    df_t1_predicted_filtered = df_t1_predicted_filtered.reset_index(drop=True)

    df_t0_filtered.to_csv(df_t0_path, index=False)
    df_t1_filtered.to_csv(df_t1_path, index=False)
    df_t1_predicted_filtered.to_csv(df_t1_predicted_path, index=False)
    
    return df_t0_filtered, df_t1_filtered, df_t1_predicted_filtered

    

if __name__ == "__main__":
    # lifestyle = pd.read_csv('src/data/raw_data/customerllm_lifestyle.csv')
    # lifestyle = pd.read_csv('src/recommendation/data/raw_data/customerllm_y_t0.csv')
    # dropna(lifestyle, 'src/data_refresher/data/preprocessed_data/lifestyle_t0.csv')

    # lifestyle = pd.read_csv('src/data_refresher/data/preprocessed_data/lifestyle.csv')
    lifestyle = pd.read_csv('src/data_refresher/data/preprocessed_data/lifestyle_t0.csv')

    # T0
    # train = pd.read_csv('src/data_refresher/data/T0/train_df.csv')
    # test = pd.read_csv('src/data_refresher/data/T0/test_df.csv')
    # train_output_filename = 'src/recommendation/data/T0/train_with_lifestyle.csv'
    # test_output_filename = 'src/recommendation/data/T0/test_with_lifestyle.csv'

    # T1
    # train = pd.read_csv('src/data_refresher/data/T1/train_T1_v3.csv')
    # test = pd.read_csv('src/data_refresher/data/T1/test_T1_actual_v3.csv')
    # train_output_filename = 'src/recommendation/data/T1/train_with_lifestyle.csv'
    # test_output_filename = 'src/recommendation/data/T1/test_with_lifestyle.csv'


    # train_with_lifestyle = df_with_lifestyle(train, lifestyle, train_output_filename)
    # test_with_lifestyle = df_with_lifestyle(test, lifestyle, test_output_filename)



    # print(keep_demog_col('src/recommendation/data/T0/demog_grouped_catbased.csv').head())


    # clean
    # df = pd.read_csv('src/recommendation/data/T0/demog_ranking_grouped_catbased.csv')
    # columns = ['Gender', 'Education level', 'Marital status', 'Region', 'Occupation Group']
    # for col in columns:
    #     print(df[col].unique())

    # print(df.head())

    # print(clean('src/recommendation/data/T1_predicted/demog_ranking_grouped_catbased.csv').head())


    # train_df_T1 = pd.read_csv('src/recommendation/data/T1/test_with_lifestyle.csv')
    # print(train_df_T1.head())
    # train_df_T1['Age'] = train_df_T1['Age'] + 2
    # print(train_df_T1.head())

    
    # train_df_T1.to_csv('src/recommendation/data/T1/test_with_lifestyle.csv', index=False)




    # train
    # remove_nan('src/recommendation/data/T0/demog_ranking_grouped_catbased.csv')
    # remove_nan('src/recommendation/data/T1/demog_ranking_grouped_catbased.csv')
    # remove_nan('src/recommendation/data/T1_predicted/demog_ranking_grouped_catbased.csv')
    # remove_nan('src/recommendation/data/T0/demog_grouped_catbased.csv')
    # remove_nan('src/recommendation/data/T1/demog_grouped_catbased.csv')
    # remove_nan('src/recommendation/data/T1_predicted/demog_grouped_catbased.csv')

    # df_t0_path = 'src/recommendation/data/T0/demog_ranking_grouped_catbased_no_norm.csv'
    # df_t1_path = 'src/recommendation/data/T1/demog_ranking_grouped_catbased_no_norm.csv'
    # df_t1_predicted_path = 'src/recommendation/data/T1_predicted/demog_ranking_grouped_catbased_no_norm_single.csv'
    # df_t0_path = 'src/recommendation/data/T0/demog_grouped_catbased.csv'
    # df_t1_path = 'src/recommendation/data/T1/demog_grouped_catbased.csv'
    # df_t1_predicted_path = 'src/recommendation/data/T1_predicted/demog_grouped_catbased.csv'



    # test
    # remove_nan('src/recommendation/data/T0/test_with_lifestyle.csv')
    # remove_nan('src/recommendation/data/T1/test_with_lifestyle.csv')
    # remove_nan('src/recommendation/data/T1_predicted/test_with_lifestyle.csv')

    # df_t0_path = 'src/recommendation/data/T0/test_with_lifestyle.csv'
    # df_t1_path = 'src/recommendation/data/T1/test_with_lifestyle.csv'
    # df_t1_predicted_path = 'src/recommendation/data/T1_predicted/test_with_lifestyle_single.csv'

    # df_t0_filtered, df_t1_filtered, df_t1_predicted_filtered = filter_matching_customers(df_t0_path, df_t1_path, df_t1_predicted_path)


    # df_T0 = 'src/recommendation/data/T0/demog_ranking_grouped_catbased.csv'
    # df_T1 = 'src/recommendation/data/T1/demog_ranking_grouped_catbased.csv'
    # df_T1_predicted = 'src/recommendation/data/T1_predicted/demog_ranking_grouped_catbased.csv'

    # df_T0 = 'src/recommendation/data/T0/demog_ranking_grouped_catbased_no_norm.csv'
    # df_T1 = 'src/recommendation/data/T1/demog_ranking_grouped_catbased_no_norm.csv'
    # df_T1_predicted = 'src/recommendation/data/T1_predicted/demog_ranking_grouped_catbased_no_norm.csv'



    # df_T0 = 'src/recommendation/data/T0/demog_grouped_catbased.csv'
    # df_T1 = 'src/recommendation/data/T1/demog_grouped_catbased.csv'
    # df_T1_predicted = 'src/recommendation/data/T1_predicted/demog_grouped_catbased.csv'

    # clean(df_T0)
    # clean(df_T1)
    # clean(df_T1_predicted)