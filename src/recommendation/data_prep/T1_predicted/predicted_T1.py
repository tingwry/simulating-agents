import pandas as pd
from src.data_refresher.clustering.approach_2_embed.utils.utils import *


# (Data refresher) Predictions_cluster_multi_ca --> Data for recommendation

def transform_customer_data(csv_path, test_csv, output_path):
    df = pd.read_csv(csv_path)
    test_df = pd.read_csv(test_csv)

    # Get the CUST_IDs from the test file
    matching_cust_ids = set(test_df['CUST_ID'])
    
    # Filter predictions to only include matching CUST_IDs
    df = df[df['CUST_ID'].isin(matching_cust_ids)]
    
    # Create a mapping dictionary for the columns to replace
    column_mapping = {
        'Education level': 'PRED_education',
        'Marital status': 'PRED_marital_status',
        'Occupation Group': 'PRED_occupation',
        'Number of Children': 'PRED_num_children',
        'Region': 'PRED_region'
    }
    
    # Replace the original columns with the predicted values
    for new_col, pred_col in column_mapping.items():
        df[new_col] = df[pred_col]
    
    # Increase Age by 2
    df['Age'] = df['Age'] + 2
    
    # Select only the columns we want to keep
    columns_to_keep = [
        'Number of Children', 'Number of Vehicles', 'Gender', 'Education level', 
        'Marital status', 'Savings Account', 'Savings Account Subgroup', 
        'Health Insurance', 'Lending', 'Payment', 'Service', 'Business Lending', 
        'Deposit Account', 'Deposit Account Balance', 'Deposit Account Transactions', 
        'Deposit Account Transactions AVG', 'Deposit Account Transactions MIN', 
        'Deposit Account Transactions MAX', 'Deposit Account Inflow', 
        'Deposit Account Inflow MIN', 'Deposit Account Inflow MAX', 
        'Deposit Account Outflow', 'Deposit Account Outflow MIN', 
        'Deposit Account Outflow MAX', 'Deposit Account Inflow Amount', 
        'Deposit Account Outflow Amount', 'Age', 'Region', 'Occupation Group', 
        'CUST_ID'
    ]

    final_output_path = output_path + '/test_with_lifestyle_single.csv'
    df[columns_to_keep].to_csv(final_output_path, index=False)
    
    # Return the transformed dataframe with only the selected columns
    return df[columns_to_keep]


if __name__ == "__main__":
    # demog_at_T1_path = 'src/data_refresher/prediction/pred_results/train_with_lifestyle_pred_T1_single.csv'
    demog_at_T1_path = 'src/data_refresher/prediction/pred_results/test_with_lifestyle_pred_T1_single.csv'

    # train_with_lifestyle_T0_path = 'src/recommendation/data/T0/train_with_lifestyle.csv'
    test_with_lifestyle_T0_path = 'src/recommendation/data/T0/test_with_lifestyle.csv'
    output_path = 'src/recommendation/data/T1_predicted'

    transformed_df = transform_customer_data(demog_at_T1_path, test_with_lifestyle_T0_path, output_path)