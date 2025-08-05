import pandas as pd
import joblib
from tqdm import tqdm  # For progress bars
import time
from src.clustering.approach_2_embed.utils.utils import *
from src.client.llm import get_aoi_client
import json

# Cluster Predictions
# TEST_DATA_PATH_T0 = pd.read_csv('src/data/T0/test_with_lifestyle.csv')

# MODEL_DIR = "src/clustering/approach_2_embed/model/model_app2_v4.pkl"
# clus_model = joblib.load(MODEL_DIR)

# # Define transaction categories
# txn_cat = ['charity', 'loan', 'utility', 'investment', 'finance', 'shopping',
#            'personal_care', 'medical', 'home_and_living', 'insurance', 'automotive',
#            'restaurant', 'business', 'entertainment', 'bank', 'education', 'pet_care',
#            'government', 'travel', 'transportation', 'visit', 'system_dpst']


# def predict_cluster(TEST_DATA_PATH_T0, clus_model):    
#     # Create embeddings and predict clusters
#     print("Step 1: Creating customer embeddings...")
#     df_embedding = embedding_creation(TEST_DATA_PATH_T0)
#     print(f"Embeddings created for {len(df_embedding)} customers")
    
#     print("\nStep 2: Predicting clusters...")
#     predicted_clus_df = predict(TEST_DATA_PATH_T0, df_embedding, clus_model)
#     cluster_counts = predicted_clus_df['cluster'].value_counts()
#     print("Cluster distribution:")
#     print(cluster_counts.to_string())
    
    
#     print("\n" + "="*50)
#     print("Processing complete!")
    
#     return predicted_clus_df


# TEST_DATA_PATH_T0 = TEST_DATA_PATH_T0.head(3)

# # Generate predictions with reasoning
# predicted_clus_df = predict_cluster(TEST_DATA_PATH_T0, clus_model)
# print(predicted_clus_df)



# (Data refresher) Predictions_cluster_multi_ca --> Data for recommendation

demog_at_T1_path = 'src/prediction/pred_results/predictions_cluster_multi_ca.csv'
test_with_lifestyle_T0_path = 'src/data/T0/test_with_lifestyle.csv'
output_path = 'src/recommendation/catboost/T1/data'

def transform_customer_data(csv_path, test_csv, output_path):
    # Read the CSV file
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

    final_output_path = output_path + '/test_with_lifestyle_T1.csv'
    df[columns_to_keep].to_csv(final_output_path, index=False)
    
    # Return the transformed dataframe with only the selected columns
    return df[columns_to_keep]


transformed_df = transform_customer_data(demog_at_T1_path, test_with_lifestyle_T0_path, output_path)