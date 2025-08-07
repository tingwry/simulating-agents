import pandas as pd

# predictions = pd.read_csv('src/recommendation/binary_classification_rand_reg/T0/predictions/transaction_predictions_grouped.csv')
predictions = pd.read_csv('src/recommendation/binary_classification_rand_reg/T0/predictions/transaction_predictions_grouped_catbased.csv')

def zeros_to_ones(df):
    df = df.copy()
    cols_to_convert = [col for col in df.columns if col != 'cust_id']
    df[cols_to_convert] = df[cols_to_convert].replace(0, 1)
    
    return df

all_ones = zeros_to_ones(predictions)
all_ones.to_csv('src/recommendation/base_line_all_1/data/all_1_grouped_catbased.csv', index=False)