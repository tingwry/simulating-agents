import pandas as pd

def preprocess(df):
    for _, row in df.iterrows():
        if row['Marital status'] == 'divorce':
            print(row['PRED_marital_status'])
            row['PRED_marital_status'] = 'divorce'
        elif row['Marital status'] == 'others':
            print(row['PRED_marital_status'])
            row['PRED_marital_status'] = 'others'
        elif row['PRED_marital_status'] == 'single':
            print(row['PRED_num_children'])
            row['PRED_num_children'] = 0
    return df
          

if __name__ == "__main__":
    predictions = pd.read_csv('src/prediction/pred_results/predictions_cluster_multi_ca.csv')

    postpc_pred = preprocess(predictions)
    postpc_pred.to_csv('src/prediction/pred_results/predictions_cluster_multi_ca_postpc.csv', index=False)