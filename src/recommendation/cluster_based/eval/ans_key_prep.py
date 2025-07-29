import pandas as pd
import numpy as np

# binary (0/1)
def binary_ans_key_prep():
    user_item_matrix = pd.read_csv('src/data/cf_demog_summary/user_item_matrix/user_item_matrix.csv')

    user_item_matrix_binary = user_item_matrix.copy()
    user_item_matrix_binary.iloc[:, 1:] = np.where(user_item_matrix.iloc[:, 1:] > 0, 1, 0)


    output_path = 'src/recommendation/cluster_based/eval/ans_key.csv'
    user_item_matrix_binary.to_csv(output_path, index=False)