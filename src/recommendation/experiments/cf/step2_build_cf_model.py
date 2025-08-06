import pandas as pd

lifestyle = pd.read_csv('src/data/preprocessed_data/lifestyle.csv')
train_T0_demog_summ = pd.read_csv('src/data/cf_demog_summary/train_T0_demog_summ.csv/train_T0_demog_summ.csv')

txn_cat = ['charity', 'loan', 'utility', 'investment', 'finance', 'shopping',
 'personal_care', 'medical', 'home_and_living', 'insurance', 'automotive',
 'restaurant', 'business', 'entertainment', 'bank', 'education', 'pet_care',
 'government', 'travel', 'transportation', 'visit', 'system_dpst']


# def create_user_item_matrix(lifestyle):
#     grouped = lifestyle.groupby(['cust_id', 'txn_catg'])['dpst_txn_cnt'].sum().unstack(fill_value=0)

#     # Normalize each row (customer) so their transaction counts sum to 1
#     user_item_matrix = grouped.div(grouped.sum(axis=1), axis=0)
    
#     # fill missing with 0
#     for cat in txn_cat:
#         if cat not in user_item_matrix.columns:
#             user_item_matrix[cat] = 0
    
#     # Reorder columns to match txn_cat list
#     user_item_matrix = user_item_matrix[txn_cat]

#     user_item_matrix.reset_index(inplace=True)
    
#     return user_item_matrix


# user_item_matrix = create_user_item_matrix(lifestyle)
# print(user_item_matrix)
# user_item_matrix.to_csv('src/data/cf_demog_summary/user_item_matrix/user_item_matrix.csv', index=False)

user_item_matrix = pd.read_csv('src/data/cf_demog_summary/user_item_matrix/user_item_matrix.csv')

