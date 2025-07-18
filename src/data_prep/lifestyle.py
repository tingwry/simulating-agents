import pandas as pd

# lifestyle = pd.read_csv('src/data/raw_data/customerllm_lifestyle.csv')
lifestyle = pd.read_csv('src/data/preprocessed_data/lifestyle.csv')

train_t0 = pd.read_csv('src/data/T0/train_df.csv')
test_t0 = pd.read_csv('src/data/T0/test_df.csv')

def dropna(lifestyle):
    print(len(lifestyle['cust_id'].unique()))

    lifestyle = lifestyle.dropna()
    lifestyle.to_csv("lifestyle.csv", index=False)
    print(lifestyle.head())

    print(len(lifestyle['cust_id'].unique()))

def df_with_lifestyle(df, lifestyle, filename):
    result = df.copy()
    cust_id_ls = lifestyle['cust_id'].unique()
    result = result[result['CUST_ID'].isin(cust_id_ls)]

    result.to_csv(filename, index=False)
    return result

# df_with_lifestyle(train_t0, lifestyle, 'src/data/T0/train_with_lifestyle.csv')
# df_with_lifestyle(test_t0, lifestyle, 'src/data/T0/test_with_lifestyle.csv')

train_with_lifestyle = pd.read_csv('src/data/T0/train_with_lifestyle.csv')
test_with_lifestyle = pd.read_csv('src/data/T0/test_with_lifestyle.csv')

print(train_with_lifestyle.shape)
print(test_with_lifestyle.shape)
