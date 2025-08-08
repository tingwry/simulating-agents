import pandas as pd



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


if __name__ == "__main__":
    # lifestyle = pd.read_csv('src/data/raw_data/customerllm_lifestyle.csv')
    lifestyle = pd.read_csv('src/data_refresher/data/preprocessed_data/lifestyle.csv')

    # T0
    # train = pd.read_csv('src/data_refresher/data/T0/train_df.csv')
    # test = pd.read_csv('src/data_refresher/data/T0/test_df.csv')
    # train_output_filename = 'src/recommendation/data/T0/train_with_lifestyle.csv'
    # test_output_filename = 'src/recommendation/data/T0/test_with_lifestyle.csv'

    # T1
    train = pd.read_csv('src/data_refresher/data/T1/train_T1_v3.csv')
    test = pd.read_csv('src/data_refresher/data/T1/test_T1_actual_v3.csv')
    train_output_filename = 'src/recommendation/data/T1/train_with_lifestyle.csv'
    test_output_filename = 'src/recommendation/data/T1/test_with_lifestyle.csv'



    train_with_lifestyle = df_with_lifestyle(train, lifestyle, train_output_filename)
    test_with_lifestyle = df_with_lifestyle(test, lifestyle, test_output_filename)

    print(train_with_lifestyle.shape)
    print(test_with_lifestyle.shape)