import pandas as pd

DATA_PATH = 'src/recommendation/binary_classification_rand_reg/data/demog_ranking_grouped_catbased.csv'

categories = ['loan', 'utility', 'finance', 'shopping', 
                 'financial_services', 'health_and_care', 'home_lifestyle', 'transport_travel',	
                 'leisure', 'public_services']

df = pd.read_csv(DATA_PATH)
for cat in categories:
    df[cat] = df[cat].apply(lambda x: 1 if x > 0 else 0)

print(df[categories].head())

df.to_csv('src/recommendation/multilabel/data/demog_grouped_catbased.csv', index=False)