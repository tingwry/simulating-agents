import pandas as pd
import os


df = pd.read_csv('src/mock_filtered.csv')
DIR = 'src/'

# df_filtered = df[df['NO_OF_CHLD'] != df['NO_OF_CHLD_prev']][['NO_OF_CHLD', 'NO_OF_CHLD_prev']]
df_filtered = df[(df['EDU_DESC'] != df['EDU_DESC_prev']) & (~df['EDU_DESC'].isna())][['EDU_DESC', 'EDU_DESC_prev']]

print(df_filtered)
