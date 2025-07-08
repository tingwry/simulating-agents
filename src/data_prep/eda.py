import pandas as pd
import os

# df = pd.read_csv('src/mock_customers.csv')
# DIR = 'src/'

# df = df[['NO_OF_CHLD', 'EDU_DESC', 'MRRY_DESC', 'ADDR_PRVC', 'OCPN_DESC', 'NO_OF_CHLD_prev', 'EDU_DESC_prev', 'MRRY_DESC_prev', 'ADDR_PRVC_prev', 'OCPN_DESC_prev']]

# result_filename = f"mock_filtered.csv"
# result_path = os.path.join(DIR, result_filename)

# df.to_csv(result_path, index=False)

# print(f"\n✅ Results saved to: {result_path}")

df = pd.read_csv('src/mock_filtered.csv')
DIR = 'src/'

# df_filtered = df[df['NO_OF_CHLD'] != df['NO_OF_CHLD_prev']][['NO_OF_CHLD', 'NO_OF_CHLD_prev']]
df_filtered = df[(df['EDU_DESC'] != df['EDU_DESC_prev']) & (~df['EDU_DESC'].isna())][['EDU_DESC', 'EDU_DESC_prev']]

print(df_filtered)

# result_filename = f"mock_filtered.csv"
# result_path = os.path.join(DIR, result_filename)

# df.to_csv(result_path, index=False)

# print(f"\n✅ Results saved to: {result_path}")