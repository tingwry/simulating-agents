import pandas as pd
from sklearn.model_selection import train_test_split
import os

DIR = 'src/clustering/data_v3'

df = pd.read_csv('src/cleaned_mock.csv')

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

print("Training set shape:", train_df.shape)
print("Testing set shape:", test_df.shape)

# Save csv
train_result_path = os.path.join(DIR, "train_df.csv")
test_result_path = os.path.join(DIR, "test_df.csv")

train_df.to_csv(train_result_path, index=False)
test_df.to_csv(test_result_path, index=False)

print(f"\n✅ Results saved")