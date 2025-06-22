from src.clustering.approach_3.utils.utils import *
import pandas as pd

train_df = pd.read_csv('src/clustering/data/train_df.csv')
test_df = pd.read_csv('src/clustering/data/test_df.csv')
MODEL_DIR = "src/clustering/approach_3/model"
# RESULTS_DIR = "src/clustering/approach_3/result"

# def test_approach_3(train_df, emb_dim=None, n_clusters=None):
#     df_num, numeric_feature_col = preprocess_num(train_df)

print(len(train_df))