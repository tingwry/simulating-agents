import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

tqdm.pandas()

train_T0_demog_summ = pd.read_csv('src/data/cf_demog_summary/train_T0_demog_summ.csv/train_T0_demog_summ.csv')
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# Embed summaries
def embed_summ(df):
    result = df.copy()

    # Add progress bar for the embedding process
    print("Generating embeddings...")
    result["embedding"] = result["Demog Summary"].progress_apply(
        lambda x: model.encode(x)
    )
    
    print(f"\nCompleted! Processed {len(result)} rows.")
    return result

embedded_demog = embed_summ(train_T0_demog_summ)
embedded_demog.to_csv('src/data/cf_demog_summary/embedded_demog/embedded_demog.csv', index=False)