import pandas as pd
import os
from src.client.qdrant import *
from src.recommendation.rag.utils.utils import *

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main_pipeline(df, collection_name="customers"):
    """Main pipeline to process customer data and upload to Qdrant"""
    client = get_qdrant_client()
    print("Qdrant client initialized successfully")
    
    print("\n" + "="*30)
    print("STEP 1: Document Preparation")
    print("="*30)
    documents = prepare_documents(df)
    
    print("\n" + "="*30)
    print("STEP 2: Embedding Generation")
    print("="*30)
    embedded_docs = generate_embeddings(documents)

    print("\n" + "="*30)
    print("STEP 3: Collection Setup")
    print("="*30)
    create_collection(client, collection_name, vector_size=len(embedded_docs[0]["embedding"]))
    
    print("\n" + "="*30)
    print("STEP 4: Qdrant Upload")
    print("="*30)
    upload_to_qdrant(client, collection_name, embedded_docs)
    
    print("\n" + "="*50)
    print("🎉 Pipeline completed successfully!")
    print("="*50)
    return embedded_docs


if __name__ == "__main__":
    train_demog = pd.read_csv("src/recommendation/data/rag/train_T0_demog_summ/train_T0_demog_summ_t0.csv")
    COLLECTION_NAME = "customers"

    embedded_docs = main_pipeline(train_demog, COLLECTION_NAME)