import pandas as pd
import os
from src.client.qdrant import *
from src.data_refresher.rag.utils.utils import *

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main_pipeline(df, collection_name="customer_transitions"):
    client = get_qdrant_client()
    print("Qdrant client initialized successfully")
    
    print("\n" + "="*30)
    print("STEP 1: Document Preparation")
    print("="*30)
    documents = prepare_documents(df)
    # print(documents)
    
    print("\n" + "="*30)
    print("STEP 2: Embedding Generation")
    print("="*30)
    embedded_docs = generate_embeddings(documents)
    # print(embedded_docs)

    print("\n" + "="*30)
    print("STEP 3: Collection Setup")
    print("="*30)
    create_collection(client, collection_name, vector_size=len(embedded_docs[0]["embedding"]))
    
    print("\n" + "="*30)
    print("STEP 4: Qdrant Upload")
    print("="*30)
    upload_to_qdrant(client, collection_name, embedded_docs)
    
    print("Pipeline completed successfully!")
    return embedded_docs


if __name__ == "__main__":
    train_cra = pd.read_csv("src/data_refresher/data/summary_reasoning/train_cra_v4.csv")
    COLLECTION_NAME = "customer_transitions"

    embedded_docs = main_pipeline(train_cra, COLLECTION_NAME)