import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler  # to standardize the features
from sklearn.decomposition import PCA  # to apply PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

from pyod.models.ecod import ECOD
from yellowbrick.cluster import KElbowVisualizer

import os
import json
import joblib

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from src.client.qdrant import *
from qdrant_client import models
from tqdm import tqdm
import torch

# train_T0 = pd.read_csv("src/clustering/data_v2/train_df.csv")
# train_T1 = pd.read_csv("src/train_T1/train_T1.csv")
train_cra = pd.read_csv("src/data/summary_reasoning/train_cra_v4.csv")
COLLECTION_NAME = "customer_transitions"



def create_collection(client, collection_name, vector_size=384):
    """Create Qdrant collection if it doesn't exist"""
    try:
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE
            ),
        )
        print(f"Collection {collection_name} created successfully")
    except Exception as e:
        print(f"Error creating collection: {e}")

def prepare_documents(df):
    """Convert DataFrame rows to Qdrant document format with detailed progress"""
    documents = []
    print("\n" + "="*50)
    print("Preparing documents for embedding...")
    print(f"Total rows to process: {len(df)}")
    print("="*50)
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        current = idx + 1
        if current % 100 == 0 or current == 1 or current == len(df):
            print(f"\nProcessing row {current}/{len(df)} - CUST_ID: {row['CUST_ID']}")
        
        try:
            document = {
                "id": int(row['CUST_ID']),  # Convert to string to ensure compatibility
                # "CUST_ID": str(row['CUST_ID']), # Convert to string to ensure compatibility
                "metadata": {
                    "full_profile_T0": {
                        "age": row['Age'],
                        "education": row['Education level'],
                        "marital_status": row['Marital status'],
                        "occupation": row['Occupation Group'],
                        "region": row['Region'],
                        "children": row['Number of Children'],
                        "vehicles": row['Number of Vehicles'],
                        "gender": row['Gender'],
                        "savings_account": row['Savings Account'],
                        "savings_account_subgroup": row['Savings Account Subgroup'],
                        "health_insurance": row['Health Insurance'],
                        "lending": row['Lending'],
                        "payment": row['Payment'],
                        "service": row['Service'],
                        "business_lending": row['Business Lending'],
                        "deposit_account": row['Deposit Account'],
                        "deposit_account_balance": row['Deposit Account Balance'],
                        "deposit_account_transactions": row['Deposit Account Transactions'],
                        "deposit_account_transactions_avg": row['Deposit Account Transactions AVG'],
                        "deposit_account_transactions_min": row['Deposit Account Transactions MIN'],
                        "deposit_account_transactions_max": row['Deposit Account Transactions MAX'],
                        "deposit_account_inflow": row['Deposit Account Inflow'],
                        "deposit_account_inflow_min": row['Deposit Account Inflow MIN'],
                        "deposit_account_inflow_max": row['Deposit Account Inflow MAX'],
                        "deposit_account_outflow": row['Deposit Account Outflow'],
                        "deposit_account_outflow_min": row['Deposit Account Outflow MIN'],
                        "deposit_account_outflow_max": row['Deposit Account Outflow MAX'],
                        "deposit_account_inflow_amount": row['Deposit Account Inflow Amount'],
                        "deposit_account_outflow_amount": row['Deposit Account Outflow Amount'],
                        "summary": row['Summary']
                    },
                    "action_T1": {
                        "education": row.get('Education_T1', None),
                        "marital_status": row.get('Marital_Status_T1', None),
                        "occupation": row.get('Occupation_Group_T1', None),
                        "region": row.get('Region_T1', None),
                        "children": row.get('Number_of_Children_T1', None)
                    },
                    "reason": row['Reason']
                },
                "embedding_text": row['Summary']  # Using the summary for embedding
            }
            documents.append(document)
            if current % 100 == 0:
                print(f"✔ Prepared document {current} | Last CUST_ID: {row['CUST_ID']}")
                
        except Exception as e:
            print(f"\n⚠️ Error processing row {current} (CUST_ID: {row.get('CUST_ID', 'UNKNOWN')}): {str(e)}")
            continue
    
    print("\n" + "="*50)
    print(f"✅ Successfully prepared {len(documents)}/{len(df)} documents")
    print("="*50)
    return documents


def generate_embeddings(documents, model_name="sentence-transformers/paraphrase-MiniLM-L6-v2"):
    """Generate embeddings for all documents"""
    """Generate embeddings with enhanced progress tracking"""
    print("\n" + "="*50)
    print("Generating document embeddings...")
    print(f"Using model: {model_name}")
    print("="*50)

    try:
        model = SentenceTransformer(model_name)
        print("Model loaded successfully")
        
        texts = []
        print("\nPreparing texts for embedding...")
        for i, doc in enumerate(tqdm(documents, desc="Preparing texts")):
            texts.append(doc["embedding_text"])
            if (i+1) % 100 == 0:
                print(f"Prepared {i+1}/{len(documents)} text chunks")
        
        print("\nStarting embedding generation...")
        embeddings = model.encode(
            texts,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=32,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        print(f"Generated {len(embeddings)} embeddings")
        
        # Add embeddings to documents
        print("\nAssigning embeddings to documents...")
        for i, (doc, emb) in enumerate(zip(documents, embeddings)):
            print(emb.tolist().dtype())
            doc["embedding"] = emb.tolist()
            if (i+1) % 100 == 0:
                print(f"Assigned embeddings to {i+1}/{len(documents)} documents")
                
    except Exception as e:
        print(f"\n⚠️ Error during embedding generation: {str(e)}")
        raise
    
    print("\n" + "="*50)
    print("✅ Embedding generation completed")
    print("="*50)
    return documents

def upload_to_qdrant(client, collection_name, documents, batch_size=100):
    """Upload documents to Qdrant in batches"""
    print("\n" + "="*50)
    print(f"Uploading documents to Qdrant collection: {collection_name}")
    print(f"Total documents: {len(documents)} | Batch size: {batch_size}")
    print("="*50)

    points = []
    success_count = 0
    error_count = 0
    
    for i, doc in enumerate(tqdm(documents, desc="Upload progress")):
        try:
            points.append(
                models.PointStruct(
                    id=doc["id"],
                    vector=doc["embedding"],
                    payload=doc["metadata"]
                )
            )
            
            # Progress reporting
            if (i+1) % 50 == 0:
                print(f"Prepared {i+1}/{len(documents)} points for upload")
                print(f"Last CUST_ID processed: {doc['id']}")
            
            # Upload in batches
            if len(points) >= batch_size or i == len(documents) - 1:
                client.upsert(
                    collection_name=collection_name,
                    points=points,
                    wait=True
                )
                success_count += len(points)
                points = []
                print(f"✔ Uploaded batch up to document {i+1}")
                
        except Exception as e:
            error_count += 1
            print(f"\n⚠️ Error uploading document {i+1} (CUST_ID: {doc.get('id', 'UNKNOWN')}):{str(e)}")
            continue
    
    print("\n" + "="*50)
    print("Upload Summary:")
    print(f"✅ Successfully uploaded: {success_count} documents")
    print(f"⚠️ Failed to upload: {error_count} documents")
    if error_count > 0:
        print("Note: Some documents failed to upload. Check error logs above.")
    print("="*50)
                  

def main_pipeline(df, collection_name="customer_transitions"):
    """Complete pipeline from DataFrame to Qdrant"""
    # Initialize clients and model
    client = get_qdrant_client()
    print("Qdrant client initialized successfully")
    
    # Step 1: Prepare documents
    print("\n" + "="*30)
    print("STEP 1: Document Preparation")
    print("="*30)
    documents = prepare_documents(df)
    # print(documents)
    
    # Step 2: Generate embeddings
    print("\n" + "="*30)
    print("STEP 2: Embedding Generation")
    print("="*30)
    embedded_docs = generate_embeddings(documents)
    # print(embedded_docs)
    
    # Step 3: Create collection (if not exists)
    # print("\n" + "="*30)
    # print("STEP 3: Collection Setup")
    # print("="*30)
    # create_collection(client, collection_name, vector_size=len(embedded_docs[0]["embedding"]))
    
    # Step 4: Upload to Qdrant
    # print("\n" + "="*30)
    # print("STEP 4: Qdrant Upload")
    # print("="*30)
    # upload_to_qdrant(client, collection_name, embedded_docs)
    
    print("Pipeline completed successfully!")
    return embedded_docs


embedded_docs = main_pipeline(train_cra, COLLECTION_NAME)


