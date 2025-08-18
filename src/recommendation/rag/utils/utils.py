from sentence_transformers import SentenceTransformer
from src.client.qdrant import *
from qdrant_client import models
from tqdm import tqdm
import torch
import pandas as pd
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

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
    
    # Transaction categories to track
    transaction_categories = [
        'loan', 'utility', 'finance', 'shopping', 'financial_services', 
        'health_and_care', 'home_lifestyle', 'transport_travel', 
        'leisure', 'public_services'
    ]
    
    print("\n" + "="*50)
    print("Preparing documents for embedding...")
    print(f"Total rows to process: {len(df)}")
    print("="*50)
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        current = idx + 1
        if current % 100 == 0 or current == 1 or current == len(df):
            print(f"\nProcessing row {current}/{len(df)} - CUST_ID: {row['CUST_ID']}")
        
        try:
            # Prepare transaction categories data
            transaction_data = {}
            performed_categories = []
            
            for category in transaction_categories:
                if category in row:
                    transaction_data[category] = int(row[category])
                    if int(row[category]) == 1:
                        performed_categories.append(category)
                else:
                    transaction_data[category] = 0
            
            document = {
                "id": int(row['CUST_ID']),
                "metadata": {
                    "demographics": {
                        "age": row['Age'],
                        "education": row['Education level'],
                        "marital_status": row['Marital status'],
                        "occupation": row['Occupation Group'],
                        "region": row['Region'],
                        "children": row['Number of Children'],
                        "gender": row['Gender'],
                        "demog_summary": row['Demog Summary']
                    },
                    "transactions": transaction_data,
                    "performed_categories": performed_categories,  # List of categories where value = 1
                    "total_active_categories": len(performed_categories)  # Count of active categories
                },
                "embedding_text": row['Demog Summary']  # This is what we'll embed
            }
            documents.append(document)
            
            if current % 100 == 0:
                print(f"✔ Prepared document {current} | Last CUST_ID: {row['CUST_ID']}")
                print(f"  Active transaction categories: {len(performed_categories)}")
                
        except Exception as e:
            print(f"\n⚠️ Error processing row {current} (CUST_ID: {row.get('CUST_ID', 'UNKNOWN')}): {str(e)}")
            continue
    
    print("\n" + "="*50)
    print(f"✅ Successfully prepared {len(documents)}/{len(df)} documents")
    print("="*50)
    return documents


def generate_embeddings(documents, model_name="sentence-transformers/all-mpnet-base-v2"):
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
        print(f"Generated {len(embeddings)} embeddings with shape: {embeddings.shape}")
        
        # Add embeddings to documents
        print("\nAssigning embeddings to documents...")
        for i, (doc, emb) in enumerate(zip(documents, embeddings)):
            doc["embedding"] = emb.tolist()  # Convert numpy array to list
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
                print(f"✔ Uploaded batch of {len(points)} documents (total: {success_count})")
                points = []
                
        except Exception as e:
            error_count += 1
            print(f"\n⚠️ Error uploading document {i+1} (CUST_ID: {doc.get('id', 'UNKNOWN')}): {str(e)}")
            continue
    
    print("\n" + "="*50)
    print("Upload Summary:")
    print(f"✅ Successfully uploaded: {success_count} documents")
    print(f"⚠️ Failed to upload: {error_count} documents")
    if error_count > 0:
        print("Note: Some documents failed to upload. Check error logs above.")
    print("="*50)

