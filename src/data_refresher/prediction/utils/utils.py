import pandas as pd
from sentence_transformers import SentenceTransformer
import os
from tqdm import tqdm
from dotenv import load_dotenv
from src.client.qdrant import get_qdrant_client
from src.client.llm import get_aoi_client


# Load environment variables
load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def retrieve_similar_customers(test_summ_df, collection_name, top_k=5):
    """Retrieve similar customers from Qdrant vector database."""
    model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")
    client = get_qdrant_client()
    
    results = {}
    
    for idx, row in tqdm(test_summ_df.iterrows(), total=len(test_summ_df), desc="Querying similar customers"):
        try:
            query_embedding = model.encode(row['Summary']).tolist()
            
            search_results = client.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                limit=top_k,
                with_payload=True,
                with_vectors=False
            )
            
            similar_customers = []
            for result in search_results:
                similar_customers.append({
                    "id": result.id,
                    "score": result.score,
                    "metadata": result.payload
                })
            
            results[row['CUST_ID']] = {
                "query_summary": row['Summary'],
                "similar_customers": similar_customers
            }
            
        except Exception as e:
            print(f"Error processing customer {row['CUST_ID']}: {str(e)}")
            results[row['CUST_ID']] = {
                "error": str(e),
                "query_summary": row['Summary'],
                "similar_customers": []
            }
    
    return results



def get_llm_prediction(prompt, predicted_actions=None):
    client = get_aoi_client()

    messages = [
        {"role": "system", "content": "You are a demographic prediction assistant."},
        {"role": "user", "content": prompt}
    ]
    
    # Add predicted actions if provided
    if predicted_actions:
        messages.append({
            "role": "user", 
            "content": f"Predicted actions at time T1: {predicted_actions}"
        })
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.7,
        max_tokens=1000
    )
    
    return response.choices[0].message.content

def format_results(customer_row, prediction):
    """Format results for output DataFrame."""
    return {
        **customer_row.to_dict(),
        **prediction['predictions'],
        **{f"CONFIDENCE_{k}": v for k, v in prediction['confidence_scores'].items()},
        **{f"REASONING_{k}": v for k, v in prediction['reasoning'].items()}
    }