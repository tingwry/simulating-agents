import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import json
import re
from tqdm import tqdm
from dotenv import load_dotenv
from src.client.qdrant import get_qdrant_client
from src.client.llm import get_aoi_client

from src.prediction.prompts import (
    indiv_create_prediction_prompt,
    cluster_create_prediction_prompt,
    rag_create_prediction_prompt,
    indiv_create_prediction_prompt_action,
    cluster_create_prediction_prompt_action,
    rag_create_prediction_prompt_action,
    create_prediction_prompt_status
)


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

# method -> indiv / cluster / rag
# stage -> single / multi
def process_customers(df, method="indiv", stage="single", change_analysis=False, batch_size=None):
    """Main processing function."""

    # Track processed and failed customers
    processed = []
    failed = []

    results = []
    total_customers = batch_size if batch_size else len(df)

    # rag
    if method == "rag":
        print("Retrieving similar customers...")
        similar_customers_data = retrieve_similar_customers(df, COLLECTION_NAME)
    
        # Save similar customers data
        with open(similar_output_file, 'w', encoding='utf-8') as f:
            json.dump(similar_customers_data, f, indent=4, ensure_ascii=False)

    
    for i in tqdm(range(total_customers), desc="Processing customers"):
        customer_row = df.iloc[i]
        customer_id = customer_row['CUST_ID']

        if method == "cluster":
            cluster_id = int(customer_row['cluster'])
        
        try:
            if method == "rag":
                # Get similar customers data
                cust_similar_data = similar_customers_data.get(customer_id, {})
                if not cust_similar_data.get('similar_customers'):
                    print(f"\nWarning: No similar customers found for {customer_id}")
                    failed.append(customer_id)
                    continue

                if stage == "single":
                    prompt = rag_create_prediction_prompt(customer_row, cust_similar_data)
                    print('prompt: ' + prompt)
                    response = get_llm_prediction(prompt)
                elif stage == "multi":
                    # actions
                    action_prompt = rag_create_prediction_prompt_action(customer_row, cust_similar_data)
                    print('action prompt: ' + action_prompt)
                    pred_actions = get_llm_prediction(action_prompt)
                    # status
                    status_prompt = create_prediction_prompt_status(customer_row)
                    print('status prompt: ' + status_prompt)
                    response = get_llm_prediction(status_prompt, pred_actions)

            elif method == "indiv":
                if stage == "single":
                    prompt = indiv_create_prediction_prompt(customer_row)
                    print('prompt: ' + prompt)
                    response = get_llm_prediction(prompt)
                elif stage == "multi":
                    # actions
                    action_prompt = indiv_create_prediction_prompt_action(customer_row)
                    print('action prompt: ' + action_prompt)
                    pred_actions = get_llm_prediction(action_prompt)
                    # status
                    status_prompt = create_prediction_prompt_status(customer_row)
                    print('status prompt: ' + status_prompt)
                    response = get_llm_prediction(status_prompt, pred_actions)

            elif method == "cluster":
                if stage == "single":
                    prompt = cluster_create_prediction_prompt(customer_row, cluster_id, change_analysis)
                    print('prompt: ' + prompt)
                    response = get_llm_prediction(prompt)
                elif stage == "multi":
                    # actions
                    action_prompt = cluster_create_prediction_prompt_action(customer_row, cluster_id, change_analysis)
                    print('action prompt: ' + action_prompt)
                    pred_actions = get_llm_prediction(action_prompt)
                    # status
                    status_prompt = create_prediction_prompt_status(customer_row)
                    print('status prompt: ' + status_prompt)
                    response = get_llm_prediction(status_prompt, pred_actions)

            # Parse response
            json_str = re.sub(r'^```json|```$', '', response.strip(), flags=re.MULTILINE).strip()
            prediction = json.loads(json_str)
            
            # Save results
            results.append(format_results(customer_row, prediction))
            processed.append(customer_id)
            
        except Exception as e:
            print(f"❌ Error processing customer {customer_id}: {str(e)}")
            failed.append(customer_id)
            continue
    
    # Print summary
    print(f"\nProcessing complete:")
    print(f"- Successfully processed: {len(processed)} customers")
    print(f"- Failed to process: {len(failed)} customers")
    if failed:
        print(f"Failed customer IDs: {failed}")
    
    return pd.DataFrame(results)


def format_results(customer_row, prediction):
    """Format results for output DataFrame."""
    return {
        **customer_row.to_dict(),
        **prediction['predictions'],
        **{f"CONFIDENCE_{k}": v for k, v in prediction['confidence_scores'].items()},
        **{f"REASONING_{k}": v for k, v in prediction['reasoning'].items()}
    }

# def save_final_results(results_df, output_dir, file_name):
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Ensure filename doesn't include extension
#     base_name = os.path.splitext(file_name)[0]
    
#     csv_path = os.path.join(output_dir, f"{base_name}.csv")
#     results_df.to_csv(csv_path, index=False)
    
#     json_path = os.path.join(output_dir, f"{base_name}.json")
#     results_df.to_json(json_path, orient='records', indent=2)
    
#     print(f"Results saved to:\n- {csv_path}\n- {json_path}")

def save_final_results(results_df, output_dir, file_name):
    """Save the final combined results by appending to existing CSV."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Define the CSV file path (use a fixed name instead of versioned)
    csv_path = os.path.join(output_dir, file_name+'.csv')
    
    # Check if file exists to determine whether to write header
    file_exists = os.path.isfile(csv_path)
    
    # Append to CSV (or create new if doesn't exist)
    results_df.to_csv(csv_path, mode='a', header=not file_exists, index=False)
    
    # Save JSON version separately (optional)
    # json_path = os.path.join(output_dir, "predictions_latest.json")
    # results_df.to_json(json_path, orient='records', indent=2)
    
    print(f"Results appended to {csv_path}")
    # print(f"Latest results also saved to {json_path}")

if __name__ == "__main__":
    # Load data indiv/rag
    test_summ = pd.read_csv("src/data/summary_reasoning/test_summ_v1.csv")
    # testtest = test_summ.head(5)
    testtest = test_summ[test_summ['CUST_ID'].isin([2808, 1399, 157])]

    # Load data cluster
    # df = pd.read_csv('src/clustering/approach_2_embed/pred_result/full_data_with_cluster/full_data_with_cluster_v2.csv')
    # testtest = df.head(2)
    # testtest = df[df['CUST_ID'].isin([2227, 3318, 1425])]


    VERSIONED_DIR = None  # Will be set when saving results
    COLLECTION_NAME = "customer_transitions"
    output_dir = "src/prediction/pred_results"
    similar_output_file = "src/prediction/similar_cust_results/v2_rag_single/sim_cust_results.json"
    
    # Process customers and save results
    results_df = process_customers(testtest, method="rag", stage="single", change_analysis=False, batch_size=None)
    save_final_results(results_df, output_dir, 'predictions_rag_single')