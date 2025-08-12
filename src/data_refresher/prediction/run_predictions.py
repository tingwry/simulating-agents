import pandas as pd
import os
import json
import re
from tqdm import tqdm
from dotenv import load_dotenv
from src.data_refresher.prediction.prompts import (
    indiv_create_prediction_prompt,
    cluster_create_prediction_prompt,
    rag_create_prediction_prompt,
    indiv_create_prediction_prompt_action,
    cluster_create_prediction_prompt_action,
    rag_create_prediction_prompt_action,
    create_prediction_prompt_status
)
from src.data_refresher.prediction.utils.utils import *


# Load environment variables
load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"



def run_predictions(df, method="indiv", stage="single", change_analysis=False, batch_size=None):
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
                    action_prompt = cluster_create_prediction_prompt_action(customer_row, cluster_id, change_analysis, with_constraints=False)
                    print('action prompt: ' + action_prompt)
                    pred_actions = get_llm_prediction(action_prompt)
                    # status
                    status_prompt = create_prediction_prompt_status(customer_row, with_constraints=False)
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




def save_final_results(results_df, output_dir, file_name):
    os.makedirs(output_dir, exist_ok=True)
    
    # Ensure filename doesn't include extension
    base_name = os.path.splitext(file_name)[0]
    
    csv_path = os.path.join(output_dir, f"{base_name}.csv")
    results_df.to_csv(csv_path, index=False)
    
    json_path = os.path.join(output_dir, f"{base_name}.json")
    results_df.to_json(json_path, orient='records', indent=2)
    
    print(f"Results saved to:\n- {csv_path}\n- {json_path}")

def save_final_results_append(results_df, output_dir, file_name):
    """Save the final combined results by appending to existing CSV."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Define the CSV file path (use a fixed name instead of versioned)
    csv_path = os.path.join(output_dir, file_name+'.csv')
    
    # Check if file exists to determine whether to write header
    file_exists = os.path.isfile(csv_path)
    
    # Append to CSV (or create new if doesn't exist)
    results_df.to_csv(csv_path, mode='a', header=not file_exists, index=False)
    
    print(f"Results appended to {csv_path}")


if __name__ == "__main__":
    """
    method -> indiv / cluster / rag
    stage -> single / multi
    """
    
    # Load data indiv/rag
    test_summ = pd.read_csv("src/data_refresher/data/summary_reasoning/test_summ_v1.csv")

    # Load data cluster
    # df = pd.read_csv('src/clustering/approach_2_embed/pred_result/full_data_with_cluster/full_data_with_cluster_v2.csv')
    train_with_lifestyle_with_clus = pd.read_csv('src/data_refresher/clustering/approach_2_embed/pred_result/full_data_with_cluster/train_with_lifestyle_with_clus.csv')
    # testtest = train_with_lifestyle_with_clus[train_with_lifestyle_with_clus['CUST_ID'].isin([2223, 353, 2013, 1183, 3358])] 
    test_wth_lifestyle_with_clus = pd.read_csv('src/data_refresher/clustering/approach_2_embed/pred_result/full_data_with_cluster/test_wth_lifestyle_with_clus.csv')

    # setup
    COLLECTION_NAME = "customer_transitions"
    similar_output_file = "src/data_refresher/prediction/similar_cust_results/v1/sim_cust_results.json"
    VERSIONED_DIR = None  # Will be set when saving results
    output_dir = "src/data_refresher/prediction/pred_results"

    file_name = "test_with_lifestyle_pred_T1"
    
    # Process customers and save results
    results_df = run_predictions(test_wth_lifestyle_with_clus, method="cluster", stage="multi", change_analysis=True, batch_size=None)
    save_final_results(results_df, output_dir, file_name)