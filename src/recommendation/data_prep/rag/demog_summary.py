import pandas as pd
import numpy as np

import os
import json
import joblib

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from src.client.llm import *

# train_T0 = pd.read_csv("src/recommendation/data/T0/demog_grouped_catbased.csv")
train_T0 = pd.read_csv("src/recommendation/data/T0/demog_ranking_grouped_catbased_no_norm_t0.csv")
# test_T0 = pd.read_csv("src/recommendation/data/T0/test_with_lifestyle.csv")
test_T0 = pd.read_csv("src/recommendation/data/T0/test_with_lifestyle_t0.csv")

train_DIR = "src/recommendation/data/rag/train_T0_demog_summ"
test_DIR = "src/recommendation/data/rag/test_T0_demog_summ"
checkpoint_dir = 'src/recommendation/data/rag/summary_checkpoints'

# Context at T0
def context_summarizer(x):
    text = f"""
        **Demographics:**
        - Age: {x['Age']},
        - Gender: {x['Gender']},
        - Education: {x['Education level']},
        - Marital Status: {x['Marital status']},
        - Occupation Group: {x['Occupation Group']},
        - Region: {x['Region']},
        - Number of Children: {x['Number of Children']}

        **Historical Transactions:**
        - Finance: {x['finance_t0']} baht
        - Financial Services: {x['financial_services_t0']} baht
        - Home Lifestyle: {x['home_lifestyle_t0']} baht
        - Loan: {x['loan_t0']} baht
        - Shopping: {x['shopping_t0']} baht
        - Utility: {x['utility_t0']} baht
        - Health and Care: {x['health_and_care_t0']} baht
        - Transport Travel: {x['transport_travel_t0']} baht
        - Leisure: {x['leisure_t0']} baht
        - Public Services: {x['public_services_t0']} baht

        """
    
    full_prompt = """You are a customer insights analyst tasked with creating comprehensive customer profiles for similarity matching and retrieval. 

Your task is to synthesize the provided customer data into a rich, descriptive paragraph that captures demographic characteristics and historical transactional categories, including the analysis of what kind of person the customer is based on given info. 
This summary will be used for vector embedding and stored in a database to find customers with similar profiles.

REQUIREMENTS:
1. Include ALL demographic elements: age, gender, education level, marital status, occupation group, region, and number of children
2. Describe which transactional categories (keep the same categories name as given) the customer performed
3. Create a cohesive narrative that describes what type of customer this person is
4. Use descriptive language that would help identify similar customers


OUTPUT FORMAT:
Create a single, flowing paragraph (100-120 words) that weaves together demographic elements into a cohesive customer persona.

EXAMPLE TONE:
"This customer represents a [age group] [gender] [occupation context] based in [region] with [education implications] and [family situation]. Past historical transactions include [historical transactions]."

Customer Demographic and Historical Transactions Data:"""

    client = get_aoi_client()

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": full_prompt},
            {"role": "user", "content": text}
        ],
        temperature=0.3,
        max_tokens=1000
    )
    
    return response.choices[0].message.content

def merge_summaries(df):    
    train_T0_with_summ = df.copy()
    total_rows = len(train_T0_with_summ)
    
    print(f"\nGenerating summaries for {total_rows} customers...")
    train_T0_with_summ['Summary'] = train_T0_with_summ.apply(
        lambda x: print(f"Processing customer {x.name + 1}/{total_rows}") or context_summarizer(x), 
        axis=1
    )
    
    print("✅ Summary generation completed")
    return train_T0_with_summ

def save_csv_file(DIR, df, file_name, next_version=None):
    os.makedirs(DIR, exist_ok=True)
    
    if next_version == None:
        existing_results = [f for f in os.listdir(DIR) 
                        if f.startswith(f'{file_name}_v') and f.endswith('.csv')]
        
        if existing_results:
            versions = [int(f.split('_v')[1].split('.csv')[0]) for f in existing_results]
            next_version = max(versions) + 1
        else:
            next_version = 1

    result_filename = f"{file_name}_v{next_version}.csv"
    result_path = os.path.join(DIR, result_filename)
    
    df.to_csv(result_path, index=False)
    
    print(f"\n✅ Results saved to: {result_path}")

def save_checkpoint(df, dir_path, checkpoint_name):
    """
    Save a DataFrame checkpoint with timestamp
    """
    os.makedirs(dir_path, exist_ok=True)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{checkpoint_name}_{timestamp}.pkl"
    path = os.path.join(dir_path, filename)
    
    csv_path = os.path.join(dir_path, f"{checkpoint_name}_{timestamp}.csv")
    df.to_csv(csv_path, index=False)
    return path



def demog_summary_prep(df, DIR):
    testtest = df.copy()

    summarized_df = merge_summaries(testtest)

    # save_csv_file(DIR, summarized_df, 'train_T0_demog_summ_t0', next_version=None)
    save_csv_file(DIR, summarized_df, 'test_T0_demog_summ_t0', next_version=None)


# demog_summary_prep(train_T0, train_DIR)
demog_summary_prep(test_T0, test_DIR)