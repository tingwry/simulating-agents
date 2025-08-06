import pandas as pd
import numpy as np

import os
import json
import joblib

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from src.client.llm import *

# train_T0 = pd.read_csv("src/data/T0/train_with_lifestyle.csv")
test_T0 = pd.read_csv("src/data/T0/test_with_lifestyle.csv")
# DIR = "src/data/cf_demog_summary/train_T0_demog_summ.csv"
DIR = "src/data/cf_demog_summary/test_T0_demog_summ.csv"
checkpoint_dir = 'src/recommendation/cf/summary_checkpoints'

# Context at T0
def context_summarizer(x):
    text = f"""
        - Age: {x['Age']},
        - Gender: {x['Gender']},
        - Education: {x['Education level']},
        - Marital Status: {x['Marital status']},
        - Occupation Group: {x['Occupation Group']},
        - Region: {x['Region']},
        - Number of Children: {x['Number of Children']}
        """
    
    full_prompt = """You are a customer analytics specialist creating demographic profiles for a recommendation system that predicts transaction categories (charity, loans, travel, insurance, etc.).

Your task is to transform customer demographic data into a contextual narrative that captures demographic intersections. This summary will be converted to vector embeddings and used to map customers to latent collaborative filtering spaces.

CRITICAL REQUIREMENTS:
1. Include ALL demographic elements naturally within the narrative: age, gender, education level, marital status, occupation group, region, and number of children
2. Keep consistent structure but vary language to avoid repetition


OUTPUT FORMAT:
Create a single, flowing paragraph (100-120 words) that weaves together demographic elements into a cohesive customer persona.

EXAMPLE STRUCTURE:
"This customer represents a [life stage descriptor] [gender] [occupation context] based in [region context] with [education implications] and [family situation]."

Customer Demographic Data:"""

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
    train_T0_with_summ['Demog Summary'] = train_T0_with_summ.apply(
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
    
    # Save as pickle (preserves dtypes better than CSV)
    df.to_pickle(path)
    print(f"✅ Checkpoint saved: {path}")
    
    # Also keep a CSV version for readability
    csv_path = os.path.join(dir_path, f"{checkpoint_name}_{timestamp}.csv")
    df.to_csv(csv_path, index=False)
    return path



def demog_summary_prep(train_T0, DIR):
    # testtest = train_T0.head()
    # second_q = train_T0.iloc[(len(train_T0)//4)*3:].copy()
    # second_q = train_T0.iloc[len(train_T0)//4:].copy()
    second_q = train_T0.copy()

    train_T0_with_summ = merge_summaries(second_q)
    # save_checkpoint(train_T0_with_summ, checkpoint_dir, "q2_with_summaries")

    # train_T0_with_summ = pd.read_csv('src/similar_indiv/rag/checkpoints/01_with_summaries_20250627_224753.csv')

    # save_checkpoint(train_T0_with_summ, checkpoint_dir, "q2_with_demog_summ")

    # save_csv_file(DIR, train_T0_with_summ, 'train_T0_demog_summ', next_version=None)
    save_csv_file(DIR, train_T0_with_summ, 'test_T0_demog_summ', next_version=None)

# demog_summary_prep(train_T0, DIR)

# concat
# part1 = pd.read_csv('src/data/cf_demog_summary/train_T0_demog_summ.csv/train_T0_demog_summ_v1.csv')
# part2 = pd.read_csv('src/data/cf_demog_summary/train_T0_demog_summ.csv/train_T0_demog_summ_v2.csv')
# train_T0_demog_summ = pd.concat([part1, part2], axis=0, ignore_index=True)
# test_T0_demog_summ.to_csv('src/data/cf_demog_summary/train_T0_demog_summ.csv/train_T0_demog_summ.csv', index=False)

demog_summary_prep(test_T0, DIR)