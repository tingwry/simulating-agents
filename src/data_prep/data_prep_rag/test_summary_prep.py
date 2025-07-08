import pandas as pd
import numpy as np

import os
import json
import joblib

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from src.client.llm import *

test_df = pd.read_csv("src/clustering/data_v3/test_df.csv")
DIR = "src/similar_indiv/rag/test_summary_v2"


# Context at T0
def context_summarizer(x):
    text = f"""
        - Age: {x['Age']},
        - Gender: {x['Gender']},
        - Education: {x['Education level']},
        - Marital Status: {x['Marital status']},
        - Occupation Group: {x['Occupation Group']},
        - Region: {x['Region']},
        - Number of Children: {x['Number of Children']},
        - Number of Vehicles: {x['Number of Vehicles']},
        - Savings Account: {x['Savings Account']},
        - Savings Account Subgroup: {x['Savings Account Subgroup']},
        - Health Insurance: {x['Health Insurance']},
        - Lending: {x['Lending']},
        - Payment: {x['Payment']},
        - Service: {x['Service']},
        - Business Lending: {x['Business Lending']},
        - Deposit Account: {x['Deposit Account']},
        - Deposit Account Balance: {x['Deposit Account Balance']},
        - Deposit Account Transactions: {x['Deposit Account Transactions']},
        - Deposit Account Transactions AVG: {x['Deposit Account Transactions AVG']},
        - Deposit Account Transactions MIN: {x['Deposit Account Transactions MIN']},
        - Deposit Account Transactions MAX: {x['Deposit Account Transactions MAX']},
        - Deposit Account Inflow: {x['Deposit Account Inflow']},
        - Deposit Account Inflow MIN: {x['Deposit Account Inflow MIN']},
        - Deposit Account Inflow MAX: {x['Deposit Account Inflow MAX']},
        - Deposit Account Outflow: {x['Deposit Account Outflow']},
        - Deposit Account Outflow MIN: {x['Deposit Account Outflow MIN']},
        - Deposit Account Outflow MAX: {x['Deposit Account Outflow MAX']},
        - Deposit Account Inflow Amount: {x['Deposit Account Inflow Amount']},
        - Deposit Account Outflow Amount: {x['Deposit Account Outflow Amount']}
        """
    
#     full_prompt = """You are a financial analyst tasked with creating comprehensive customer profiles. Based on the provided customer data, write a detailed paragraph that incorporates ALL the given information and provides insights into the customer's financial behavior.

# Your summary must include:

# 1. DEMOGRAPHIC PROFILE: Integrate age, gender, education level, marital status, occupation group, region, number of children, and number of vehicles into a cohesive description of the customer.

# 2. FINANCIAL PRODUCTS & SERVICES: Mention their savings account status and subgroup, health insurance, lending products, payment methods, services used, business lending, and deposit account details.

# 3. FINANCIAL BEHAVIOR ANALYSIS: Analyze and summarize their financial patterns based on:
#    - Deposit account balance and transaction patterns
#    - Transaction frequency (average, minimum, maximum transactions)
#    - Cash flow patterns (inflow and outflow amounts and ranges)
#    - Financial stability and money management habits
#    - Risk profile and financial sophistication level

# Write the summary as a single, flowing paragraph that reads naturally while ensuring every piece of data is incorporated. Focus on creating a narrative that tells the story of this customer's financial life and banking relationship.

# Example structure: "This [age]-year-old [gender] customer from [region] represents a [financial behavior summary] profile. With [education/occupation details] and [family situation], they demonstrate [financial patterns] through their banking activities, including [specific financial behaviors and product usage]..."

# Remember: Every single data point must be mentioned in your summary, but weave them together into meaningful insights about the customer's financial behavior and profile."""

    full_prompt = """You are a customer insights analyst tasked with creating comprehensive customer profiles for similarity matching and retrieval. 

Your task is to synthesize the provided customer data into a rich, descriptive paragraph that captures both demographic characteristics and financial behavioral patterns. This summary will be used for vector embedding and stored in a database to find customers with similar profiles.

Requirements:
1. Include ALL demographic elements: age, gender, education level, marital status, occupation group, region, number of children, and number of vehicles
2. Incorporate ALL financial service usage: savings account details, health insurance, lending, payment services, business lending, and deposit account information
3. Analyze and describe financial behavioral patterns based on transaction data (frequency, amounts, inflows/outflows, variability)
4. Create a cohesive narrative that describes what type of customer this person is
5. Use descriptive language that would help identify similar customers
6. Focus on behavioral insights, not just data points

Structure your response as a single, flowing paragraph (150-200 words) that tells the story of this customer's demographic profile and financial relationship patterns. Emphasize characteristics that would be useful for finding similar customers.

Example tone: "This customer represents a [age group] [gender] professional in [region] with [family situation]. Their financial behavior indicates [spending/saving patterns], demonstrating [financial characteristics]. They utilize [services] and show [transaction patterns], suggesting [customer type/segment]..."

Customer Data:"""

    client = get_aoi_client()

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": full_prompt},
            {"role": "user", "content": text}
        ],
        temperature=0.7,
        max_tokens=1000
    )
    
    # return {
    #     "content": response.choices[0].message.content,
    #     "usage": dict(response.usage),
    #     # "id": response.id
    # }
    return response.choices[0].message.content

def merge_summaries(df):    
    # # Apply summarizer to each row and create summary column
    # train_T0_with_summ = df.copy()
    # train_T0_with_summ['Summary'] = train_T0_with_summ.apply(lambda x: context_summarizer(x), axis=1)

    # return train_T0_with_summ
    # Apply summarizer to each row and create summary column
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
    
    # Save as pickle (preserves dtypes better than CSV)
    df.to_pickle(path)
    print(f"✅ Checkpoint saved: {path}")
    
    # Also keep a CSV version for readability
    csv_path = os.path.join(dir_path, f"{checkpoint_name}_{timestamp}.csv")
    df.to_csv(csv_path, index=False)
    return path

checkpoint_dir = 'src/similar_indiv/rag/test_checkpoints'

def test_summ_prep(test_df, DIR):
    # testtest = test_df.head()
    # second_q = test_df.iloc[len(test_df)//2:(len(test_df)//4)*3].copy()

    test_summ = merge_summaries(test_df)
    # save_checkpoint(test_summ, checkpoint_dir, "testtest_with_summaries")


    save_csv_file(DIR, test_summ, 'test_summ', next_version=None)

test_summ_prep(test_df, DIR)