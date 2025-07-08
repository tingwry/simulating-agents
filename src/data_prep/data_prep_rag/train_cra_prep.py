import pandas as pd
import numpy as np

import os
import json
import joblib

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from src.client.llm import *

train_T0 = pd.read_csv("src/clustering/data_v3/train_df.csv")
train_T1 = pd.read_csv("src/train_T1/train_T1_v3.csv")
DIR = "src/similar_indiv/rag/train_cra_v2"


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

# Reason generation
def reason_generator(x_merged_train_T0, x_train_T1):
    text_T0 = f"""
        Customer Profile at Time T0:
        - Demographic:
            - Age: {x_merged_train_T0['Age']},
            - Gender: {x_merged_train_T0['Gender']},
            - Education: {x_merged_train_T0['Education level']},
            - Marital Status: {x_merged_train_T0['Marital status']},
            - Occupation Group: {x_merged_train_T0['Occupation Group']},
            - Region: {x_merged_train_T0['Region']},
            - Number of Children: {x_merged_train_T0['Number of Children']},
        - Financial Situation and Banking Activity:
            - Number of Vehicles: {x_merged_train_T0['Number of Vehicles']},
            - Savings Account: {x_merged_train_T0['Savings Account']},
            - Savings Account Subgroup: {x_merged_train_T0['Savings Account Subgroup']},
            - Health Insurance: {x_merged_train_T0['Health Insurance']},
            - Lending: {x_merged_train_T0['Lending']},
            - Payment: {x_merged_train_T0['Payment']},
            - Service: {x_merged_train_T0['Service']},
            - Business Lending: {x_merged_train_T0['Business Lending']},
            - Deposit Account: {x_merged_train_T0['Deposit Account']},
            - Deposit Account Balance: {x_merged_train_T0['Deposit Account Balance']},
            - Deposit Account Transactions: {x_merged_train_T0['Deposit Account Transactions']},
            - Deposit Account Transactions AVG: {x_merged_train_T0['Deposit Account Transactions AVG']},
            - Deposit Account Transactions MIN: {x_merged_train_T0['Deposit Account Transactions MIN']},
            - Deposit Account Transactions MAX: {x_merged_train_T0['Deposit Account Transactions MAX']},
            - Deposit Account Inflow: {x_merged_train_T0['Deposit Account Inflow']},
            - Deposit Account Inflow MIN: {x_merged_train_T0['Deposit Account Inflow MIN']},
            - Deposit Account Inflow MAX: {x_merged_train_T0['Deposit Account Inflow MAX']},
            - Deposit Account Outflow: {x_merged_train_T0['Deposit Account Outflow']},
            - Deposit Account Outflow MIN: {x_merged_train_T0['Deposit Account Outflow MIN']},
            - Deposit Account Outflow MAX: {x_merged_train_T0['Deposit Account Outflow MAX']},
            - Deposit Account Inflow Amount: {x_merged_train_T0['Deposit Account Inflow Amount']},
            - Deposit Account Outflow Amount: {x_merged_train_T0['Deposit Account Outflow Amount']},
        - Summary: {x_merged_train_T0['Summary']}
        """
    text_T1 = f"""
        Observed Changes at Time T1:
        - Education: Changed from {x_merged_train_T0['Education level']} to {x_train_T1['Education level']}
        - Marital Status: Changed from {x_merged_train_T0['Marital status']} to {x_train_T1['Marital status']}
        - Occupation: Changed from {x_merged_train_T0['Occupation Group']} to {x_train_T1['Occupation Group']}
        - Region: Changed from {x_merged_train_T0['Region']} to {x_train_T1['Region']}
        - Number of Children: Changed from {x_merged_train_T0['Number of Children']} to {x_train_T1['Number of Children']}
    """

    
#     full_prompt = f"""
#     You are a financial behavior analyst. Analyze how a customer's initial situation (T0) may have led to their observed changes (T1).

#     Context at T0:
#     {text_T0}

#     Observed Changes at T1:
#     {text_T1}

#     Generate a detailed reasoning report that:
#     1. Identifies the most significant changes between T0 and T1
#     2. Explains plausible financial/life circumstances that could lead to these changes
#     3. Connects specific T0 factors that likely influenced each T1 outcome
#     4. Suggests possible underlying motivations or constraints
#     5. Estimates the financial impact of these changes

#     Structure your response with clear sections:
#     - Key Changes Identified
#     - Probable Causes and Triggers
#     - Financial Situation Analysis
#     - Behavioral Insights
#     - Predicted Future Implications

#     Provide specific, evidence-based reasoning using the data points provided.
# """
#     full_prompt = f"""
#     You are a financial behavior analyst. Analyze how a customer's initial situation (T0) may have led to their observed changes (T1).

#     Context at T0:
#     {text_T0}

#     Observed Changes at T1:
#     {text_T1}

#     Generate a reasoning report that:
#     1. Identifies the changes between T0 and T1
#     2. Explains plausible financial/life circumstances that could lead to these changes
#     3. Connects specific T0 factors that likely influenced each T1 outcome
#     4. Suggests possible underlying motivations or constraints
#     5. Estimates the financial impact of these changes

#     Structure your response with clear sections:
#     - Key Changes Identified
#     - Probable Causes and Triggers
#     - Financial Situation Analysis
#     - Behavioral Insights
#     - Predicted Future Implications

#     Provide specific, evidence-based reasoning using the data points provided. Respond in no more than 1500 tokens.
# """
    full_prompt = f"""
You are a financial behavior analyst. Analyze the customer's transition from T0 to T1 and provide a concise reasoning report.

Context at T0:
{text_T0}

Observed Changes at T1:
{text_T1}

Provide a focused analysis covering:

**Key Changes:** List the 3 most significant changes between T0 and T1

**Root Causes:** Explain the most likely life/financial circumstances driving these changes

**Financial Impact:** Assess the overall financial implications (positive/negative/neutral)

**Behavioral Pattern:** Describe the customer's financial behavior shift in 2-3 sentences

Keep your response under 200 words total. Focus on the most important insights with specific evidence from the data.
"""

    client = get_aoi_client()

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert financial behavior analyst."},
            {"role": "user", "content": full_prompt}
        ],
        temperature=0.7,
        max_tokens=1500
    )
    
    # return {
    #     "content": response.choices[0].message.content,
    #     "usage": dict(response.usage),
    #     # "id": response.id
    # }
    return response.choices[0].message.content

def merge_reasoning(merged_train_T0, train_T1, cust_id_col='CUST_ID'):    
    train_T0_with_reason = merged_train_T0.copy()
    total_rows = len(train_T0_with_reason)

    # Convert T1 data to dict with CUST_ID as key
    t1_dict = train_T1.set_index(cust_id_col).to_dict('index')
    
    # Columns to merge from T1 and their new names with _T1 suffix
    t1_columns = {
        'Education level': 'Education_T1',
        'Marital status': 'Marital_Status_T1',
        'Occupation Group': 'Occupation_Group_T1',
        'Region': 'Region_T1',
        'Number of Children': 'Number_of_Children_T1'
    }
    
    # Initialize new columns with appropriate dtypes
    for orig_col, new_col in t1_columns.items():
        # Get the dtype from the original T1 column
        dtype = train_T1[orig_col].dtype
        train_T0_with_reason[new_col] = pd.Series(dtype=dtype)
    
    print(f"\nGenerating reasons for {total_rows} customers...")
    
    for idx, row in train_T0_with_reason.iterrows():
        cust_id = row[cust_id_col]
        current = idx + 1
        
        if current % 10 == 0 or current == 1 or current == total_rows:
            print(f"Processing customer {current}/{total_rows} (ID: {cust_id})")
        
        if cust_id in t1_dict:
            t1_data = t1_dict[cust_id]
            
            # Merge T1 data into the row with proper type conversion
            for orig_col, new_col in t1_columns.items():
                if orig_col in t1_data:
                    # Convert to the appropriate dtype
                    train_T0_with_reason.at[idx, new_col] = train_T1[orig_col].dtype.type(t1_data[orig_col])
            
            # Generate reason
            reason = reason_generator(row, pd.Series(t1_data))
            train_T0_with_reason.at[idx, 'Reason'] = reason
        else:
            train_T0_with_reason.at[idx, 'Reason'] = "No matching T1 data found"
            print(f"No T1 data found for customer {cust_id}")
    
    print("✅ Reason generation completed")
    return train_T0_with_reason

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

checkpoint_dir = 'src/similar_indiv/rag/checkpoints'

def train_cra_prep(train_T0, train_T1, DIR):
    # testtest = train_T0.head()
    # second_q = train_T0.iloc[(len(train_T0)//4)*3:].copy()
    second_q = train_T0.iloc[0:len(train_T0)//4].copy()

    train_T0_with_summ = merge_summaries(second_q)
    save_checkpoint(train_T0_with_summ, checkpoint_dir, "q1_with_summaries")

    # train_T0_with_summ = pd.read_csv('src/similar_indiv/rag/checkpoints/01_with_summaries_20250627_224753.csv')

    result = merge_reasoning(train_T0_with_summ, train_T1)
    save_checkpoint(result, checkpoint_dir, "q1_with_reasons")

    save_csv_file(DIR, result, 'train_cra', next_version=None)

train_cra_prep(train_T0, train_T1, DIR)