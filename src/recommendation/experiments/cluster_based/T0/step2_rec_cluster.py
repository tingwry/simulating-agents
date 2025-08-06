import pandas as pd
import joblib
from tqdm import tqdm  # For progress bars
import time
from src.clustering.approach_2_embed.utils.utils import *
from src.client.llm import get_aoi_client
import json


# cold_cust = pd.read_csv('src/data/summary_reasoning/test_summ_v1.csv')
cold_cust = pd.read_csv('src/data/T0/test_with_lifestyle.csv')
# cold_cust = cold_cust_df.head(1)

warm_cust_df = pd.read_csv('src/clustering/approach_2_embed/result/full_data_with_cluster/full_data_with_cluster_v4.csv')

lifestyle = pd.read_csv('src/data/preprocessed_data/lifestyle.csv')

all_cluster_stats = pd.read_csv('src/recommendation/cluster_based/T0/data/cluster_transaction_stats.csv')

MODEL_DIR = "src/clustering/approach_2_embed/model/model_app2_v4.pkl"
clus_model = joblib.load(MODEL_DIR)


def get_cluster_desc():
    cluster_descriptions = {
        "cluster_0_descriptions" : "Cluster 0 represents mid-career, single female corporate employees, predominantly in their late 30s, with a high school education and minimal family or vehicle ownership. This segment exhibits low engagement with financial products, limited savings account usage, and negligible cash flow activity, suggesting a preference for basic financial services and a conservative approach to money management. Unlike other segments, their financial behavior is characterized by minimal transactions and inflows, making them distinct as a low-activity, low-adoption customer group.",

        "cluster_1_descriptions" : "Cluster 1 represents mid-career, highly educated professionals, predominantly single women in their early-to-mid 40s, residing in central regions. This segment demonstrates strong financial stability with high savings account balances and significant adoption of health insurance and payment services, though they maintain low transaction amounts and limited vehicle or family-related expenses. Distinct from other segments, Cluster 1 combines a focus on savings and financial security with modest cash flow activity, reflecting a lifestyle centered on individual priorities and professional growth.",

        "cluster_2_descriptions" : "Cluster 2 represents young, single, predominantly female corporate employees in their early 30s, primarily located in the Central region. They exhibit high engagement with savings accounts and payment services but maintain low balances, modest inflows, and minimal lending or vehicle ownership, reflecting a transactional, cash-flow-focused financial behavior. Distinct from other segments, their high transaction volumes and limited product diversification align with their life stage and education level, making them an ideal target for streamlined, digital-first financial solutions that prioritize convenience and affordability.",

        'cluster_3_descriptions' : "Cluster 3 represents a predominantly young, single, female demographic, with an average age of 35.6 years and a concentration of corporate employees in central regions. This segment exhibits minimal engagement with financial products, low savings adoption (13.8% average), and negligible cash flow activity, with average inflows of $19 and outflows of $394. Distinct from other clusters, Cluster 3 is characterized by limited financial activity and product usage, likely reflecting a lifestyle focused on immediate needs rather than long-term financial planning.",

        'cluster_4_descriptions' : "Cluster 4 represents a mature, predominantly female segment of entrepreneurial professionals, averaging 47.8 years old and typically holding a bachelor's degree. They exhibit high balances and inflows, with a strong preference for savings accounts, health insurance, and payment services, while engaging minimally in business lending. Distinct from younger clusters, this group's financial behavior is characterized by stability, moderate transaction volumes, and a focus on wealth preservation rather than high-frequency spending or borrowing.",

        'cluster_5_descriptions' : "Cluster 5 represents middle-aged, predominantly female entrepreneurs with high school education, primarily located in the Central region. This segment is characterized by high savings account balances and strong usage of payment services, while exhibiting conservative lending and business credit behaviors. Distinct from other segments, they maintain significant cash inflows and outflows with high average balances, yet their transactional activity remains modest, reflecting a financially stable but cautious approach to financial product adoption.",

        'cluster_6_descriptions' : "Cluster 6 represents single, male corporate employees in their late 30s, predominantly with a high school education and residing in central regions. This segment demonstrates moderate savings account usage and payment services adoption, with low engagement in lending and general services. What sets them apart is their relatively high average deposit balances and inflow amounts, despite minimal transaction activity and limited financial product diversification, indicating a preference for simplicity and stability in financial management.",

        'cluster_7_descriptions' : "Cluster 7 represents young, single, male corporate employees, predominantly located in central regions, with limited vehicle ownership and no children. This segment exhibits high usage of savings accounts and payment services, paired with low balances and inflows but frequent, small-value transactions. Distinct from older and more educated clusters, Cluster 7 prioritizes transactional convenience over diverse financial product adoption, making them ideal for streamlined, digital-first solutions.",

        'cluster_8_descriptions' : "Cluster 8 represents financially active, single male professionals in their mid-30s, primarily holding vocational certificates and working as corporate employees in central regions. This segment demonstrates strong adoption of savings accounts and payment services, with moderate health insurance usage, but minimal engagement with lending or business-related financial products. Distinct from other segments, they exhibit steady cash flow patterns with relatively high transaction volumes but low transaction amounts, reflecting a preference for frequent, small-scale financial activity.",

        'cluster_9_descriptions' : "Cluster 9 represents middle-aged, predominantly single, female freelancers with a high school education, residing in central regions. This segment demonstrates moderate engagement with savings accounts and payment services but limited adoption of lending and general financial products, with low transaction volumes and minimal cash flow activity despite occasional high average balances. Distinct from other clusters, Cluster 9's financial behavior reflects a conservative approach to spending and borrowing, likely influenced by their freelance occupation and modest household structure.",

        'cluster_10_descriptions' : "Cluster 10 represents mid-career, single corporate employees, predominantly female, with a high school education and residing in central regions. This segment exhibits moderate financial engagement, favoring savings accounts and payment services while showing limited activity in lending and business-related products. Distinct from other clusters, their cash flow patterns reveal modest inflows and outflows with low transaction amounts, suggesting a conservative financial approach and minimal discretionary spending.",
    }

    return cluster_descriptions


def format_stats_for_prompt(cluster_id, txn_cat, all_cluster_stats):
    """
    Formats statistics from all_cluster_stats DataFrame into natural language for LLM prompt.
    """
    stats_row = all_cluster_stats[
        (all_cluster_stats['cluster_id'] == cluster_id) & 
        (all_cluster_stats['transaction_category'] == txn_cat)
    ]
    
    if stats_row.empty:
        return f"No data available for cluster {cluster_id} and transaction category {txn_cat}"
    
    # Extract values from the row
    stats = stats_row.iloc[0]
    customer_count = stats['customer_count']
    total_in_cluster = stats['total_in_cluster']
    
    stats_text = [
        # f"{customer_count} customer(s) from a total of {total_in_cluster} customers in the cluster"
    ]
    
    if customer_count > 0:
        stats_text.extend([
            f"Average age: {stats['avg_age']:.1f} years",
            f"Average number of children: {stats.get('avg_number_of_children', 'N/A')}"
        ])
        
        def add_category_stats(prefix, name):
            parts = []
            if f"{prefix}_top_1" in stats:
                parts.append(f"{stats[f'{prefix}_top_1_pct']:.0%} {stats[f'{prefix}_top_1']}")
                if f"{prefix}_top_2" in stats and pd.notna(stats[f"{prefix}_top_2"]):
                    parts.append(f"{stats[f'{prefix}_top_2_pct']:.0%} {stats[f'{prefix}_top_2']}")
                    if f"{prefix}_top_3" in stats and pd.notna(stats[f"{prefix}_top_3"]):
                        parts.append(f"{stats[f'{prefix}_top_3_pct']:.0%} {stats[f'{prefix}_top_3']}")
            return ", ".join(parts)
        
        # Add education stats
        edu_stats = add_category_stats("education_level", "Education")
        if edu_stats:
            stats_text.append(f"Education: {edu_stats}")
        
        # Add marital status stats
        marital_stats = add_category_stats("marital_status", "Marital status")
        if marital_stats:
            stats_text.append(f"Marital status: {marital_stats}")
        
        # Add region stats
        region_stats = add_category_stats("region", "Region")
        if region_stats:
            stats_text.append(f"Region: {region_stats}")
        
        # Add occupation stats
        occupation_stats = add_category_stats("occupation_group", "Occupation")
        if occupation_stats:
            stats_text.append(f"Occupation: {occupation_stats}")
    
    return "\n- ".join(stats_text)

def format_cold_cust_demog(cold_cust_row):
    """
    Formats the cold customer's demographic information for the LLM prompt.
    """
    education = cold_cust_row.get('Education level')
    marital_status = cold_cust_row.get('Marital status')
    age = cold_cust_row.get('Age')
    children = cold_cust_row.get('Number of Children')
    occupation = cold_cust_row.get('Occupation Group')
    region = cold_cust_row.get('Region')
    
    demog_info = [
        f"Education: {education}",
        f"Marital status: {marital_status}",
        f"Age: {age} years",
        f"Number of children: {children}",
        f"Occupation: {occupation}",
        f"Region: {region}"
    ]
    
    return ", ".join(demog_info)



def get_llm_pred(cold_cust_row, cluster_id, txn_cat):
    client = get_aoi_client()

    cluster_descriptions = get_cluster_desc()
    cluster_desc = cluster_descriptions.get(f"cluster_{cluster_id}_descriptions", 
                                          f"Cluster {cluster_id} description not found")
    
    formatted_stats = format_stats_for_prompt(cluster_id, txn_cat, all_cluster_stats)
    cold_cust_demog = format_cold_cust_demog(cold_cust_row)

    prompt = f"""
    You are an expert customer behavior analyst specializing in financial transaction prediction. 
    Your task is to predict whether a customer will engage with a specific transaction category based on their demographic profile and cluster analysis.
    
    ## Customer Profile:
    {cold_cust_demog}

    ## Cluster Assignment:
    The customer has been assigned to Cluster {cluster_id} based on their demographic and behavioral similarities.
    
    ## Cluster Characteristics:
    {cluster_desc}

    ## Cluster Transaction Patterns:
    Among customers in Cluster {cluster_id}, the transaction category "{txn_cat}" shows the following patterns:
    {formatted_stats}

    ## Analysis Framework:
    Consider the following factors in your prediction:
    1. Demographic alignment: How well does the customer's profile match those who typically engage with {txn_cat}?
    2. Cluster behavior patterns: What does the cluster's historical engagement with {txn_cat} suggest?
    3. Life stage indicators: Does the customer's age, marital status, and family situation align with {txn_cat} engagement?
    4. Geographic and occupational factors: How might the customer's region and occupation influence their likelihood to engage with {txn_cat}?
    5. Financial behavior context: Consider the cluster's overall financial activity patterns and risk tolerance.

    ## Prediction Task:
    Based on the customer's demographic profile and their cluster's behavioral patterns, predict whether this customer will engage with the transaction category "{txn_cat}".

    ## Response Requirements:
    Provide your response in the following JSON format:
    {{
        "answer": "Yes" or "No",
        "reasoning": "Provide a detailed explanation of your prediction, referencing specific demographic factors, cluster patterns, and behavioral indicators that support your conclusion. Include both supporting and contradicting factors if applicable."
    }}

    ## Important Guidelines:
    - Base your prediction on data-driven insights from the cluster analysis
    - Consider both demographic fit and behavioral patterns
    - Be specific about which factors most strongly influence your prediction
    - Acknowledge uncertainty when the evidence is mixed
    - Provide actionable insights that could inform marketing or product strategies
"""

    messages = [
        {"role": "system", "content": "You are a senior customer behavior analyst with expertise in demographic segmentation and transaction prediction. You provide data-driven insights with clear reasoning and measurable confidence levels."},
        {"role": "user", "content": prompt}
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.7,
        max_tokens=1500
    )
    
    return response.choices[0].message.content


def rec_cluster(cold_cust, clus_model, txn_categories):
    print("\n" + "="*50)
    print("Starting cluster-based recommendation process")
    print(f"Processing {len(cold_cust)} customers for {len(txn_categories)} transaction categories")
    print("="*50 + "\n")
    
    start_time = time.time()
    
    # Create embeddings and predict clusters
    print("Step 1/4: Creating customer embeddings...")
    df_embedding = embedding_creation(cold_cust)
    print(f"Embeddings created for {len(df_embedding)} customers")
    
    print("\nStep 2/4: Predicting clusters...")
    predicted_clus_df = predict(cold_cust, df_embedding, clus_model)
    cluster_counts = predicted_clus_df['cluster'].value_counts()
    print("Cluster distribution:")
    print(cluster_counts.to_string())
    
    # Initialize response DataFrame
    print("\nStep 3/4: Initializing response dataframe...")
    response_df = pd.DataFrame()
    response_df['cust_id'] = predicted_clus_df['CUST_ID']
    
    # Add all transaction categories as columns initialized to 0
    for cat in txn_categories:
        response_df[cat] = 0
        response_df[f'{cat}_reasoning'] = ''
    
    print(f"Initialized dataframe with {len(response_df)} customers and {len(txn_categories)} categories")
    
    # Process each customer with progress bar
    print("\nStep 4/4: Getting LLM predictions (this may take a while)...")
    for idx, row in tqdm(predicted_clus_df.iterrows(), total=len(predicted_clus_df), desc="Processing customers"):
        cust_id = row['CUST_ID']
        clus_id = row['cluster']
        
        # Print cluster assignment for first few customers
        if idx < 3:  # Print for first 3 customers as sample
            print(f"\nSample customer {idx+1}/{len(predicted_clus_df)}: CUST_ID {cust_id} → Cluster {clus_id}")
        
        # Get predictions for each transaction category
        for txn_cat in txn_categories:
            try:
                if idx < 3:  # Print sample processing
                    print(f"  Processing {txn_cat}...", end=" ")
                
                pred_response = get_llm_pred(row, clus_id, txn_cat)
                
                # Try to parse as JSON
                try:
                    pred_data = json.loads(pred_response)
                    answer = pred_data.get('answer', '').lower()
                    reasoning = pred_data.get('reasoning', pred_response)
                except json.JSONDecodeError:
                    answer = 'yes' if 'yes' in pred_response.lower() else 'no'
                    reasoning = pred_response
                
                # Update response dataframe
                if answer == 'yes':
                    response_df.loc[response_df['cust_id'] == cust_id, txn_cat] = 1
                response_df.loc[response_df['cust_id'] == cust_id, f'{txn_cat}_reasoning'] = reasoning
                
                if idx < 3:  # Print sample result
                    print(f"Prediction: {answer.upper()}")
                    
            except Exception as e:
                error_msg = f"Error processing customer {cust_id} for {txn_cat}: {str(e)}"
                response_df.loc[response_df['cust_id'] == cust_id, f'{txn_cat}_reasoning'] = error_msg
                if idx < 3:  # Print sample error
                    print(f"ERROR: {error_msg}")
                continue
    
    # Final summary
    elapsed_time = time.time() - start_time
    print("\n" + "="*50)
    print("Processing complete!")
    print(f"Total time: {elapsed_time:.2f} seconds")
    print(f"Average time per customer: {elapsed_time/len(predicted_clus_df):.2f} seconds")
    
    # Show prediction statistics
    print("\nPrediction Summary:")
    for txn_cat in txn_categories[:5]:  # Show first 5 categories for brevity
        pred_count = response_df[txn_cat].sum()
        print(f"{txn_cat}: {pred_count} positive predictions ({pred_count/len(response_df):.1%})")
    
    return response_df


# Define transaction categories
txn_cat = ['charity', 'loan', 'utility', 'investment', 'finance', 'shopping',
           'personal_care', 'medical', 'home_and_living', 'insurance', 'automotive',
           'restaurant', 'business', 'entertainment', 'bank', 'education', 'pet_care',
           'government', 'travel', 'transportation', 'visit', 'system_dpst']

# Process customers (using head(1) for testing)
cold_cust = cold_cust.head(3)

# Generate predictions with reasoning
response_df = rec_cluster(cold_cust, clus_model, txn_cat)

# Save to CSV
response_df.to_csv('src/recommendation/cluster_based/T0/predictions/transaction_predictions.csv', index=False)

print("Predictions saved with reasoning columns!")
print(f"DataFrame shape: {response_df.shape}")
print(f"Columns: {list(response_df.columns)}")
print(response_df.head())