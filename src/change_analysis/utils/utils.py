import pandas as pd
import os
from src.client.llm import get_aoi_client

def cluster_stats_info(df_no_outliers, col):
    df_group = df_no_outliers.groupby(col).agg(
        {
            # Original columns (for backward compatibility)
            'Age': ['mean', 'median'],
            'Gender': lambda x: x.value_counts().index[0],
            'Education level': lambda x: x.value_counts().index[0],
            'Marital status': lambda x: x.value_counts().index[0],
            'Occupation Group': lambda x: x.value_counts().index[0],
            'Region': lambda x: x.value_counts().index[0],
            'Number of Children': ['mean', 'median'],
            'Number of Vehicles': ['mean', 'median'],
            'Savings Account': ['mean', 'median'],
            'Savings Account Subgroup': ['mean', 'median'],
            'Health Insurance': ['mean', 'median'],
            'Lending': ['mean', 'median'],
            'Payment': ['mean', 'median'],
            'Service': ['mean', 'median'],
            'Business Lending': ['mean', 'median'],
            'Deposit Account': ['mean', 'median'],
            'Deposit Account Balance': ['mean', 'median'],
            'Deposit Account Transactions': ['mean', 'median'],
            'Deposit Account Transactions AVG': ['mean', 'median'],
            'Deposit Account Transactions MIN': ['mean', 'median'],
            'Deposit Account Transactions MAX': ['mean', 'median'],
            'Deposit Account Inflow': ['mean', 'median'],
            'Deposit Account Inflow MIN': ['mean', 'median'],
            'Deposit Account Inflow MAX': ['mean', 'median'],
            'Deposit Account Outflow': ['mean', 'median'],
            'Deposit Account Outflow MIN': ['mean', 'median'],
            'Deposit Account Outflow MAX': ['mean', 'median'],
            'Deposit Account Inflow Amount': ['mean', 'median'],
            'Deposit Account Outflow Amount': ['mean', 'median'],
        }
    )

    df_group.columns = ['_'.join(col).strip('_') for col in df_group.columns.values]

    df_group = df_group.reset_index()
    df_group.columns = df_group.columns.str.replace('_<lambda>', '', regex=False)

    # cluster_size
    cluster_counts = df_no_outliers[col].value_counts().reset_index()
    cluster_counts.columns = [col, 'cluster_size']

    df_group = pd.merge(df_group, cluster_counts, on=col)

    cols = [col, 'cluster_size'] + [c for c in df_group.columns if c not in [col, 'cluster_size']]
    df_group = df_group[cols]
    
    return df_group



# LLM

def format_cluster_stats_for_analysis(train_T1_df_group, train_T0_df_group, cluster_id, col):
    """Format cluster statistics into readable text for LLM analysis"""
    cluster_data_T0 = train_T0_df_group[train_T0_df_group[col] == cluster_id].iloc[0]
    cluster_data_T1 = train_T1_df_group[train_T1_df_group[col] == cluster_id].iloc[0]
    
    formatted_text = f"CLUSTER {cluster_id} CHANGE ANALYSIS:\n"
    formatted_text += "=" * 40 + "\n\n"
    
    # T0
    formatted_text += "## CUSTOMER PROFILE AT T0:\n"
    # Demographics
    formatted_text += "DEMOGRAPHICS:\n"
    formatted_text += f"• Average Age: {cluster_data_T0['Age_mean']:.1f} years\n"
    formatted_text += f"• Median Age: {cluster_data_T0['Age_median']:.1f} years\n"
    formatted_text += f"• Most common Gender: {cluster_data_T0['Gender']}\n"
    formatted_text += f"• Most common Education: {cluster_data_T0['Education level']}\n"
    formatted_text += f"• Most common Marital Status: {cluster_data_T0['Marital status']}\n"
    formatted_text += f"• Most common Occupation: {cluster_data_T0['Occupation Group']}\n"
    formatted_text += f"• Most common Region: {cluster_data_T0['Region']}\n"
    formatted_text += f"• Average Number of Children: {cluster_data_T0['Number of Children_mean']:.1f}\n"
    formatted_text += f"• Median Number of Children: {cluster_data_T0['Number of Children_median']:.1f}\n"
    formatted_text += f"• Average Number of Vehicles: {cluster_data_T0['Number of Vehicles_mean']:.1f}\n"
    formatted_text += f"• Median Number of Vehicles: {cluster_data_T0['Number of Vehicles_median']:.1f}\n\n"
    
    # Financial Products & Services
    formatted_text += "FINANCIAL PRODUCTS & SERVICES:\n"
    formatted_text += f"• Average Savings Account: {cluster_data_T0['Savings Account_mean']:.1%}\n"
    formatted_text += f"• Median Savings Account: {cluster_data_T0['Savings Account_median']:.1%}\n"
    formatted_text += f"• Average Savings Account Subgroup: {cluster_data_T0['Savings Account Subgroup_mean']:.1%}\n"
    formatted_text += f"• Median Savings Account Subgroup: {cluster_data_T0['Savings Account Subgroup_median']:.1%}\n"
    formatted_text += f"• Average Health Insurance: {cluster_data_T0['Health Insurance_mean']:.1%}\n"
    formatted_text += f"• Median Health Insurance: {cluster_data_T0['Health Insurance_median']:.1%}\n"
    formatted_text += f"• Average Lending: {cluster_data_T0['Lending_mean']:.1%}\n"
    formatted_text += f"• Median Lending: {cluster_data_T0['Lending_median']:.1%}\n"
    formatted_text += f"• Average Payment Services: {cluster_data_T0['Payment_mean']:.1%}\n"
    formatted_text += f"• Median Payment Services: {cluster_data_T0['Payment_median']:.1%}\n"
    formatted_text += f"• Average General Services: {cluster_data_T0['Service_mean']:.1%}\n"
    formatted_text += f"• Median General Services: {cluster_data_T0['Service_median']:.1%}\n"
    formatted_text += f"• Average Business Lending: {cluster_data_T0['Business Lending_mean']:.1%}\n"
    formatted_text += f"• Median Business Lending: {cluster_data_T0['Business Lending_median']:.1%}\n\n"
    
    # Deposit Account Behavior
    formatted_text += "DEPOSIT ACCOUNT BEHAVIOR:\n"
    formatted_text += f"• Average Deposit Account: {cluster_data_T0['Deposit Account_mean']:.1%}\n"
    formatted_text += f"• Median Deposit Account: {cluster_data_T0['Deposit Account_median']:.1%}\n"
    formatted_text += f"• Average Balance: ${cluster_data_T0['Deposit Account Balance_mean']:,.0f}\n"
    formatted_text += f"• Median Balance: ${cluster_data_T0['Deposit Account Balance_median']:,.0f}\n"
    formatted_text += f"• Average Monthly Transactions: {cluster_data_T0['Deposit Account Transactions_mean']:.1f}\n"
    formatted_text += f"• Median Monthly Transactions: {cluster_data_T0['Deposit Account Transactions_median']:.1f}\n"
    formatted_text += f"• Average Transaction Amount: ${cluster_data_T0['Deposit Account Transactions AVG_mean']:,.0f}\n"
    formatted_text += f"• Median Transaction Amount: ${cluster_data_T0['Deposit Account Transactions AVG_median']:,.0f}\n"
    formatted_text += f"• Average Min Transaction: ${cluster_data_T0['Deposit Account Transactions MIN_mean']:,.0f}\n"
    formatted_text += f"• Median Min Transaction: ${cluster_data_T0['Deposit Account Transactions MIN_median']:,.0f}\n"
    formatted_text += f"• Average Max Transaction: ${cluster_data_T0['Deposit Account Transactions MAX_mean']:,.0f}\n"
    formatted_text += f"• Median Max Transaction: ${cluster_data_T0['Deposit Account Transactions MAX_median']:,.0f}\n\n"
    
    # Cash Flow Patterns
    formatted_text += "CASH FLOW PATTERNS:\n"
    formatted_text += f"• Average Monthly Inflows: {cluster_data_T0['Deposit Account Inflow_mean']:.1f}\n"
    formatted_text += f"• Median Monthly Inflows: {cluster_data_T0['Deposit Account Inflow_median']:.1f}\n"
    formatted_text += f"• Average Min Inflow: ${cluster_data_T0['Deposit Account Inflow MIN_mean']:,.0f}\n"
    formatted_text += f"• Median Min Inflow: ${cluster_data_T0['Deposit Account Inflow MIN_median']:,.0f}\n"
    formatted_text += f"• Average Max Inflow: ${cluster_data_T0['Deposit Account Inflow MAX_mean']:,.0f}\n"
    formatted_text += f"• Median Max Inflow: ${cluster_data_T0['Deposit Account Inflow MAX_median']:,.0f}\n"
    formatted_text += f"• Average Monthly Outflows: {cluster_data_T0['Deposit Account Outflow_mean']:.1f}\n"
    formatted_text += f"• Median Monthly Outflows: {cluster_data_T0['Deposit Account Outflow_median']:.1f}\n"
    formatted_text += f"• Average Min Outflow: ${cluster_data_T0['Deposit Account Outflow MIN_mean']:,.0f}\n"
    formatted_text += f"• Median Min Outflow: ${cluster_data_T0['Deposit Account Outflow MIN_median']:,.0f}\n"
    formatted_text += f"• Average Max Outflow: ${cluster_data_T0['Deposit Account Outflow MAX_mean']:,.0f}\n"
    formatted_text += f"• Median Max Outflow: ${cluster_data_T0['Deposit Account Outflow MAX_median']:,.0f}\n"
    formatted_text += f"• Average Total Inflow Amount: ${cluster_data_T0['Deposit Account Inflow Amount_mean']:,.0f}\n"
    formatted_text += f"• Median Total Inflow Amount: ${cluster_data_T0['Deposit Account Inflow Amount_median']:,.0f}\n"
    formatted_text += f"• Average Total Outflow Amount: ${cluster_data_T0['Deposit Account Outflow Amount_mean']:,.0f}\n"
    formatted_text += f"• Median Total Outflow Amount: ${cluster_data_T0['Deposit Account Outflow Amount_median']:,.0f}\n\n"

    # T1
    formatted_text += "## CUSTOMER PROFILE AT T1:\n"
    # Demographics
    formatted_text += "DEMOGRAPHICS:\n"
    formatted_text += f"• Average Age: {cluster_data_T1['Age_mean']:.1f} years\n"
    formatted_text += f"• Median Age: {cluster_data_T1['Age_median']:.1f} years\n"
    formatted_text += f"• Most common Gender: {cluster_data_T1['Gender']}\n"
    formatted_text += f"• Most common Education: {cluster_data_T1['Education level']}\n"
    formatted_text += f"• Most common Marital Status: {cluster_data_T1['Marital status']}\n"
    formatted_text += f"• Most common Occupation: {cluster_data_T1['Occupation Group']}\n"
    formatted_text += f"• Most common Region: {cluster_data_T1['Region']}\n"
    formatted_text += f"• Average Number of Children: {cluster_data_T1['Number of Children_mean']:.1f}\n"
    formatted_text += f"• Median Number of Children: {cluster_data_T1['Number of Children_median']:.1f}\n"
    formatted_text += f"• Average Number of Vehicles: {cluster_data_T1['Number of Vehicles_mean']:.1f}\n"
    formatted_text += f"• Median Number of Vehicles: {cluster_data_T1['Number of Vehicles_median']:.1f}\n\n"
    
    # Financial Products & Services
    formatted_text += "FINANCIAL PRODUCTS & SERVICES:\n"
    formatted_text += f"• Average Savings Account: {cluster_data_T1['Savings Account_mean']:.1%}\n"
    formatted_text += f"• Median Savings Account: {cluster_data_T1['Savings Account_median']:.1%}\n"
    formatted_text += f"• Average Savings Account Subgroup: {cluster_data_T1['Savings Account Subgroup_mean']:.1%}\n"
    formatted_text += f"• Median Savings Account Subgroup: {cluster_data_T1['Savings Account Subgroup_median']:.1%}\n"
    formatted_text += f"• Average Health Insurance: {cluster_data_T1['Health Insurance_mean']:.1%}\n"
    formatted_text += f"• Median Health Insurance: {cluster_data_T1['Health Insurance_median']:.1%}\n"
    formatted_text += f"• Average Lending: {cluster_data_T1['Lending_mean']:.1%}\n"
    formatted_text += f"• Median Lending: {cluster_data_T1['Lending_median']:.1%}\n"
    formatted_text += f"• Average Payment Services: {cluster_data_T1['Payment_mean']:.1%}\n"
    formatted_text += f"• Median Payment Services: {cluster_data_T1['Payment_median']:.1%}\n"
    formatted_text += f"• Average General Services: {cluster_data_T1['Service_mean']:.1%}\n"
    formatted_text += f"• Median General Services: {cluster_data_T1['Service_median']:.1%}\n"
    formatted_text += f"• Average Business Lending: {cluster_data_T1['Business Lending_mean']:.1%}\n"
    formatted_text += f"• Median Business Lending: {cluster_data_T1['Business Lending_median']:.1%}\n\n"
    
    # Deposit Account Behavior
    formatted_text += "DEPOSIT ACCOUNT BEHAVIOR:\n"
    formatted_text += f"• Average Deposit Account: {cluster_data_T1['Deposit Account_mean']:.1%}\n"
    formatted_text += f"• Median Deposit Account: {cluster_data_T1['Deposit Account_median']:.1%}\n"
    formatted_text += f"• Average Balance: ${cluster_data_T1['Deposit Account Balance_mean']:,.0f}\n"
    formatted_text += f"• Median Balance: ${cluster_data_T1['Deposit Account Balance_median']:,.0f}\n"
    formatted_text += f"• Average Monthly Transactions: {cluster_data_T1['Deposit Account Transactions_mean']:.1f}\n"
    formatted_text += f"• Median Monthly Transactions: {cluster_data_T1['Deposit Account Transactions_median']:.1f}\n"
    formatted_text += f"• Average Transaction Amount: ${cluster_data_T1['Deposit Account Transactions AVG_mean']:,.0f}\n"
    formatted_text += f"• Median Transaction Amount: ${cluster_data_T1['Deposit Account Transactions AVG_median']:,.0f}\n"
    formatted_text += f"• Average Min Transaction: ${cluster_data_T1['Deposit Account Transactions MIN_mean']:,.0f}\n"
    formatted_text += f"• Median Min Transaction: ${cluster_data_T1['Deposit Account Transactions MIN_median']:,.0f}\n"
    formatted_text += f"• Average Max Transaction: ${cluster_data_T1['Deposit Account Transactions MAX_mean']:,.0f}\n"
    formatted_text += f"• Median Max Transaction: ${cluster_data_T1['Deposit Account Transactions MAX_median']:,.0f}\n\n"
    
    # Cash Flow Patterns
    formatted_text += "CASH FLOW PATTERNS:\n"
    formatted_text += f"• Average Monthly Inflows: {cluster_data_T1['Deposit Account Inflow_mean']:.1f}\n"
    formatted_text += f"• Median Monthly Inflows: {cluster_data_T1['Deposit Account Inflow_median']:.1f}\n"
    formatted_text += f"• Average Min Inflow: ${cluster_data_T1['Deposit Account Inflow MIN_mean']:,.0f}\n"
    formatted_text += f"• Median Min Inflow: ${cluster_data_T1['Deposit Account Inflow MIN_median']:,.0f}\n"
    formatted_text += f"• Average Max Inflow: ${cluster_data_T1['Deposit Account Inflow MAX_mean']:,.0f}\n"
    formatted_text += f"• Median Max Inflow: ${cluster_data_T1['Deposit Account Inflow MAX_median']:,.0f}\n"
    formatted_text += f"• Average Monthly Outflows: {cluster_data_T1['Deposit Account Outflow_mean']:.1f}\n"
    formatted_text += f"• Median Monthly Outflows: {cluster_data_T1['Deposit Account Outflow_median']:.1f}\n"
    formatted_text += f"• Average Min Outflow: ${cluster_data_T1['Deposit Account Outflow MIN_mean']:,.0f}\n"
    formatted_text += f"• Median Min Outflow: ${cluster_data_T1['Deposit Account Outflow MIN_median']:,.0f}\n"
    formatted_text += f"• Average Max Outflow: ${cluster_data_T1['Deposit Account Outflow MAX_mean']:,.0f}\n"
    formatted_text += f"• Median Max Outflow: ${cluster_data_T1['Deposit Account Outflow MAX_median']:,.0f}\n"
    formatted_text += f"• Average Total Inflow Amount: ${cluster_data_T1['Deposit Account Inflow Amount_mean']:,.0f}\n"
    formatted_text += f"• Median Total Inflow Amount: ${cluster_data_T1['Deposit Account Inflow Amount_median']:,.0f}\n"
    formatted_text += f"• Average Total Outflow Amount: ${cluster_data_T1['Deposit Account Outflow Amount_mean']:,.0f}\n"
    formatted_text += f"• Median Total Outflow Amount: ${cluster_data_T1['Deposit Account Outflow Amount_median']:,.0f}\n\n"
    
    return formatted_text

def analyze_clusters_with_llm(train_T1_df_group, train_T0_df_group, col):
    """Use LLM to analyze clusters and generate descriptions"""
    client = get_aoi_client()
    
    # Get all cluster statistics
    all_cluster_stats = []
    cluster_ids = sorted(train_T0_df_group[col].unique())
    
    for cluster_id in cluster_ids:
        formatted_stats = format_cluster_stats_for_analysis(train_T1_df_group, train_T0_df_group, cluster_id, col)
        all_cluster_stats.append(formatted_stats)
    
    # First, identify key differentiating factors
    comparison_text = "\n".join(all_cluster_stats)
    
    differentiator_prompt = """Analyze these customer cluster profiles at time T0 and T1 and identify the differences in how these customer segments tend to change in T1 compared to T0:

{cluster_stats}

Focus on the most significant changes in:
- Demographics (age, education, occupation, family size)
- Financial behavior (account balances, transaction patterns, cash flows)
- Product usage (savings, lending, insurance, services)
""".format(cluster_stats=comparison_text)

    # Use CHAT COMPLETIONS API for GPT-4o
    differentiator_response = client.chat.completions.create(
        model="gpt-4o",  # Your GPT-4o deployment name
        messages=[
            {"role": "system", "content": "You are a data analyst who identifies key differences between customer segments' change trends from T0 to T1"},
            {"role": "user", "content": differentiator_prompt}
        ],
        temperature=0.3,
        max_tokens=200
    )
    differentiators = differentiator_response.choices[0].message.content
    print(f"Key change trends: {differentiators}\n")
    
    # Analyze each cluster individually
    cluster_prompt_template = """Based on this customer profile at time T0 and T1, provide concise bdifferences in how this customer segment tends to change in T1 compared to T0:

{cluster_stats}

Key differentiating factors to consider: {differentiators}

Provide a 2-3 sentence description that captures:
1. The change in demographic profile of this segment from T0 to T1
2. The change in their key financial behaviors and product usage patterns from T0 to T1

Write in a business-friendly tone suitable for marketing and customer strategy."""

    print("Generating cluster descriptions...")
    descriptions = []
    
    for cluster_id in cluster_ids:
        print(f"Analyzing Cluster {cluster_id}...")
        formatted_stats = format_cluster_stats_for_analysis(train_T1_df_group, train_T0_df_group, cluster_id, col)
        
        prompt = cluster_prompt_template.format(
            cluster_stats=formatted_stats,
            differentiators=differentiators
        )
        
        response = client.chat.completions.create(
            model="gpt-4o",  # Your GPT-4o deployment name
            messages=[
                {"role": "system", "content": "You are a marketing analyst describing customer segments change analysis from time T0 to T1."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=300
        )
        
        descriptions.append(response.choices[0].message.content.strip())
    
    return descriptions, differentiators

def add_llm_descriptions_to_clusters(train_T1_df_group, train_T0_df_group, col):
    """Add LLM-generated descriptions to the cluster DataFrame"""
    # Generate descriptions using LLM
    descriptions, differentiators = analyze_clusters_with_llm(train_T1_df_group, train_T0_df_group, col)
    
    # Create a mapping of cluster to description
    cluster_ids = sorted(train_T0_df_group[col].unique())
    description_mapping = dict(zip(cluster_ids, descriptions))
    
    # Add description column to dataframe
    df_group_with_descriptions = train_T0_df_group.copy()
    df_group_with_descriptions['change_analysis_description'] = df_group_with_descriptions[col].map(description_mapping)
    df_group_with_descriptions = df_group_with_descriptions[[col, 'change_analysis_description']]
    
    return df_group_with_descriptions, differentiators

# def add_descriptions_simple(df_group, col):
#     """Simplified version to just add descriptions to existing df_group"""
#     try:
#         descriptions, differentiators = analyze_clusters_with_llm(df_group, col)
#         df_group['description'] = df_group[col].map(dict(zip(sorted(df_group[col].unique()), descriptions)))
#         print("✅ Descriptions added successfully!")
#         print(f"\nKey differentiators: {differentiators}")
#         return df_group
#     except Exception as e:
#         print(f"❌ Error: {e}")
#         return df_group

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