from src.client.llm import get_aoi_client
from openai import AzureOpenAI
from typing import Optional, Dict, Any
import os
from dotenv import load_dotenv
import pandas as pd
import json
import re

load_dotenv()

cluster_descriptions = {
    "cluster_0_descriptions" : "Cluster 0 represents a mature, financially engaged customer segment primarily composed of married, educated women in their mid-40s, often employed in corporate roles and residing in central regions. They exhibit high balances in savings accounts and robust usage of payment services and health insurance products, paired with steady cash flow patterns characterized by significant inflows and outflows. Distinct from younger clusters, this segment’s financial behavior reflects stability, higher product adoption, and a preference for savings and lending solutions, making them an ideal target for premium financial services and long-term wealth management offerings.",

    "cluster_1_descriptions" : "Cluster 1 represents a predominantly single, male demographic with an average age of 37, high school education, and employment in corporate roles, primarily located in the Central region. Financially, this segment exhibits moderate savings account engagement, low lending activity, and limited use of insurance and general services, with relatively small account balances and transaction amounts. Distinct from older and married clusters, Cluster 1’s financial behavior reflects lower overall inflows, outflows, and product usage intensity, making them a cost-conscious, transactional-focused segment with minimal long-term financial commitments.",

    "cluster_2_descriptions" : "Cluster 2 represents a younger, predominantly single, female demographic with an average age of 32, most commonly employed in corporate roles and residing in central regions. This segment demonstrates moderate financial engagement, characterized by high savings account usage and payment services activity, but minimal lending or business product adoption. Distinct from older, married segments with higher balances and inflows, Cluster 2’s financial behavior reflects a focus on transactional convenience and short-term cash flow management rather than long-term financial commitments.",

    'cluster_3_descriptions' : "Cluster 3 represents middle-aged, highly educated, entrepreneurial women, predominantly single, living in central regions with minimal family or vehicle commitments. This segment is characterized by high financial engagement, including substantial savings account balances, frequent payment service usage, and moderate health insurance adoption, but limited lending activity. Distinct from other segments, they exhibit strong cash flow patterns with significant inflows and outflows, reflecting their entrepreneurial nature and financial independence.",

    'cluster_4_descriptions' : "Cluster 4 represents predominantly single, high-school-educated females in their early 40s, often working as freelancers and residing in central regions. This segment exhibits modest financial engagement, characterized by low balances, minimal lending activity, and limited insurance usage, though they show higher-than-average adoption of savings accounts and payment services. Distinct from older, married, and more affluent clusters, Cluster 4’s financial behavior reflects a lean cash flow pattern and a preference for straightforward financial products over complex offerings.",

    'cluster_5_descriptions' : "Cluster 5 represents financially independent, single, and highly educated women in their late 30s, primarily employed in corporate roles and residing in central regions. This segment demonstrates moderate financial engagement, with strong savings account usage (above average balances) and high adoption of payment services, while showing minimal reliance on lending or business financial products. Distinct from older, married segments with higher inflows and outflows, Cluster 5 is characterized by lower transaction amounts and a conservative approach to cash flow management, reflecting a focus on financial stability over credit dependency.",

    'cluster_6_descriptions' : "Cluster 6 represents financially stable, middle-aged professionals, predominantly married males with bachelor’s degrees, working in corporate roles and residing in central regions. This segment is characterized by high average savings balances, strong inflows and outflows, and active usage of savings, payment, and health insurance products, while maintaining minimal engagement with business lending. Distinct from younger or less financially engaged segments, Cluster 6 stands out for its higher financial sophistication, consistent cash flow patterns, and preference for personal financial growth over entrepreneurial activities.",

    'cluster_7_descriptions' : "Cluster 7 represents predominantly single, high-school-educated women in their late 30s, primarily employed in corporate roles and residing in central regions. Financially, they exhibit moderate engagement with savings and payment services, maintaining relatively low account balances and transaction amounts, with minimal reliance on lending or business financial products. This segment is distinct for its lower financial complexity, limited cash flow activity, and a preference for straightforward financial solutions, making them ideal candidates for simplified, high-value savings and payment offerings.",

    'cluster_8_descriptions' : "Cluster 8 represents young, single, predominantly male corporate employees, with an average age of 33.7 years and limited financial responsibilities, as evidenced by no children and minimal vehicle ownership. This segment demonstrates moderate engagement with financial products, prioritizing savings (200%+ of average) and payment services, while showing low activity in lending and general services. Distinct from older, married segments, Cluster 8 is characterized by lower account balances, smaller cash flow volumes, and a transactional focus on smaller, frequent payments rather than substantial financial commitments."
}

t_diff = '1'

change_analysis_descriptions = {
    "cluster_0_descriptions" : "From T0 to T1, the demographic profile of Cluster 0 remains largely stable, with a minor increase in the average number of children (from 0.2 to 0.3), suggesting a slight shift toward larger families. Financial behaviors, however, show notable changes: deposit account usage and average balances increased significantly, while median balances and transactions decreased, indicating growing wealth concentration and more variability in transaction activity. These shifts highlight evolving financial engagement patterns, with higher-value customers driving growth.",

    "cluster_1_descriptions" : "From T0 to T1, the demographic profile of Cluster 1 remains stable, with no notable changes in age, marital status, education, or occupation. However, their financial behaviors and product usage patterns show significant shifts: there is a marked increase in savings account usage (average up from 196.7% to 315.1%), payment services (average up from 148.0% to 214.6%), and health insurance adoption (average up from 77.4% to 140.5%). Additionally, while average deposit account balances and monthly transactions have grown, median values for both have declined, indicating greater disparity in financial activity within the segment.",

    "cluster_2_descriptions" : "From T0 to T1, the demographic profile of Cluster 2 remained stable, with no notable changes in age, marital status, or occupation. However, their financial behaviors and product usage patterns shifted significantly, with a marked increase in savings account balances (average up from 220.0% to 285.7%), health insurance adoption (average up from 86.9% to 139.7%), and payment services usage (average up from 156.1% to 179.2%). Additionally, while average deposit account balances grew (from $15,390 to $19,172), monthly transactions decreased (average down from 71.8 to 55.7), indicating a shift toward higher savings and reduced transaction frequency.",

    'cluster_3_descriptions' : "From T0 to T1, Cluster 3's demographic profile remained stable, with no notable changes in age, gender, education, marital status, or family size. However, their financial behaviors shifted, showing increased engagement with savings accounts (average up to 397.8%) and health insurance (average up to 247.8%), alongside a notable rise in business lending usage (average up to 7.0%). Despite higher deposit account usage (average up to 144.6%), average balances and total cash inflows/outflows decreased, indicating potential financial tightening or reduced liquidity within the segment.",

    'cluster_4_descriptions' : "From T0 to T1, Cluster 4's demographic profile remained largely stable, with the most notable shift being a change in the most common occupation from freelancers to corporate employees, suggesting increased professional stability. Financial behaviors showed significant growth in product usage, with sharp increases in savings account balances (average up to 290.2%) and payment services (average up to 191.5%), alongside a rise in transaction frequency (average monthly transactions up to 119.2). However, average deposit balances decreased substantially (from $43,703 to $14,341), indicating a shift toward higher transaction activity but lower retained balances.",

    'cluster_5_descriptions' : "From T0 to T1, Cluster 5's demographic profile remained largely consistent, with a slight increase in the average number of children (from 0.0 to 0.1), suggesting a marginal shift toward family growth. Key financial behaviors showed notable changes, including a significant rise in average deposit account balances (from $266,363 to $371,150) and increased usage of savings accounts and health insurance products. However, median balances and transactions declined, indicating a growing disparity in financial activity within the segment.",

    'cluster_6_descriptions' : "From T0 to T1, Cluster 6 shows minimal demographic shifts, with a slight increase in the average number of children (from 0.2 to 0.3) while maintaining stability in age, marital status, and occupation profiles. Financially, the segment exhibits a decline in average savings account usage (from 481.5% to 436.3%) and payment services (from 284.5% to 240.6%), while deposit account balances grew (average increased from $148,099 to $165,465) alongside higher monthly transactions. These changes suggest a shift toward more active deposit behavior and slightly reduced reliance on savings and payment products.",

    'cluster_7_descriptions' : """From T0 to T1, Cluster 7's demographic profile remained largely stable, with no significant changes in age, marital status, education, or occupation. However, there was a slight increase in the median age, indicating a gradual aging of the segment. 
                                Financially, this segment exhibited notable growth in product usage, with significant increases in average savings account balances (+93.5%), health insurance adoption (+50.9%), and payment services usage (+41.9%). Deposit account activity also intensified, with a 50.3% rise in average deposit account usage and a 44% increase in average monthly transactions, 
                                though balances slightly declined. Cash flow patterns showed higher total inflows (+13.9%) but also increased outflows (+39.7%), suggesting greater financial activity and spending behavior.""",

    'cluster_8_descriptions' : "From T0 to T1, the demographic profile of Cluster 8 remained stable, with no notable changes in age, marital status, occupation, or family size. However, their financial behaviors and product usage patterns showed significant growth, with a marked increase in savings account usage (average up from 217.3% to 300.0%), payment services (143.1% to 184.6%), and general services (46.1% to 58.4%). Additionally, average deposit account balances rose substantially (from $12,937 to $20,450), and total cash inflows and outflows increased, reflecting greater financial activity and engagement with banking products."
}


def format_customer_data_for_prompt(customer_row):
    """
    Format individual customer data from dataframe row for demographic prediction prompt
    
    Args:
        customer_row: Single row from dataframe containing customer data
        
    Returns:
        dict: Formatted customer data ready for prompt template
    """
    # Safely handle NaN values for numeric fields
    def safe_int(value, default=0):
        try:
            return int(float(value)) if not pd.isna(value) else default
        except (ValueError, TypeError):
            return default
    
    def safe_float(value, default=0.0):
        try:
            return float(value) if not pd.isna(value) else default
        except (ValueError, TypeError):
            return default
    
    # Extract demographics (T0 state)
    demographics = {
        'EDUCATION_T0': str(customer_row.get('Education level', 'Unknown')),
        'MARITAL_STATUS_T0': str(customer_row.get('Marital status', 'Unknown')),
        'OCCUPATION_T0': str(customer_row.get('Occupation Group', 'Unknown')),
        'NUM_CHILDREN_T0': safe_int(customer_row.get('Number of Children', 0)),
        'REGION_T0': str(customer_row.get('Region', 'Unknown')),
        'AGE_T0': safe_int(customer_row.get('Age', 'Unknown')),
        'GENDER_T0': str(customer_row.get('Gender', 'Unknown')),
        'NUM_VEHICLES_T0': safe_int(customer_row.get('Number of Vehicles', 0)),
        'CUSTOMER_ID': str(customer_row.get('CUST_ID', 'Unknown'))
    }
    
    # Extract financial profile
    financial_profile = {
        'DEPOSIT_BALANCE': f"${safe_float(customer_row.get('Deposit Account Balance', 0)):,.2f}",
        'AVG_TRANSACTION': f"${safe_float(customer_row.get('Deposit Account Transactions AVG', 0)):,.2f}",
        'TRANSACTION_FREQ': f"{safe_float(customer_row.get('Deposit Account Transactions', 0)):.1f} per month",
        'MIN_TRANSACTION': f"${safe_float(customer_row.get('Deposit Account Transactions MIN', 0)):,.2f}",
        'MAX_TRANSACTION': f"${safe_float(customer_row.get('Deposit Account Transactions MAX', 0)):,.2f}",
        'SAVINGS_USAGE': f"{safe_float(customer_row.get('Savings Account', 0)):.1%}",
        'SAVINGS_SUBGROUP_USAGE': f"{safe_float(customer_row.get('Savings Account Subgroup', 0)):.1%}",
        'PAYMENT_SERVICES': f"{safe_float(customer_row.get('Payment', 0)):.1%}",
        'LENDING_ACTIVITY': f"{safe_float(customer_row.get('Lending', 0)):.1%}",
        'INSURANCE_USAGE': f"{safe_float(customer_row.get('Health Insurance', 0)):.1%}",
        'GENERAL_SERVICES': f"{safe_float(customer_row.get('Service', 0)):.1%}",
        'BUSINESS_LENDING': f"{safe_float(customer_row.get('Business Lending', 0)):.1%}",
        'DEPOSIT_ACCOUNT_USAGE': f"{safe_float(customer_row.get('Deposit Account', 0)):.1%}"
    }
    
    # Extract cash flow patterns
    cash_flow = {
        'MONTHLY_INFLOWS': f"{safe_float(customer_row.get('Deposit Account Inflow', 0)):.1f} transactions",
        'MONTHLY_OUTFLOWS': f"{safe_float(customer_row.get('Deposit Account Outflow', 0)):.1f} transactions",
        'MIN_INFLOW': f"${safe_float(customer_row.get('Deposit Account Inflow MIN', 0)):,.2f}",
        'MAX_INFLOW': f"${safe_float(customer_row.get('Deposit Account Inflow MAX', 0)):,.2f}",
        'MIN_OUTFLOW': f"${safe_float(customer_row.get('Deposit Account Outflow MIN', 0)):,.2f}",
        'MAX_OUTFLOW': f"${safe_float(customer_row.get('Deposit Account Outflow MAX', 0)):,.2f}",
        'TOTAL_INFLOW_AMOUNT': f"${safe_float(customer_row.get('Deposit Account Inflow Amount', 0)):,.2f}",
        'TOTAL_OUTFLOW_AMOUNT': f"${safe_float(customer_row.get('Deposit Account Outflow Amount', 0)):,.2f}"
    }
    
    # Combine all data
    customer_data = {
        **demographics,
        **financial_profile,
        **cash_flow
    }
    
    return customer_data


def format_customer_prompt_section(customer_row):
    """
    Format customer data into readable prompt section
    
    Args:
        customer_row: Single row from dataframe containing customer data
        
    Returns:
        str: Formatted customer state section for prompt
    """
    data = format_customer_data_for_prompt(customer_row)
    
    formatted_section = f"""## Current Customer State (Time T0)

**Customer ID:** {data['CUSTOMER_ID']}

**Demographics:**
- Age: {data['AGE_T0']} years
- Gender: {data['GENDER_T0']}
- Educational Level: {data['EDUCATION_T0']}
- Marital Status: {data['MARITAL_STATUS_T0']}
- Occupation: {data['OCCUPATION_T0']}
- Number of Children: {data['NUM_CHILDREN_T0']}
- Number of Vehicles: {data['NUM_VEHICLES_T0']}
- Region: {data['REGION_T0']}

**Financial Account Status:**
- Deposit Account Usage: {data['DEPOSIT_ACCOUNT_USAGE']}
- Current Balance: {data['DEPOSIT_BALANCE']}
- Monthly Transaction Frequency: {data['TRANSACTION_FREQ']}
- Average Transaction Amount: {data['AVG_TRANSACTION']}
- Minimum Transaction: {data['MIN_TRANSACTION']}
- Maximum Transaction: {data['MAX_TRANSACTION']}

**Product Holdings:**
- Savings Account: {data['SAVINGS_USAGE']}
- Savings Account Subgroup: {data['SAVINGS_SUBGROUP_USAGE']}
- Payment Services: {data['PAYMENT_SERVICES']}
- Lending Products: {data['LENDING_ACTIVITY']}
- Health Insurance: {data['INSURANCE_USAGE']}
- General Services: {data['GENERAL_SERVICES']}
- Business Lending: {data['BUSINESS_LENDING']}

**Cash Flow Patterns:**
- Monthly Inflow Transactions: {data['MONTHLY_INFLOWS']}
- Monthly Outflow Transactions: {data['MONTHLY_OUTFLOWS']}
- Total Monthly Inflow Amount: {data['TOTAL_INFLOW_AMOUNT']}
- Total Monthly Outflow Amount: {data['TOTAL_OUTFLOW_AMOUNT']}
- Minimum Inflow: {data['MIN_INFLOW']}
- Maximum Inflow: {data['MAX_INFLOW']}
- Minimum Outflow: {data['MIN_OUTFLOW']}
- Maximum Outflow: {data['MAX_OUTFLOW']}
"""
    
    return formatted_section


def create_full_customer_prompt(customer_row, cluster_id, cluster_descriptions, t_diff, change_analysis_descriptions=None):
    """
    Create complete prompt for a specific customer
    
    Args:
        customer_row: Single row from dataframe containing customer data
        cluster_id: Customer's cluster ID (0-8)
        cluster_descriptions: Dict mapping cluster_id to description text
        
    Returns:
        str: Complete prompt ready for LLM
    """
    
    # Get cluster description
    cluster_desc = cluster_descriptions.get(f"cluster_{cluster_id}_descriptions", 
                                          f"Cluster {cluster_id} description not found")
    
    change_desc = None
    if change_analysis_descriptions is not None:
        change_desc = change_analysis_descriptions.get(f"cluster_{cluster_id}_descriptions", 
                                          f"Cluster {cluster_id} change analysis not found")
    
    # Format customer data section
    customer_section = format_customer_prompt_section(customer_row)

    allowed_values = """
    **Allowed Prediction Values:**
    - Education: 'less than high school', 'others'  'vocational certificate/diploma', 'high school', "bachelor's degree", 'vocational certificate', "master's degree", "doctorate's degree"
    - Occupation Group: 'Freelancer', 'Corporate Employee', 'Student', 'Entrepreneur', 'Other', 'Unemployed', 'Professional', 'Homemaker', 'Agriculture/Trade'
    - Marital Status: 'single', 'others', 'widow', 'married', 'separate', 'divorce'
    - Region: 'Northeastern', 'Southern', 'Northern', 'Central', 'Eastern', 'Western'
    """
    
    # Build cluster profile section with optional change analysis
    cluster_profile_section = f"""## Cluster Profile
**You represent the following customer segment:**
{cluster_desc}"""
    
    # Add change analysis section if provided
    if change_desc is not None:
        cluster_profile_section += f"""

## Historical Cluster Evolution (T0 to T1)
**Understanding how your cluster has evolved helps inform individual predictions:**
{change_desc}

**Key Implications for Predictions:**
- Use these historical patterns as context for individual customer predictions
- Consider whether the customer's current profile aligns with typical cluster evolution trends
- Factor in cluster-wide behavioral shifts when predicting individual changes
- Balance individual customer signals with cluster-level transformation patterns"""
    
    # Build complete prompt
    full_prompt = f"""# Demographic Prediction Agent Prompt

## Your Role
You are a sophisticated demographic prediction agent representing **Cluster {cluster_id}** customers. Your task is to predict how a customer's demographic profile will evolve from time T0 to time T1 based on their financial behavior patterns and cluster characteristics.
Given that the difference between T0 and T1 is {t_diff} year(s).

{cluster_profile_section}

{customer_section}

## Prediction Task
Based on your cluster's behavioral patterns and the customer's current financial profile, predict their demographic status at time T1 using only the allowed values indicated in {allowed_values}. 
Consider the following factors:

### Life Stage Progression Factors:
1. **Age-Related Changes**: How might natural aging affect their status?
2. **Financial Maturity**: What do their financial patterns suggest about life changes?
3. **Career Progression**: Does their financial behavior indicate career advancement?
4. **Family Formation**: Do spending/saving patterns suggest family planning?
5. **Geographic Mobility**: Do transaction patterns indicate potential relocation?

### Cluster-Specific Behavioral Indicators:
- **Savings Patterns**: High/low balances may indicate life stage transitions
- **Transaction Behavior**: Frequency and amounts can signal lifestyle changes
- **Product Adoption**: New services may indicate changing needs
- **Cash Flow Changes**: Inflow/outflow patterns suggest economic status shifts

## Prediction Guidelines

### Educational Level Prediction:
- Age Constraints: Apply realistic age-based educational progression limits:
    - Ages 18-25: High likelihood of educational advancement (high school → bachelor's, bachelor's → master's)
    - Ages 26-35: Moderate likelihood, especially for career advancement (bachelor's → master's, professional certifications)
    - Ages 36-45: Lower likelihood, mainly for career-driven education (MBA, professional development)
    - Ages 46+: Very low likelihood of formal degree advancement; focus on professional certifications if any
    - Key Rule: If customer is 40+ with high school or less education, educational level will likely remain unchanged unless exceptional financial indicators suggest otherwise
- Consider if financial stability/growth suggests continued education
- Evaluate if career-related financial patterns indicate skill development
- Factor in cluster's typical educational progression patterns
- Stability Principle: Educational level tends to remain stable after age 35 unless there are strong financial indicators of career-driven education

### Marital Status Prediction:
- Rule-Based Constraints: Apply strict marital status transition rules:
    - If T0 = "divorce": T1 must remain "divorce" (divorced individuals don't revert to "single" status)
    - If T0 = "others": T1 must remain "others" (maintain special status categories)
- Analyze spending patterns for relationship-related changes
- Consider age and cluster demographics for marriage/divorce likelihood
- Look for financial behavior suggesting household formation/dissolution

### Occupation Prediction:
- **Age-Based Transition Constraints**: Apply realistic age-related occupation progression rules:
    - **Student Transitions (Ages 18-30)**:
        - If T0 = "student" and age 18-22: High likelihood of remaining student or transitioning to entry-level occupations upon graduation
        - If T0 = "student" and age 23-26: Moderate to high likelihood of transitioning to professional occupations, but can remain student for advanced degrees
        - If T0 = "student" and age 27-30: Lower likelihood of remaining student unless pursuing advanced degrees (PhD, professional programs); higher likelihood of career transition
        - Key Rule: Students can continue as students if pursuing higher education, but consider financial patterns for graduation timing
    
    - **Career Progression (All Ages)**:
        - Ages 22-35: High likelihood of career advancement or job changes
        - Ages 36-50: Moderate likelihood of career shifts, focus on advancement within field
        - Ages 51+: Lower likelihood of major career changes, focus on stability or retirement preparation
    
    - **Career Stability Factors**:
        - Consider financial growth patterns for career advancement
        - Evaluate transaction patterns for job changes or entrepreneurial shifts
        - Factor in cluster's typical career trajectories
        - Assess if financial behavior suggests career stability vs. transition

### Number of Children Prediction:
- **Rule-Based Constraints**: Apply strict marital status-based rules:
    - If marital_status_t1 = "single": num_children_t1 must be 0 (single individuals are assumed to have no children in this model)
- Analyze spending patterns for family-related expenses
- Consider age, marital status, and financial capacity
- Factor in cluster's typical family formation patterns
- Ensure logical consistency between marital status and number of children predictions

### Region Prediction:
- Evaluate transaction patterns for geographic indicators
- Consider career and life stage factors affecting mobility
- Assess financial capacity for relocation

## Output Format
Provide your predictions in the following JSON format:

```json
{{
  "cluster_id": {cluster_id},
  "predictions": {{
    "educational_level_t1": "predicted_education",
    "marital_status_t1": "predicted_marital_status",
    "occupation_t1": "predicted_occupation",
    "num_children_t1": predicted_number,
    "region_t1": "predicted_region"
  }},
  "confidence_scores": {{
    "educational_level": 0.0-1.0,
    "marital_status": 0.0-1.0,
    "occupation": 0.0-1.0,
    "num_children": 0.0-1.0,
    "region": 0.0-1.0
  }},
  "reasoning": {{
    "educational_level": "brief explanation",
    "marital_status": "brief explanation",
    "occupation": "brief explanation",
    "num_children": "brief explanation",
    "region": "brief explanation"
  }}
}}
```

## Key Considerations
- **Temporal Consistency**: Ensure predictions align with natural life progression
- **Financial Logic**: Base predictions on observable financial behavior patterns
- **Cluster Alignment**: Stay true to your cluster's characteristic behaviors
- **Realistic Probability**: Consider what changes are most likely given the time frame
- **Interconnected Changes**: Account for how one demographic change affects others
- **Age-Appropriate Transitions**: Ensure occupation changes are realistic for the customer's age and current status

Remember: You are an expert in understanding how financial behavior reflects and predicts life changes. Use your cluster's behavioral patterns as a lens to interpret the customer's likely demographic evolution.
"""
    
    return full_prompt



def agent(full_prompt: str) -> Optional[Dict[str, Any]]:
    client = get_aoi_client()

    prompt = full_prompt
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": ""}
        ],
        temperature=0.7,
        max_tokens=1000
    )
    
    return {
        "content": response.choices[0].message.content,
        "usage": dict(response.usage),
        # "id": response.id
    }

def process_all_customers(df, cluster_descriptions, batch_size=None):
    """
    Process all customers in the dataframe to predict their T1 demographics
    
    Args:
        df: DataFrame containing all customer data
        cluster_descriptions: Dict mapping cluster_id to description text
        batch_size: Optional number of customers to process (None for all)
        
    Returns:
        pd.DataFrame: Combined results with original data + predictions
    """
    results = []
    total_customers = batch_size if batch_size else len(df)
    
    for i in range(total_customers):
        try:
            customer_row = df.iloc[i]
            cluster_id = int(customer_row['cluster'])
            customer_id = customer_row['CUST_ID']
            
            print(f"\nProcessing customer {i+1}/{total_customers}: ID {customer_id} (Cluster {cluster_id})")
            
            # Generate prompt and get prediction
            # prompt = create_full_customer_prompt(customer_row, cluster_id, cluster_descriptions, t_diff)
            prompt = create_full_customer_prompt(customer_row, cluster_id, cluster_descriptions, t_diff, change_analysis_descriptions)
            print(prompt)
            response = agent(prompt)
            raw_content = response['content']
            
            json_str = re.sub(r'^```json|```$', '', raw_content.strip(), flags=re.MULTILINE).strip()
            parsed_json = json.loads(json_str)
            
            # Save individual JSON result
            save_individual_result(parsed_json, customer_id)
            
            flattened = flatten_prediction_results(customer_row, parsed_json)
            results.append(flattened)
            
        except Exception as e:
            print(f"❌ Error processing customer {i+1}: {str(e)}")
            continue
    
    # Combine all results into a dataframe
    results_df = pd.DataFrame(results)
    return results_df



# Global variable to store the determined version directory
VERSIONED_DIR = None

def save_individual_result(parsed_json: Dict[str, Any], customer_id: str) -> None:
    """Save individual customer prediction to a versioned directory (created once)"""
    global VERSIONED_DIR
    
    base_dir = "src/demog_T1_pred"
    os.makedirs(base_dir, exist_ok=True)
    
    # Create versioned directory only once
    if VERSIONED_DIR is None:
        # Find all existing version directories
        existing_versions = []
        for item in os.listdir(base_dir):
            if os.path.isdir(os.path.join(base_dir, item)):
                match = re.fullmatch(r'individual_results_v(\d+)', item)
                if match:
                    existing_versions.append(int(match.group(1)))
        
        # Determine next version number
        next_version = max(existing_versions) + 1 if existing_versions else 1
        VERSIONED_DIR = os.path.join(base_dir, f"individual_results_v{next_version}")
        os.makedirs(VERSIONED_DIR, exist_ok=True)
        print(f"📁 Created new version directory: {VERSIONED_DIR}")
    
    # Save the individual file
    file_path = os.path.join(VERSIONED_DIR, f"customer_{customer_id}_prediction.json")
    with open(file_path, 'w') as f:
        json.dump(parsed_json, f, indent=4)
    
    print(f"✅ Saved prediction for customer {customer_id}")

def flatten_prediction_results(customer_row, prediction_json):
    """
    Combine original customer data with prediction results
    
    Args:
        customer_row: Original customer data from dataframe
        prediction_json: Prediction results from LLM
        
    Returns:
        dict: Combined data with original + predicted fields
    """
    # Convert customer row to dict
    customer_data = customer_row.to_dict()
    
    # Extract predictions
    predictions = prediction_json.get('predictions', {})
    confidence = prediction_json.get('confidence_scores', {})
    reasoning = prediction_json.get('reasoning', {})
    
    # Create new keys for predicted values
    predicted_data = {
        'PRED_education': predictions.get('educational_level_t1'),
        'PRED_marital_status': predictions.get('marital_status_t1'),
        'PRED_occupation': predictions.get('occupation_t1'),
        'PRED_num_children': predictions.get('num_children_t1'),
        'PRED_region': predictions.get('region_t1'),
        
        'CONFIDENCE_education': confidence.get('educational_level'),
        'CONFIDENCE_marital_status': confidence.get('marital_status'),
        'CONFIDENCE_occupation': confidence.get('occupation'),
        'CONFIDENCE_num_children': confidence.get('num_children'),
        'CONFIDENCE_region': confidence.get('region'),
        
        'REASONING_education': reasoning.get('educational_level'),
        'REASONING_marital_status': reasoning.get('marital_status'),
        'REASONING_occupation': reasoning.get('occupation'),
        'REASONING_num_children': reasoning.get('num_children'),
        'REASONING_region': reasoning.get('region')
    }
    
    # Combine with original data
    return {**customer_data, **predicted_data}


if __name__ == "__main__":
    df = pd.read_csv('src/clustering/approach_2/pred_result/full_data_with_cluster/full_data_with_cluster_v1.csv')
    
    results_df = process_all_customers(df, cluster_descriptions, batch_size=None)
    
    output_dir = "src/demog_T1_pred"
    os.makedirs(output_dir, exist_ok=True)
    
    existing_files = [f for f in os.listdir(output_dir) if f.startswith('all_predictions_v') and f.endswith('.csv')]
    versions = [int(f.split('_v')[1].split('.csv')[0]) for f in existing_files] if existing_files else [0]
    next_version = max(versions) + 1
    
    output_path = os.path.join(output_dir, f'all_predictions_v{next_version}.csv')
    results_df.to_csv(output_path, index=False)
    print(f"\n✅ All predictions saved to: {output_path}")
    
    # Also save as JSON
    json_path = os.path.join(output_dir, f'all_predictions_v{next_version}.json')
    results_df.to_json(json_path, orient='records', indent=4)
    print(f"✅ JSON version saved to: {json_path}")