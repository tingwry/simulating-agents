import pandas as pd

# Configuration
T_DIFF = '1'

# MAIN

# Single-Stage

def indiv_create_prediction_prompt(customer_row, with_constraints=False): # not tried yet
    """Create the complete prediction prompt."""
    customer_section = format_customer_prompt_section(customer_row)
    allowed_values = get_allowed_values()

    constraints = ""
    if with_constraints:
        constraints = get_constraints()
    
    base_prompt = indiv_get_base_prompt()
    
    return base_prompt.format(
        time_diff=T_DIFF,
        customer_section=customer_section,
        allowed_values=allowed_values,
        constraints=constraints,
        customer_id=customer_row['CUST_ID']
    )

def cluster_create_prediction_prompt(customer_row, cluster_id, change_analysis, with_constraints=False):
    """Create the complete prediction prompt."""
    cluster_profile_section = format_cluster_profile_section(cluster_id, change_analysis)
    customer_section = format_customer_prompt_section(customer_row)
    allowed_values = get_allowed_values()
    
    constraints = ""
    if with_constraints:
        constraints = get_constraints()

    base_prompt = cluster_get_base_prompt()
    
    return base_prompt.format(
        cluster_id = cluster_id,
        time_diff=T_DIFF,
        cluster_profile_section=cluster_profile_section,
        customer_section=customer_section,
        allowed_values=allowed_values,
        constraints=constraints,
        customer_id=customer_row['CUST_ID']
    )

def rag_create_prediction_prompt(customer_row, similar_customers_data, with_constraints=False):
    """Create the complete prediction prompt."""
    customer_section = format_customer_prompt_section(customer_row)
    similar_customers_section = format_similar_customers_section(similar_customers_data)
    allowed_values = get_allowed_values()
    
    constraints = ""
    if with_constraints:
        constraints = get_constraints()
        
    base_prompt = rag_get_base_prompt()
    
    return base_prompt.format(
        time_diff=T_DIFF,
        similar_customers_section=similar_customers_section,
        customer_section=customer_section,
        allowed_values=allowed_values,
        constraints=constraints,
        customer_id=customer_row['CUST_ID']
    )

# Multi-Stage

def indiv_create_prediction_prompt_action(customer_row):
    """Create the complete prediction prompt."""
    customer_section = format_customer_prompt_section(customer_row)
    base_prompt = indiv_get_base_prompt_action()
    
    return base_prompt.format(
        time_diff=T_DIFF,
        customer_section=customer_section,
        customer_id=customer_row['CUST_ID']
    )

def cluster_create_prediction_prompt_action(customer_row, cluster_id, change_analysis, with_constraints=False):
    """Create the complete prediction prompt."""
    cluster_profile_section = format_cluster_profile_section(cluster_id, change_analysis)
    customer_section = format_customer_prompt_section(customer_row)
    base_prompt = cluster_get_base_prompt_action()

    constraints = ""
    if with_constraints:
        constraints = get_constraints_action()
    
    return base_prompt.format(
        cluster_id = cluster_id,
        time_diff=T_DIFF,
        cluster_profile_section=cluster_profile_section,
        customer_section=customer_section,
        constraints=constraints,
        customer_id=customer_row['CUST_ID']
    )

def rag_create_prediction_prompt_action(customer_row, similar_customers_data):
    """Create the complete prediction prompt."""
    customer_section = format_customer_prompt_section(customer_row)
    similar_customers_section = format_similar_customers_section(similar_customers_data)
    base_prompt = rag_get_base_prompt_action()
    
    return base_prompt.format(
        time_diff=T_DIFF,
        similar_customers_section=similar_customers_section,
        customer_section=customer_section,
        customer_id=customer_row['CUST_ID']
    )

def create_prediction_prompt_status(customer_row, with_constraints=False):
    """Create the complete prediction prompt."""
    customer_section = format_customer_prompt_section(customer_row)
    allowed_values = get_allowed_values()
    base_prompt = get_base_prompt_status()

    constraints = ""
    if with_constraints:
        constraints = get_constraints_status()
    
    return base_prompt.format(
        customer_section=customer_section,
        allowed_values=allowed_values,
        constraints=constraints,
        customer_id=customer_row['CUST_ID']
    )

# BASE

# Single-Stage

def indiv_get_base_prompt():
    """Return the base structure of the prompt"""
    return """# Customer Demographic Prediction Task

            ## Time Frame
            Predicting changes over {time_diff} year(s) from current state (T0) to future state (T1)

            {customer_section}


            ## Prediction Task
            Based on the customer's current financial profile, predict their demographic status at time T1.
            Consider the following factors:

            ### Life Stage Progression Factors:
            1. **Age-Related Changes**: How might natural aging affect their status?
            2. **Financial Maturity**: What do their financial patterns suggest about life changes?
            3. **Career Progression**: Does their financial behavior indicate career advancement?
            4. **Family Formation**: Do spending/saving patterns suggest family planning?
            5. **Geographic Mobility**: Do transaction patterns indicate potential relocation?


            ## Prediction Guidelines
            1. Review changes in similar customers above
            2. Analyze current customer's financial indicators
            3. Predict only realistic changes possible in {time_diff} year(s)
            4. Use only the allowed values below:

            {allowed_values}


            ## Prediction Guidelines

            ### Educational Level Prediction:
            - Consider if financial stability/growth suggests continued education
            - Evaluate if career-related financial patterns indicate skill development
            - Stability Principle: Educational level tends to remain stable after age 35 unless there are strong financial indicators of career-driven education

            ### Marital Status Prediction:
            - Analyze spending patterns for relationship-related changes
            - Consider age and demographics for marriage/divorce likelihood
            - Look for financial behavior suggesting household formation/dissolution

            ### Number of Children Prediction:
            - Analyze spending patterns for family-related expenses
            - Consider age, marital status, and financial capacity
            - Ensure logical consistency between marital status and number of children predictions

            ### Region Prediction:
            - Evaluate transaction patterns for geographic indicators
            - Consider career and life stage factors affecting mobility
            - Assess financial capacity for relocation

            {constraints}

            ## Output Format
            ```json
            {{
            "customer_id": "{customer_id}",
            "predictions": {{
                "PRED_education": "<prediction>",
                "PRED_marital_status": "<prediction>",
                "PRED_occupation": "<prediction>",
                "PRED_num_children": <integer>,
                "PRED_region": "<prediction>"
            }},
            "confidence_scores": {{
                "education": 0.0-1.0,
                "marital_status": 0.0-1.0,
                "occupation": 0.0-1.0,
                "num_children": 0.0-1.0,
                "region": 0.0-1.0
            }},
            "reasoning": {{
                "education": "<explanation>",
                "marital_status": "<explanation>",
                "occupation": "<explanation>",
                "num_children": "<explanation>",
                "region": "<explanation>"
            }}
            }}
            ```

            ## Key Considerations
            - **Temporal Consistency**: Ensure predictions align with natural life progression
            - **Financial Logic**: Base predictions on observable financial behavior patterns
            - **Realistic Probability**: Consider what changes are most likely given the time frame
            - **Interconnected Changes**: Account for how one demographic change affects others
            - **Age-Appropriate Transitions**: Ensure occupation changes are realistic for the customer's age and current status

            """

def cluster_get_base_prompt():
    """Return the base structure of the prompt"""
    return """
            # Demographic Prediction Agent Prompt

        ## Your Role
        You are a sophisticated demographic prediction agent representing **Cluster {cluster_id}** customers. Your task is to predict how a customer's demographic profile will evolve from time T0 to time T1 based on their financial behavior patterns and cluster characteristics.
        Given that the difference between T0 and T1 is {time_diff} year(s).

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
        - Consider if financial stability/growth suggests continued education
        - Evaluate if career-related financial patterns indicate skill development
        - Factor in cluster's typical educational progression patterns
        - Stability Principle: Educational level tends to remain stable after age 35 unless there are strong financial indicators of career-driven education

        ### Marital Status Prediction:
        - Analyze spending patterns for relationship-related changes
        - Consider age and cluster demographics for marriage/divorce likelihood
        - Look for financial behavior suggesting household formation/dissolution

        ### Occupation Prediction:
        - **Career Stability Factors**:
            - Consider financial growth patterns for career advancement
            - Evaluate transaction patterns for job changes or entrepreneurial shifts
            - Factor in cluster's typical career trajectories
            - Assess if financial behavior suggests career stability vs. transition

        ### Number of Children Prediction:
        - Analyze spending patterns for family-related expenses
        - Consider age, marital status, and financial capacity
        - Factor in cluster's typical family formation patterns
        - Ensure logical consistency between marital status and number of children predictions

        ### Region Prediction:
        - Evaluate transaction patterns for geographic indicators
        - Consider career and life stage factors affecting mobility
        - Assess financial capacity for relocation

        {constraints}

        ## Output Format
        Provide your predictions in the following JSON format:

        ```json
        {{
        "cluster_id": {cluster_id},
        "predictions": {{
            "PRED_education": "predicted_education",
            "PRED_marital_status": "predicted_marital_status",
            "PRED_occupation": "predicted_occupation",
            "PRED_num_children": predicted_number,
            "PRED_region": "predicted_region"
        }},
        "confidence_scores": {{
            "education": 0.0-1.0,
            "marital_status": 0.0-1.0,
            "occupation": 0.0-1.0,
            "num_children": 0.0-1.0,
            "region": 0.0-1.0
        }},
        "reasoning": {{
            "education": "brief explanation",
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

def rag_get_base_prompt():
    """Return the base structure of the prompt"""
    return """# Customer Demographic Prediction Task

            ## Time Frame
            Predicting changes over {time_diff} year(s) from current state (T0) to future state (T1)

            {similar_customers_section}

            {customer_section}


            ## Prediction Task
            Based on the customer's current financial profile, predict their demographic status at time T1. Also use the similar customers' changes and their reasons as case studies to predict what the customer would do.
            Consider the following factors:

            ### Life Stage Progression Factors:
            1. **Age-Related Changes**: How might natural aging affect their status?
            2. **Financial Maturity**: What do their financial patterns suggest about life changes?
            3. **Career Progression**: Does their financial behavior indicate career advancement?
            4. **Family Formation**: Do spending/saving patterns suggest family planning?
            5. **Geographic Mobility**: Do transaction patterns indicate potential relocation?


            ## Prediction Guidelines
            1. Review changes in similar customers above
            2. Analyze current customer's financial indicators
            3. Predict only realistic changes possible in {time_diff} year(s)
            4. Use only the allowed values below:

            {allowed_values}


            ## Prediction Guidelines

            ### Educational Level Prediction:
            - Consider if financial stability/growth suggests continued education
            - Evaluate if career-related financial patterns indicate skill development
            - Stability Principle: Educational level tends to remain stable after age 35 unless there are strong financial indicators of career-driven education

            ### Marital Status Prediction:
            - Analyze spending patterns for relationship-related changes
            - Consider age and demographics for marriage/divorce likelihood
            - Look for financial behavior suggesting household formation/dissolution

            ### Number of Children Prediction:
            - Analyze spending patterns for family-related expenses
            - Consider age, marital status, and financial capacity
            - Ensure logical consistency between marital status and number of children predictions

            ### Region Prediction:
            - Evaluate transaction patterns for geographic indicators
            - Consider career and life stage factors affecting mobility
            - Assess financial capacity for relocation

            {constraints}

            ## Output Format
            ```json
            {{
            "customer_id": "{customer_id}",
            "predictions": {{
                "PRED_education": "<prediction>",
                "PRED_marital_status": "<prediction>",
                "PRED_occupation": "<prediction>",
                "PRED_num_children": <integer>,
                "PRED_region": "<prediction>"
            }},
            "confidence_scores": {{
                "education": 0.0-1.0,
                "marital_status": 0.0-1.0,
                "occupation": 0.0-1.0,
                "num_children": 0.0-1.0,
                "region": 0.0-1.0
            }},
            "reasoning": {{
                "education": "<explanation referencing similar customers>",
                "marital_status": "<explanation>",
                "occupation": "<explanation>",
                "num_children": "<explanation>",
                "region": "<explanation>"
            }}
            }}
            ```

            ## Key Considerations
            - **Temporal Consistency**: Ensure predictions align with natural life progression
            - **Financial Logic**: Base predictions on observable financial behavior patterns
            - **Realistic Probability**: Consider what changes are most likely given the time frame
            - **Interconnected Changes**: Account for how one demographic change affects others
            - **Age-Appropriate Transitions**: Ensure occupation changes are realistic for the customer's age and current status

            """

# Multi-Stage

def indiv_get_base_prompt_action():
    """Return the base structure of the prompt"""
    return """
            # Role
            You are a customer with the following demographics and financial behaviors at time T0.
            {customer_section}

            # Task
            As time has changed by {time_diff} year(s) from T0, You have to select your actions for the following aspects:
            1. Education: whether you will further pursue your studies
            2. Marital Status: whether you will marry or divorce
            3. Occupation: whether you will start a job, change job, or be unemployed
            4. Number of Children: whether you will have kids, or have more kids
            5. Region: whether you will move to a different region

            # Action Prediction Guidelines
            Consider the following factors:

            ### Life Stage Progression Factors:
            1. **Age-Related Changes**: How might natural aging affect their actions?
            2. **Financial Maturity**: What do their financial patterns suggest about life changes?
            3. **Career Progression**: Does their financial behavior indicate career advancement?
            4. **Family Formation**: Do spending/saving patterns suggest family planning?
            5. **Geographic Mobility**: Do transaction patterns indicate potential relocation?

            ### Educational Level Prediction:
            - Consider if financial stability/growth suggests continued education
            - Evaluate if career-related financial patterns indicate skill development
            - Stability Principle: Educational level tends to remain stable after age 35 unless there are strong financial indicators of career-driven education

            ### Marital Status Prediction:
            - Analyze spending patterns for relationship-related changes
            - Consider age and demographics for marriage/divorce likelihood
            - Look for financial behavior suggesting household formation/dissolution

            ### Number of Children Prediction:
            - Analyze spending patterns for family-related expenses
            - Consider age, marital status, and financial capacity
            - Ensure logical consistency between marital status and number of children predictions

            ### Region Prediction:
            - Evaluate transaction patterns for geographic indicators
            - Consider career and life stage factors affecting mobility
            - Assess financial capacity for relocation



            ## Output Format
            ```json
            {{
            "predictions": {{
                "action_for_education": "<predicted action>",
                "action_for_marital_status": "<predicted action>",
                "action_for_occupation": "<predicted action>",
                "action_for_num_children": <predicted action>,
                "action_for_region": "<predicted action>"
            }},
            "confidence_scores": {{
                "education": 0.0-1.0,
                "marital_status": 0.0-1.0,
                "occupation": 0.0-1.0,
                "num_children": 0.0-1.0,
                "region": 0.0-1.0
            }},
            "reasoning": {{
                "education": "<explanation>",
                "marital_status": "<explanation>",
                "occupation": "<explanation>",
                "num_children": "<explanation>",
                "region": "<explanation>"
            }}
            }}
            ```

            ## Key Considerations
            - **Temporal Consistency**: Ensure predictions align with natural life progression
            - **Financial Logic**: Base predictions on observable financial behavior patterns
            - **Realistic Probability**: Consider what changes are most likely given the time frame
            - **Interconnected Changes**: Account for how one demographic change affects others
            - **Age-Appropriate Transitions**: Ensure occupation changes are realistic for the customer's age and current status

            """

def cluster_get_base_prompt_action():
    """Return the base structure of the prompt"""
    return """
            # Role
            You are a customer with the following demographics and financial behaviors at time T0.
            {customer_section}

            You are also one of the customers representing **Cluster {cluster_id}** customers.


            # Cluster {cluster_id} characteristics
            {cluster_profile_section}

            # Task
            As time has changed by {time_diff} year(s) from T0, You have to select your actions for the following aspects:
            1. Education: whether you will further pursue your studies
            2. Marital Status: whether you will marry or divorce
            3. Occupation: whether you will start a job, change job, or be unemployed
            4. Number of Children: whether you will have kids, or have more kids
            5. Region: whether you will move to a different region

            # Action Prediction Guidelines
            Consider the following factors, as well as the characteristics of the cluster:

            ### Life Stage Progression Factors:
            1. **Age-Related Changes**: How might natural aging affect their actions?
            2. **Financial Maturity**: What do their financial patterns suggest about life changes?
            3. **Career Progression**: Does their financial behavior indicate career advancement?
            4. **Family Formation**: Do spending/saving patterns suggest family planning?
            5. **Geographic Mobility**: Do transaction patterns indicate potential relocation?

            ### Educational Level Prediction:
            - Consider if financial stability/growth suggests continued education
            - Evaluate if career-related financial patterns indicate skill development
            - Stability Principle: Educational level tends to remain stable after age 35 unless there are strong financial indicators of career-driven education

            ### Marital Status Prediction:
            - Analyze spending patterns for relationship-related changes
            - Consider age and demographics for marriage/divorce likelihood
            - Look for financial behavior suggesting household formation/dissolution

            ### Number of Children Prediction:
            - Analyze spending patterns for family-related expenses
            - Consider age, marital status, and financial capacity
            - Ensure logical consistency between marital status and number of children predictions

            ### Region Prediction:
            - Evaluate transaction patterns for geographic indicators
            - Consider career and life stage factors affecting mobility
            - Assess financial capacity for relocation

            {constraints}

            ## Output Format
            ```json
            {{
            "predictions": {{
                "action_for_education": "<predicted action>",
                "action_for_marital_status": "<predicted action>",
                "action_for_occupation": "<predicted action>",
                "action_for_num_children": <predicted action>,
                "action_for_region": "<predicted action>"
            }},
            "confidence_scores": {{
                "education": 0.0-1.0,
                "marital_status": 0.0-1.0,
                "occupation": 0.0-1.0,
                "num_children": 0.0-1.0,
                "region": 0.0-1.0
            }},
            "reasoning": {{
                "education": "<explanation>",
                "marital_status": "<explanation>",
                "occupation": "<explanation>",
                "num_children": "<explanation>",
                "region": "<explanation>"
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

            """

def rag_get_base_prompt_action():
    """Return the base structure of the prompt"""
    return """
            # Role
            You are a customer with the following demographics and financial behaviors at time T0.
            {customer_section}

            # Task
            As time has changed by {time_diff} year(s) from T0, You have to select your actions for the following aspects:
            1. Education: whether you will further pursue your studies
            2. Marital Status: whether you will marry or divorce
            3. Occupation: whether you will start a job, change job, or be unemployed
            4. Number of Children: whether you will have kids, or have more kids
            5. Region: whether you will move to a different region

            # Action Prediction Guidelines
            Consider the following factors, as well as the actions and reasons of the similar customers (as well as their profiles at T0):

            ### Life Stage Progression Factors:
            1. **Age-Related Changes**: How might natural aging affect their actions?
            2. **Financial Maturity**: What do their financial patterns suggest about life changes?
            3. **Career Progression**: Does their financial behavior indicate career advancement?
            4. **Family Formation**: Do spending/saving patterns suggest family planning?
            5. **Geographic Mobility**: Do transaction patterns indicate potential relocation?

            ### Educational Level Prediction:
            - Consider if financial stability/growth suggests continued education
            - Evaluate if career-related financial patterns indicate skill development
            - Stability Principle: Educational level tends to remain stable after age 35 unless there are strong financial indicators of career-driven education

            ### Marital Status Prediction:
            - Analyze spending patterns for relationship-related changes
            - Consider age and demographics for marriage/divorce likelihood
            - Look for financial behavior suggesting household formation/dissolution

            ### Number of Children Prediction:
            - Analyze spending patterns for family-related expenses
            - Consider age, marital status, and financial capacity
            - Ensure logical consistency between marital status and number of children predictions

            ### Region Prediction:
            - Evaluate transaction patterns for geographic indicators
            - Consider career and life stage factors affecting mobility
            - Assess financial capacity for relocation


            # Case Studies
            Here is what the customers with similar profiles would do. Use these as case studies.
            {similar_customers_section}


            ## Output Format
            ```json
            {{
            "predictions": {{
                "action_for_education": "<predicted action>",
                "action_for_marital_status": "<predicted action>",
                "action_for_occupation": "<predicted action>",
                "action_for_num_children": <predicted action>,
                "action_for_region": "<predicted action>"
            }},
            "confidence_scores": {{
                "education": 0.0-1.0,
                "marital_status": 0.0-1.0,
                "occupation": 0.0-1.0,
                "num_children": 0.0-1.0,
                "region": 0.0-1.0
            }},
            "reasoning": {{
                "education": "<explanation>",
                "marital_status": "<explanation>",
                "occupation": "<explanation>",
                "num_children": "<explanation>",
                "region": "<explanation>"
            }}
            }}
            ```

            ## Key Considerations
            - **Temporal Consistency**: Ensure predictions align with natural life progression
            - **Financial Logic**: Base predictions on observable financial behavior patterns
            - **Realistic Probability**: Consider what changes are most likely given the time frame
            - **Interconnected Changes**: Account for how one demographic change affects others
            - **Age-Appropriate Transitions**: Ensure occupation changes are realistic for the customer's age and current status

            """


def get_base_prompt_status():
    """Return the base structure of the prompt"""
    return """
            # Role
            You are a customer demographics status predictor. Your task is to use demographics at time T0 and the predicted actions (provided by the user) to predict the demographics status of the customer at time T1.

            ## Demographics and Financial Behaviors at Time T0
            {customer_section}

            
            # Constraints
            The status of each demographics must be in the allowed values:
            {allowed_values}

            Note: Pass over the confidence scores and reasoning provided by the user to put in the response.

            {constraints}

            ## Output Format
            ```json
            {{
            "customer_id": "{customer_id}",
            "predictions": {{
                "PRED_education": "<prediction>",
                "PRED_marital_status": "<prediction>",
                "PRED_occupation": "<prediction>",
                "PRED_num_children": <integer>,
                "PRED_region": "<prediction>"
            }},
            "confidence_scores": {{
                "education": 0.0-1.0,
                "marital_status": 0.0-1.0,
                "occupation": 0.0-1.0,
                "num_children": 0.0-1.0,
                "region": 0.0-1.0
            }},
            "reasoning": {{
                "education": "<explanation>",
                "marital_status": "<explanation>",
                "occupation": "<explanation>",
                "num_children": "<explanation>",
                "region": "<explanation>"
            }}
            }}
            ```
            """

# CLUSTER DESC

def get_cluster_desc():
    cluster_descriptions = {
        "cluster_0_descriptions" : "Cluster 0 represents mid-career, single female corporate employees, predominantly in their late 30s, with a high school education and minimal family or vehicle ownership. This segment exhibits low engagement with financial products, limited savings account usage, and negligible cash flow activity, suggesting a preference for basic financial services and a conservative approach to money management. Unlike other segments, their financial behavior is characterized by minimal transactions and inflows, making them distinct as a low-activity, low-adoption customer group.",

        "cluster_1_descriptions" : "Cluster 1 represents mid-career, highly educated professionals, predominantly single women in their early-to-mid 40s, residing in central regions. This segment demonstrates strong financial stability with high savings account balances and significant adoption of health insurance and payment services, though they maintain low transaction amounts and limited vehicle or family-related expenses. Distinct from other segments, Cluster 1 combines a focus on savings and financial security with modest cash flow activity, reflecting a lifestyle centered on individual priorities and professional growth.",

        "cluster_2_descriptions" : "Cluster 2 represents young, single, predominantly female corporate employees in their early 30s, primarily located in the Central region. They exhibit high engagement with savings accounts and payment services but maintain low balances, modest inflows, and minimal lending or vehicle ownership, reflecting a transactional, cash-flow-focused financial behavior. Distinct from other segments, their high transaction volumes and limited product diversification align with their life stage and education level, making them an ideal target for streamlined, digital-first financial solutions that prioritize convenience and affordability.",

        'cluster_3_descriptions' : "Cluster 3 represents a predominantly young, single, female demographic, with an average age of 35.6 years and a concentration of corporate employees in central regions. This segment exhibits minimal engagement with financial products, low savings adoption (13.8% average), and negligible cash flow activity, with average inflows of $19 and outflows of $394. Distinct from other clusters, Cluster 3 is characterized by limited financial activity and product usage, likely reflecting a lifestyle focused on immediate needs rather than long-term financial planning.",

        'cluster_4_descriptions' : "Cluster 4 represents a mature, predominantly female segment of entrepreneurial professionals, averaging 47.8 years old and typically holding a bachelor's degree. They exhibit high balances and inflows, with a strong preference for savings accounts, health insurance, and payment services, while engaging minimally in business lending. Distinct from younger clusters, this group’s financial behavior is characterized by stability, moderate transaction volumes, and a focus on wealth preservation rather than high-frequency spending or borrowing.",

        'cluster_5_descriptions' : "Cluster 5 represents middle-aged, predominantly female entrepreneurs with high school education, primarily located in the Central region. This segment is characterized by high savings account balances and strong usage of payment services, while exhibiting conservative lending and business credit behaviors. Distinct from other segments, they maintain significant cash inflows and outflows with high average balances, yet their transactional activity remains modest, reflecting a financially stable but cautious approach to financial product adoption.",

        'cluster_6_descriptions' : "Cluster 6 represents single, male corporate employees in their late 30s, predominantly with a high school education and residing in central regions. This segment demonstrates moderate savings account usage and payment services adoption, with low engagement in lending and general services. What sets them apart is their relatively high average deposit balances and inflow amounts, despite minimal transaction activity and limited financial product diversification, indicating a preference for simplicity and stability in financial management.",

        'cluster_7_descriptions' : "Cluster 7 represents young, single, male corporate employees, predominantly located in central regions, with limited vehicle ownership and no children. This segment exhibits high usage of savings accounts and payment services, paired with low balances and inflows but frequent, small-value transactions. Distinct from older and more educated clusters, Cluster 7 prioritizes transactional convenience over diverse financial product adoption, making them ideal for streamlined, digital-first solutions.",

        'cluster_8_descriptions' : "Cluster 8 represents financially active, single male professionals in their mid-30s, primarily holding vocational certificates and working as corporate employees in central regions. This segment demonstrates strong adoption of savings accounts and payment services, with moderate health insurance usage, but minimal engagement with lending or business-related financial products. Distinct from other segments, they exhibit steady cash flow patterns with relatively high transaction volumes but low transaction amounts, reflecting a preference for frequent, small-scale financial activity.",

        'cluster_9_descriptions' : "Cluster 9 represents middle-aged, predominantly single, female freelancers with a high school education, residing in central regions. This segment demonstrates moderate engagement with savings accounts and payment services but limited adoption of lending and general financial products, with low transaction volumes and minimal cash flow activity despite occasional high average balances. Distinct from other clusters, Cluster 9’s financial behavior reflects a conservative approach to spending and borrowing, likely influenced by their freelance occupation and modest household structure.",

        'cluster_10_descriptions' : "Cluster 10 represents mid-career, single corporate employees, predominantly female, with a high school education and residing in central regions. This segment exhibits moderate financial engagement, favoring savings accounts and payment services while showing limited activity in lending and business-related products. Distinct from other clusters, their cash flow patterns reveal modest inflows and outflows with low transaction amounts, suggesting a conservative financial approach and minimal discretionary spending.",
    }

    return cluster_descriptions

def get_change_analysis_desc():
    change_analysis_descriptions = {
        "cluster_0_descriptions" : "From T0 to T1, the demographic profile of Cluster 0 remained stable, with consistent attributes such as age, marital status, education, and occupation. However, their financial behaviors and product usage patterns shifted significantly, marked by a dramatic activation of deposit accounts, increased monthly cash flow activity, and substantial growth in the adoption of savings accounts, health insurance, and payment services. This evolution suggests a transition toward more engaged financial management and broader utilization of financial products.",

        "cluster_1_descriptions" : "The demographic profile of Cluster 1 remained stable between T0 and T1, with consistent age, gender, education, marital status, occupation, and regional characteristics. However, financial behaviors and product usage patterns showed notable shifts: average deposit account balances increased significantly (from $312,895 to $485,800), alongside higher monthly transactions and slightly larger maximum transaction amounts. Product usage trends revealed declines in savings account and payment services engagement, while health insurance and lending saw modest increases, indicating evolving priorities in financial product adoption.",

        "cluster_2_descriptions" : """
            1. **Demographic Profile**: The demographic profile of Cluster 2 remained consistent from T0 to T1, with no notable changes in age, gender, marital status, occupation, or regional distribution.  

            2. **Financial Behaviors and Product Usage**: From T0 to T1, Cluster 2 exhibited a shift toward higher engagement with financial products, including a notable increase in health insurance usage (134.4% to 164.5%) and lending (11.4% to 20.4%). Deposit account balances rose modestly (average: $12,507 to $13,222), with a significant increase in median balances ($350 to $1,003) and a slight decline in average monthly transactions (72.2 to 60.8). Cash flow patterns showed reduced average inflows and outflows, but median inflows and outflows increased, indicating more consistent financial activity across the segment.
        """,

        'cluster_3_descriptions' : "From T0 to T1, **Cluster 3's demographic profile remained largely stable**, with the exception of a slight increase in the average number of vehicles owned (from 0.0 to 0.2). However, their financial behaviors and product usage patterns shifted significantly, marked by a dramatic activation of deposit accounts (average balance rising to $19,389) and increased engagement with savings accounts (236.6%), health insurance (101.2%), and payment services (139.0%). Additionally, cash flow activity surged, with average monthly inflows and outflows becoming highly active, reflecting a more financially engaged and transactional customer segment at T1.",

        'cluster_4_descriptions' : """
            1. **Demographic Profile**: The demographic profile of this segment remained stable from T0 to T1, with no notable changes in age, gender, education, marital status, occupation, or regional distribution.  

            2. **Financial Behaviors & Product Usage**: This segment exhibited a decline in financial activity and engagement. Average savings account usage decreased slightly (from 455.2% to 446.8%), while payment services usage dropped more significantly (from 264.5% to 223.6%). Deposit account balances grew (from $362,107 to $405,033), but monthly transactions and cash flow activity declined, with average monthly inflows dropping from 32.9 to 22.4 and total inflow amounts decreasing by 8.7%. Business lending saw a modest increase, but overall financial engagement softened.
        """,

        'cluster_5_descriptions' : "The demographic profile of Cluster 5 remains largely consistent, with no changes in age, education, occupation, or region, though marital status shifted from predominantly married to single. Financial behaviors indicate reduced average savings account balances and subgroup usage, alongside increased adoption of health insurance (+22.5%) and business lending (+5.4%). Deposit account activity intensified, with higher average monthly transactions (+6.3%) and deposit account activation (+19.3%), though average balances decreased significantly (-$28,436). Cash flow patterns show reduced total inflow and outflow amounts, suggesting a contraction in overall financial activity.",

        'cluster_6_descriptions' : "From T0 to T1, the demographic profile of Cluster 6 remains stable, with no significant changes in age, gender, education, marital status, or regional distribution. However, their financial behaviors and product usage patterns show notable shifts: savings account usage surged significantly (average increased from 192.3% to 307.7%), along with a marked rise in payment services (140.6% to 206.8%) and health insurance adoption (77.3% to 136.4%). Additionally, deposit account activity intensified, with average monthly transactions tripling (43.6 to 132.9), although average balances declined ($53,809 to $36,988), reflecting increased transactional engagement but potentially lower retained balances.",

        'cluster_7_descriptions' : "From T0 to T1, the demographic profile of Cluster 7 remained stable, with no notable changes in age, gender, marital status, or occupation. However, their financial behaviors and product usage patterns evolved significantly. Deposit account activity increased, with average balances rising from $18,882 to $19,767 and monthly transactions surging from 54.4 to 88.1. Product usage trends showed a decline in Payment Services (228.2% to 206.1%) and Savings Account usage (343.9% to 337.8%), while Health Insurance adoption grew notably from 140.8% to 172.1%. Additionally, cash flow patterns became more dynamic, with higher average monthly inflows and outflows, and total inflow amounts increasing by 46%.",

        'cluster_8_descriptions' : "Between T0 and T1, the demographic profile of Cluster 8 remained stable, with no significant changes in age, gender, marital status, occupation, or region. However, key financial behaviors and product usage patterns shifted notably: savings account usage increased (average rising from 287.0% to 325.0%), health insurance adoption grew significantly (average increasing from 104.3% to 145.7%), and lending activity expanded (average rising from 7.6% to 22.8%). Additionally, deposit account engagement intensified, with average monthly transactions increasing from 78.0 to 99.1, though average balances declined slightly from $17,653 to $15,173, reflecting higher cash flow activity and outflows.",

        'cluster_9_descriptions' : """
            1. **Demographic Profile**: The demographic profile of Cluster 9 remained largely stable from T0 to T1, with no changes in age, gender, education, marital status, or region. However, there was a notable shift in occupation, with "Freelancer" transitioning to "Corporate Employee" as the most common occupation.  

            2. **Financial Behaviors & Product Usage**: Cluster 9 demonstrated significant growth in financial engagement at T1, with marked increases in average savings account usage (286.4% vs. 169.0%), health insurance (119.4% vs. 69.8%), and payment services (188.4% vs. 122.1%). Deposit account activity surged, with monthly transactions tripling (131.9 vs. 46.3), while average balances decreased ($14,156 vs. $46,411), indicating higher transaction frequency but smaller balances. Cash flow patterns also intensified, with total inflows and outflows rising substantially, reflecting increased financial activity and liquidity.
        """,

        'cluster_10_descriptions' : """
            1. **Demographics**: The demographic profile of Cluster 10 remained stable from T0 to T1, with no notable shifts in age, gender, marital status, education, occupation, or regional distribution.

            2. **Financial Behaviors and Product Usage**: Cluster 10 exhibited notable changes in financial behaviors, including a significant increase in average deposit account balances (from $43,306 to $56,807) and monthly transactions (from 65.1 to 131.6). Product usage patterns showed increased engagement with health insurance (average usage rose from 156.4% to 170.7%) and lending services (average lending increased from 13.5% to 19.7%), while payment services usage declined slightly (from 284.9% to 254.4%). Additionally, cash flow activity intensified, with average monthly inflows more than tripling (from 24.5 to 85.2) and total outflows rising significantly (from $53,134 to $76,825).
        """,
    }

    return change_analysis_descriptions

# CONSTRAINTS

def get_allowed_values():
    """Return the allowed prediction values section"""
    return """
    Prediction Values must STRICTLY be as the following only:
    **Allowed Prediction Values:**
    - Education: 'less than high school', 'others', 'high school', "bachelor's degree", 'vocational certificate', "master's degree", "doctorate's degree"
    - Occupation Group: 'Freelancer', 'Corporate Employee', 'Student', 'Entrepreneur', 'Other', 'Unemployed', 'Professional', 'Homemaker', 'Agriculture/Trade'
    - Marital Status: 'single', 'others', 'widow', 'married', 'separate', 'divorce'
    - Region: 'Northeastern', 'Southern', 'Northern', 'Central', 'Eastern', 'Western'
    """

def get_constraints():
    return """
    ## Prediction Constraints

    ### Educational Level Prediction:
    - Age Constraints: Apply realistic age-based educational progression limits:
        - Ages 18-25: High likelihood of educational advancement (high school → bachelor's, bachelor's → master's)
        - Ages 26-35: Moderate likelihood, especially for career advancement (bachelor's → master's, professional certifications)
        - Ages 36-45: Lower likelihood, mainly for career-driven education (MBA, professional development)
        - Ages 46+: Very low likelihood of formal degree advancement; focus on professional certifications if any
        - Key Rule: If customer is 40+ with high school or less education, educational level will likely remain unchanged unless exceptional financial indicators suggest otherwise

    ### Marital Status Prediction:
    - Rule-Based Constraints: Apply strict marital status transition rules:
        - If T0 = "divorce": T1 must remain "divorce" (divorced individuals don't revert to "single" status)
        - If T0 = "others": T1 must remain "others" (maintain special status categories)

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

    ### Number of Children Prediction:
    - **Rule-Based Constraints**: Apply strict marital status-based rules:
        - If marital_status_t1 = "single": num_children_t1 must be 0 (single individuals are assumed to have no children in this model)
    """

def get_constraints_action():
    return """
    ## Constraints

    ### Educational Level:
    - Age Constraints: Apply realistic age-based educational progression limits:
        - Ages 18-25: High likelihood of educational advancement (high school → bachelor's, bachelor's → master's)
        - Ages 26-35: Moderate likelihood, especially for career advancement (bachelor's → master's, professional certifications)
        - Ages 36-45: Lower likelihood, mainly for career-driven education (MBA, professional development)
        - Ages 46+: Very low likelihood of formal degree advancement; focus on professional certifications if any
        - Key Rule: If customer is 40+ with high school or less education, educational level will likely remain unchanged unless exceptional financial indicators suggest otherwise

    ### Occupation:
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
    """

def get_constraints_status():
    return """
    ## Constraints

    ### Marital Status:
    - Rule-Based Constraints: Apply strict marital status transition rules:
        - If T0 = "divorce": T1 must remain "divorce" (divorced individuals don't revert to "single" status)
        - If T0 = "others": T1 must remain "others" (maintain special status categories)

    ### Number of Children:
    - **Rule-Based Constraints**: Apply strict marital status-based rules:
        - If marital_status_t1 = "single": num_children_t1 must be 0 (single individuals are assumed to have no children in this model)
    """



# FORMAT

def format_cluster_profile_section(cluster_id, change_analysis):
    """
    Format customer data into readable prompt section
    """
    cluster_descriptions = get_cluster_desc()
    cluster_desc = cluster_descriptions.get(f"cluster_{cluster_id}_descriptions", 
                                          f"Cluster {cluster_id} description not found")
    
    change_analysis_descriptions=None
    if change_analysis:
        change_analysis_descriptions=get_change_analysis_desc()

    change_desc = None
    if change_analysis_descriptions is not None:
        change_desc = change_analysis_descriptions.get(f"cluster_{cluster_id}_descriptions", 
                                          f"Cluster {cluster_id} change analysis not found")
    
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
    
    return cluster_profile_section


def format_customer_prompt_section(customer_row):
    """
    Format customer data into readable prompt section
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


def format_customer_data_for_prompt(customer_row):
    """
    Format individual customer data from dataframe row for demographic prediction prompt
    """
    # Safely handle NaN values for numeric fields
    def safe_int(value, default=0):
        try:
            return int(float(value)) if not pd.isna(value) and str(value).lower() not in ['unknown', ''] else default
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
    
    # Extract financial profile with safer float conversion
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

def format_similar_customers_section(similar_customers_data):
    """Format the similar customers data into a prompt section."""
    if not similar_customers_data or not similar_customers_data.get('similar_customers'):
        return "No similar customers found for reference."
    
    section = "## Reference: Changes in Similar Customers\n"
    section += "Consider these observed changes from customers with similar profiles:\n\n"
    
    for i, customer in enumerate(similar_customers_data['similar_customers'], 1):
        meta = customer.get('metadata', {})
        action_t1 = meta.get('action_T1', {})
        reason = meta.get('reason', 'No reason provided')
        
        section += f"### Similar Customer {i} (Similarity Score: {customer['score']:.2f})\n"
        
        # Get T0 data from full_profile_T0 if available
        t0_data = meta.get('full_profile_T0', {})
        
        # Compare T0 and T1 states
        changes = []
        for field in ['education', 'marital_status', 'occupation', 'region', 'children']:
            t0_value = t0_data.get(field, 'Unknown')
            t1_value = action_t1.get(field, 'Unknown')
            
            if str(t0_value).lower() != str(t1_value).lower():
                changes.append(
                    f"- {field.replace('_', ' ').title()}: "
                    f"{t0_value} → {t1_value}"
                )
        
        if changes:
            section += "**Changes:**\n" + "\n".join(changes) + "\n"
        else:
            section += "- No significant demographic changes observed\n"
        
        # Add the reason for changes
        section += f"\n**Reason for changes:**\n{reason}\n"
        
        # Add financial indicators if available
        if 'full_profile_T0' in meta:
            financial_info = [
                f"- Deposit Balance: ${safe_float(meta['full_profile_T0'].get('deposit_account_balance', 0)):,.2f}",
                f"- Monthly Transactions: {meta['full_profile_T0'].get('deposit_account_transactions', 'N/A')}",
                f"- Avg Transaction: ${safe_float(meta['full_profile_T0'].get('deposit_account_transactions_avg', 0)):,.2f}"
            ]

            section += "\n**Financial Indicators at T0:**\n" + "\n".join(financial_info) + "\n"
        
        section += "\n"
    
    section += ("\n### Prediction Guidance\n"
                "Consider these patterns when predicting changes for the current customer, "
                "but focus primarily on their unique financial indicators and behavior.\n")
    
    return section


# UTILS

def safe_float(value, default=0.0):
    """Safely convert value to float, handling various edge cases."""
    if pd.isna(value) or value in ['', 'Unknown', 'unknown', 'N/A', 'NaN']:
        return default
    try:
        # Remove any currency symbols or commas
        if isinstance(value, str):
            value = value.replace('$', '').replace(',', '').strip()
        return float(value)
    except (ValueError, TypeError):
        return default