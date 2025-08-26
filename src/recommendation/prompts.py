import pandas as pd


T_DIFF = '2'

def create_transaction_prediction_prompt(customer_row, categories):
    """Create prediction prompt for transaction categories."""
    customer_section = format_customer_prompt_section(customer_row)
    categories_list = ", ".join(categories)
    
    prompt = f"""# Transaction Category Prediction Task

## Task Description
Based on the customer's demographic profile and historical transaction amounts in different categories, predict which transaction categories they are likely to engage with in the next {T_DIFF} years.

{customer_section}

## Available Transaction Categories
{categories_list}

## Prediction Guidelines

### Category Descriptions:
- **loan**: Personal loans, mortgages, credit facilities
- **utility**: Electricity, water, gas, internet, phone bills
- **finance**: Banking services, investments, financial planning
- **shopping**: Retail purchases, e-commerce, consumer goods
- **financial_services**: Insurance, wealth management, financial advisory
- **health_and_care**: Medical expenses, healthcare, wellness services
- **home_lifestyle**: Home improvement, furniture, household items
- **transport_travel**: Transportation costs, travel, fuel, vehicle services
- **leisure**: Entertainment, hobbies, recreational activities
- **public_services**: Government services, taxes, public utilities

### Prediction Logic:
1. **Age-based patterns**: Consider life stage and typical spending patterns
2. **Income indicators**: Higher balances may suggest broader category engagement
3. **Occupation influence**: Different occupations have different spending patterns
4. **Family status**: Marital status and children affect spending categories
5. **Geographic factors**: Regional differences in service availability and preferences
6. **Financial behavior**: Transaction patterns indicate engagement likelihood

### Scoring Guidelines:
- 1 = Very likely to engage with this category
- 0 = Unlikely to engage with this category
- Confidence scores should reflect certainty (0.0 = very uncertain, 1.0 = very certain)

## Output Format
```json
{{
"customer_id": "{customer_row.get('CUST_ID', 'Unknown')}",
"reasoning": {{
    "loan": "",
    "utility": "",
    "finance": "",
    "shopping": "",
    "financial_services": "",
    "health_and_care": "",
    "home_lifestyle": "",
    "transport_travel": "",
    "leisure": "",
    "public_services": ""
}},
"confidence_scores": {{
    "loan": ,
    "utility": ,
    "finance": ,
    "shopping": ,
    "financial_services": ,
    "health_and_care": ,
    "home_lifestyle": ,
    "transport_travel": ,
    "leisure": ,
    "public_services": 
}},
"predictions": {{
    "loan": ,
    "utility": ,
    "finance": ,
    "shopping": ,
    "financial_services": ,
    "health_and_care": ,
    "home_lifestyle": ,
    "transport_travel": ,
    "leisure": ,
    "public_services": 
}}
}}
```

## Key Considerations
- **Life Stage Alignment**: Ensure predictions match the customer's life stage and priorities
- **Financial Capacity**: Consider the customer's apparent financial capacity for different categories
- **Essential vs. Discretionary**: Distinguish between necessary services and optional spending
- **Demographic Consistency**: Align predictions with typical patterns for similar demographics
- **Regional Factors**: Consider regional availability and cultural preferences
"""

    return prompt


def format_customer_prompt_section(customer_row):
    """
    Format customer data into readable prompt section
    """
    data = format_customer_data_for_prompt(customer_row)
    
    formatted_section = f"""## Current Customer State (Time T0)

**Customer ID:** {data['CUST_ID']}

**Demographics:**
- Age: {data['Age']} years
- Gender: {data['Gender']}
- Education level: {data['Education level']}
- Marital Status: {data['Marital status']}
- Occupation: {data['Occupation Group']}
- Number of Children: {data['Number of Children']}
- Region: {data['Region']}

**Historical Transactions:**
- Finance: {data['finance_t0']} baht
- Financial Services: {data['financial_services_t0']} baht
- Home Lifestyle: {data['home_lifestyle_t0']} baht
- Loan: {data['loan_t0']} baht
- Shopping: {data['shopping_t0']} baht
- Utility: {data['utility_t0']} baht
- Health and Care: {data['health_and_care_t0']} baht
- Transport Travel: {data['transport_travel_t0']} baht
- Leisure: {data['leisure_t0']} baht
- Public Services: {data['public_services_t0']} baht
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
        'Education level': str(customer_row.get('Education level', 'Unknown')),
        'Marital status': str(customer_row.get('Marital status', 'Unknown')),
        'Occupation Group': str(customer_row.get('Occupation Group', 'Unknown')),
        'Number of Children': safe_int(customer_row.get('Number of Children', 0)),
        'Region': str(customer_row.get('Region', 'Unknown')),
        'Age': safe_int(customer_row.get('Age', 'Unknown')),
        'Gender': str(customer_row.get('Gender', 'Unknown')),
        'CUST_ID': str(customer_row.get('CUST_ID', 'Unknown'))
    }

    transactions = {
        'finance_t0': str(customer_row.get('finance_t0', 'Unknown')),
        'financial_services_t0': str(customer_row.get('financial_services_t0', 'Unknown')),
        'home_lifestyle_t0': str(customer_row.get('home_lifestyle_t0', 'Unknown')),
        'loan_t0': safe_int(customer_row.get('loan_t0', 0)),
        'shopping_t0': str(customer_row.get('shopping_t0', 'Unknown')),
        'utility_t0': safe_int(customer_row.get('utility_t0', 'Unknown')),
        'health_and_care_t0': str(customer_row.get('health_and_care_t0', 'Unknown')),
        'transport_travel_t0': str(customer_row.get('transport_travel_t0', 'Unknown')),
        'leisure_t0': str(customer_row.get('leisure_t0', 'Unknown')),
        'public_services_t0': str(customer_row.get('public_services_t0', 'Unknown')),
    }
    
    # Combine all data
    customer_data = {
        **demographics,
        **transactions
    }
    
    return customer_data



# rag
def create_rag_transaction_prediction_prompt(customer_row, categories, similar_customers_data):
    """
    Create prediction prompt with similar customers context for transaction categories.
    
    Args:
        customer_row: Customer data row
        categories: List of transaction categories to predict
        similar_customers_data: Retrieved similar customers data
    
    Returns:
        str: Complete prompt with context
    """
    
    # Get customer profile
    customer_profile = format_customer_profile_for_transactions(customer_row)
    
    # Format similar customers examples
    similar_examples = format_similar_customers_transaction_examples(similar_customers_data)
    
    # Create categories list
    categories_str = "', '".join(categories)
    
    prompt = f"""# Transaction Category Prediction with Similar Customer Context

## Your Task
You are a financial behavior analyst. Based on the customer's demographic and financial profile, predict which transaction categories they are likely to engage with. Use the similar customers' examples as reference patterns.

## Target Customer Profile
{customer_profile}

## Similar Customer Examples
{similar_examples}

## Transaction Categories to Predict
Predict likelihood (0 = will not use, 1 = will use) for these categories:
['{categories_str}']

## Category Definitions
- **loan**: Personal loans, credit facilities
- **utility**: Electricity, water, phone bills, internet
- **finance**: Investment products, financial planning services
- **shopping**: Retail purchases, e-commerce transactions
- **financial_services**: Banking services, insurance, financial consultations
- **health_and_care**: Healthcare payments, medical services, pharmacy
- **home_lifestyle**: Home improvement, furniture, household items
- **transport_travel**: Transportation, travel bookings, vehicle services
- **leisure**: Entertainment, dining, recreation, hobbies
- **public_services**: Government services, taxes, official payments

## Prediction Guidelines

### Consider Customer Context:
1. **Demographic Factors**: Age, education, occupation, family status
2. **Financial Behavior**: Spending patterns, account usage, cash flow
3. **Life Stage**: Career phase, family formation, financial maturity
4. **Regional Factors**: Location-based service availability and preferences

### Learn from Similar Customers:
1. **Pattern Recognition**: How do customers with similar profiles behave?
2. **Life Stage Patterns**: What categories are common for this demographic?
3. **Financial Capacity**: Do spending patterns support certain categories?
4. **Behavioral Consistency**: Are there consistent patterns across similar customers?

### Prediction Logic:
- **Essential Services**: Utility, basic financial services (high likelihood for most)
- **Life Stage Services**: Health care varies by age, family services vary by children
- **Income-Dependent**: Loans, investments depend on financial capacity
- **Lifestyle Services**: Shopping, leisure depend on demographics and income
- **Professional Needs**: Business services depend on occupation type

## Output Format
```json
{{
  "customer_id": "{customer_row.get('CUST_ID', customer_row.get('cust_id', 'unknown'))}",
  "predictions": {{
    {', '.join([f'"{cat}": 0' for cat in categories])}
  }},
  "confidence_scores": {{
    {', '.join([f'"{cat}": 0.0' for cat in categories])}
  }},
  "reasoning": {{
    {', '.join([f'"{cat}": "explanation based on customer profile and similar examples"' for cat in categories])}
  }}
}}
```

## Key Considerations
- **Similar Customer Patterns**: Reference the examples but adapt to this customer's unique profile
- **Demographic Logic**: Ensure predictions align with customer's life stage and capacity
- **Financial Realism**: Consider if customer's financial profile supports each category
- **Consistency**: Ensure predictions are internally consistent and logical
- **Context Integration**: Blend insights from similar customers with individual customer analysis

Provide realistic predictions based on the customer's profile and the patterns observed in similar customers.
"""
    
    return prompt


def format_customer_profile_for_transactions(customer_row):
    """Format customer profile for transaction prediction using available data."""
    try:
        # Extract key demographic info
        customer_id = customer_row.get('CUST_ID', customer_row.get('cust_id', 'unknown'))
        age = customer_row.get('Age', 'unknown')
        gender = customer_row.get('Gender', 'unknown')
        education = customer_row.get('Education level', 'unknown')
        marital_status = customer_row.get('Marital status', 'unknown')
        occupation = customer_row.get('Occupation Group', 'unknown')
        region = customer_row.get('Region', 'unknown')
        children = customer_row.get('Number of Children', 0)
        
        # Use the existing demographic summary
        demog_summary = customer_row.get('Demog Summary', 'No demographic summary available')
        
        profile = f"""**Customer ID:** {customer_id}

**Demographic Summary:**
{demog_summary}

**Key Demographics:**
- Age: {age} years
- Gender: {gender}
- Education: {education}
- Marital Status: {marital_status}
- Occupation: {occupation}
- Region: {region}
- Number of Children: {children}

**Current Transaction Categories (if available):**"""
        
        # Add current transaction usage if available
        transaction_categories = ['loan', 'utility', 'finance', 'shopping', 'financial_services', 
                                'health_and_care', 'home_lifestyle', 'transport_travel', 
                                'leisure', 'public_services']
        
        current_usage = []
        for cat in transaction_categories:
            if cat in customer_row:
                usage = customer_row.get(cat, 0)
                status = "✓ Active" if usage == 1 else "✗ Not Active"
                current_usage.append(f"- {cat}: {status}")
        
        if current_usage:
            profile += "\n" + "\n".join(current_usage)
        else:
            profile += "\n- Current transaction data not available (this is what we're predicting)"
        
        # Add financial indicators if available
        financial_cols = ['Deposit Account Balance', 'Savings Account', 'Payment', 'Health Insurance']
        if any(col in customer_row for col in financial_cols):
            profile += f"""

**Financial Indicators:**"""
            if 'Deposit Account Balance' in customer_row:
                profile += f"\n- Deposit Balance: ${customer_row.get('Deposit Account Balance', 0):,.2f}"
            if 'Savings Account' in customer_row:
                profile += f"\n- Savings Account Usage: {customer_row.get('Savings Account', 0):.1%}"
            if 'Payment' in customer_row:
                profile += f"\n- Payment Services: {customer_row.get('Payment', 0):.1%}"
            if 'Health Insurance' in customer_row:
                profile += f"\n- Health Insurance: {customer_row.get('Health Insurance', 0):.1%}"
        
        return profile
        
    except Exception as e:
        return f"**Customer ID:** {customer_row.get('CUST_ID', 'unknown')}\n**Error formatting profile:** {str(e)}"


def format_similar_customers_transaction_examples(similar_customers_data):
    """Format similar customers data into examples for the prompt."""
    if not similar_customers_data or not similar_customers_data.get('similar_customers'):
        return "No similar customers found for reference."
    
    examples = "Here are examples from customers with similar demographic profiles:\n\n"
    
    for i, customer in enumerate(similar_customers_data['similar_customers'][:3], 1):  # Limit to top 3
        try:
            metadata = customer.get('metadata', {})
            demographics = metadata.get('demographics', {})
            transactions = metadata.get('transactions', {})
            score = customer.get('score', 0)
            
            examples += f"### Similar Customer {i} (Similarity: {score:.2f})\n"
            
            # Demographics
            examples += "**Profile:**\n"
            examples += f"- Age: {demographics.get('age', 'unknown')}, "
            examples += f"Gender: {demographics.get('gender', 'unknown')}, "
            examples += f"Education: {demographics.get('education', 'unknown')}\n"
            examples += f"- Occupation: {demographics.get('occupation', 'unknown')}, "
            examples += f"Marital Status: {demographics.get('marital_status', 'unknown')}, "
            examples += f"Children: {demographics.get('children', 0)}\n"
            examples += f"- Region: {demographics.get('region', 'unknown')}\n"
            
            # Transaction patterns
            examples += "\n**Transaction Categories Used:**\n"
            active_categories = []
            inactive_categories = []
            
            for category, usage in transactions.items():
                if usage == 1:
                    active_categories.append(category)
                else:
                    inactive_categories.append(category)
            
            if active_categories:
                examples += f"- ✓ Active: {', '.join(active_categories)}\n"
            if inactive_categories:
                examples += f"- ✗ Not Used: {', '.join(inactive_categories)}\n"
            
            examples += "\n"
            
        except Exception as e:
            examples += f"### Similar Customer {i}\n**Error processing data:** {str(e)}\n\n"
    
    examples += "\n**Pattern Analysis:**\n"
    examples += "Use these examples to understand how customers with similar demographics typically engage with different transaction categories. "
    examples += "Look for patterns in age groups, occupations, family status, and regional preferences.\n"
    
    return examples





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