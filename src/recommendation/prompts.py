import pandas as pd

T_DIFF = '2'

def create_transaction_prediction_prompt(customer_row, categories):
    """Create prediction prompt for transaction categories."""
    customer_section = format_customer_prompt_section(customer_row)
    categories_list = ", ".join(categories)
    
    prompt = f"""# Transaction Category Prediction Task

## Task Description
Based on the customer's demographic profile, predict which transaction categories they are likely to engage with in the next {T_DIFF} years.

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
    
    # Combine all data
    customer_data = {
        **demographics,
    }
    
    return customer_data

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