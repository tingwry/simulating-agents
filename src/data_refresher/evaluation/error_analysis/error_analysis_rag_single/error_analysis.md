## Demographic Prediction Accuracy Summary

*(Excluding cases where actual value is 'Unknown')*

| Field          |   Correct |   Total | Accuracy   | Error Rate   |   Excluded (Unknown) |
|:---------------|----------:|--------:|:-----------|:-------------|---------------------:|
| education      |       581 |     758 | 76.6%      | 23.4%        |                  112 |
| marital_status |       634 |     868 | 73.0%      | 27.0%        |                    2 |
| occupation     |       448 |     870 | 51.5%      | 48.5%        |                    0 |
| num_children   |       782 |     870 | 89.9%      | 10.1%        |                    0 |
| region         |       638 |     870 | 73.3%      | 26.7%        |                    0 |


## F1 Scores Evaluation

| Field          |   F1 Macro |   F1 Micro |   Num Classes |
|:---------------|-----------:|-----------:|--------------:|
| education      |      0.73  |      0.766 |             7 |
| marital_status |      0.379 |      0.73  |             7 |
| occupation     |      0.265 |      0.515 |             9 |
| num_children   |      0.571 |      0.899 |             5 |
| region         |      0.686 |      0.733 |             6 |

- **F1 Macro**: Treats all classes equally (good for imbalanced data)

- **F1 Micro**: Aggregates across all classes (weights by class size)


## Error Analysis by Field


### Education (177 errors)

**Most common prediction errors:**

- Predicted 'bachelor's degree': 61 cases
- Predicted 'high school': 32 cases
- Predicted 'master's degree': 31 cases

**Example cases:**

- **Customer 1508** (Confidence: 85%)
  - Predicted: `bachelor's degree`
  - Actual: `others`
  - Reasoning: Similar Customer 2 transitioned from 'high school' to 'bachelor's degree' within a year, reflecting an effort to improve financial and career prospects. Given this customer's age (24 years) and current occupation ('Student'), it is realistic to assume they may pursue higher education to enhance employability and financial independence. The progression to a bachelor's degree aligns with observed patterns in similar customers and age-related educational milestones.

- **Customer 1789** (Confidence: 85%)
  - Predicted: `master's degree`
  - Actual: `others`
  - Reasoning: Similar Customer 5 pursued a master's degree to enhance entrepreneurial skills and professional growth. Given the financial stability and disciplined cash flow management of Customer 1789, combined with her entrepreneurial occupation and age (37), it is likely she will seek higher education for career development. Her substantial monthly inflows suggest she has the financial capacity to invest in advanced education.

- **Customer 18** (Confidence: 85%)
  - Predicted: `master's degree`
  - Actual: `bachelor's degree`
  - Reasoning: Similar Customer 4 pursued a master's degree to advance her career, and given Customer 18's age, corporate occupation, and lack of significant financial activity, it is likely she may pursue higher education to enhance her earning potential or career prospects. Her current bachelor's degree and professional focus suggest she values career growth.

### Marital_Status (234 errors)

**Most common prediction errors:**

- Predicted 'married': 89 cases
- Predicted 'single': 80 cases
- Predicted 'married - registered': 31 cases

**Example cases:**

- **Customer 1628** (Confidence: 90%)
  - Predicted: `single`
  - Actual: `married`
  - Reasoning: There are no financial indicators (e.g., increased spending on household formation or relational expenses) suggesting a shift in marital status from 'single.' Similar Customer 1, who demonstrated financial consistency and stability, also showed no change in marital status. Additionally, the customer's age and spending patterns do not strongly suggest marriage or other relationship-related transitions in the one-year timeframe.

- **Customer 599** (Confidence: 85%)
  - Predicted: `divorce`
  - Actual: `single`
  - Reasoning: Similar Customer 1 demonstrated a marital status change due to likely life events such as divorce/separation. The customer's age and current financial inactivity suggest potential strain in household structure, which could lead to marital dissolution, particularly if financial stability is a concern.

- **Customer 2098** (Confidence: 75%)
  - Predicted: `married`
  - Actual: `divorce`
  - Reasoning: The customer is currently divorced, but at age 41, remarriage is plausible based on life-stage factors. Similar Customer 3 transitioned from divorce to marriage, suggesting this is a realistic possibility. While the customer's financial behavior remains minimalistic, remarriage could be influenced by social or emotional factors rather than financial indicators. Confidence is moderate.

### Occupation (422 errors)

**Most common prediction errors:**

- Predicted 'Corporate Employee': 168 cases
- Predicted 'Entrepreneur': 103 cases
- Predicted 'Freelancer': 71 cases

**Example cases:**

- **Customer 2995** (Confidence: 85%)
  - Predicted: `Entrepreneur`
  - Actual: `Other`
  - Reasoning: The customer’s current occupation as an entrepreneur aligns with their financial behavior, which shows disciplined cash flow management and significant account balances. Similar customers in entrepreneurial roles have shown stability unless facing major life disruptions. Therefore, the occupation is likely to remain Entrepreneur.

- **Customer 1628** (Confidence: 85%)
  - Predicted: `Homemaker`
  - Actual: `Entrepreneur`
  - Reasoning: The customer is currently unemployed and exhibits financial stability with consistent inflows and outflows, suggesting reliance on external income or support. Similar Customer 3 transitioned to 'Homemaker' due to stable financial patterns and personal dynamics. Given the lack of career-driven financial activity or skill development, it is likely this customer may also identify as a 'Homemaker,' reflecting a formal acknowledgment of household responsibilities.

- **Customer 904** (Confidence: 85%)
  - Predicted: `Entrepreneur`
  - Actual: `Unemployed`
  - Reasoning: The customer is an entrepreneur with consistent cash flow patterns. There are no signs of business instability or major shifts in financial activity that would suggest a transition to unemployment or another occupation. While Similar Customer 4 transitioned to unemployment, this change was linked to their age (57) and potential retirement planning. At 54, this customer still appears actively engaged in their business.

### Num_Children (88 errors)

**Most common prediction errors:**

- Predicted '0': 45 cases
- Predicted '1': 33 cases
- Predicted '2': 8 cases

**Example cases:**

- **Customer 4050** (Confidence: 80%)
  - Predicted: `1`
  - Actual: `0.0`
  - Reasoning: At 35 years old, the customer is within the common age range for family formation. Stable financial behavior, combined with the marital status 'married,' suggests readiness for family expansion, similar to Similar Customers 2 and 5 who added a child. Therefore, the number of children is predicted to increase from 0 to 1.

- **Customer 3431** (Confidence: 99%)
  - Predicted: `0`
  - Actual: `1.0`
  - Reasoning: At 55 years old and with no children currently, it is highly unlikely that she will have children in the next year. There are no financial indicators (e.g., increased spending on family-related items) or demographic changes suggesting a shift in her family structure. Similar customers (e.g., Customer 3 and 4) demonstrated no changes in the number of children, reinforcing the stability of this prediction.

- **Customer 3334** (Confidence: 85%)
  - Predicted: `1`
  - Actual: `0.0`
  - Reasoning: The customer is in a life stage where family formation, including having a child, is plausible. Similar Customers 3 and 5 demonstrated a pattern of transitioning from no children to one child, which aligns with the customer's age, marital status, and stable financial foundation. Increased expenses for family formation may emerge, but her financial stability indicates readiness for this transition.

### Region (232 errors)

**Most common prediction errors:**

- Predicted 'Central': 94 cases
- Predicted 'Eastern': 50 cases
- Predicted 'Northeastern': 41 cases

**Example cases:**

- **Customer 4050** (Confidence: 95%)
  - Predicted: `Northeastern`
  - Actual: `Central`
  - Reasoning: The customer exhibits consistent transaction patterns and no financial indicators of geographic relocation, similar to Similar Customer 4 who remained in their region. Additionally, the Northeastern region aligns with the customer’s current life stage and occupation stability. Thus, the region is predicted to remain 'Northeastern.'

- **Customer 3604** (Confidence: 95%)
  - Predicted: `Central`
  - Actual: `Eastern`
  - Reasoning: The customer resides in the Central region, and there are no financial indicators (e.g., inflows, outflows, or geographic spending patterns) suggesting relocation. Similar Customer 1 and Similar Customer 3, who demonstrated similar static financial behavior, remained in their respective regions. Therefore, the region is predicted to remain 'Central.'

- **Customer 3963** (Confidence: 80%)
  - Predicted: `Northeastern`
  - Actual: `Central`
  - Reasoning: Similar customers with similar profiles have demonstrated geographic mobility, often relocating for career opportunities or lifestyle preferences. Given her age and potential entrepreneurial aspirations, the Northeastern region may offer better opportunities for business growth or professional development. Financial indicators such as stable inflows and consistent spending patterns suggest she has the capacity to relocate.