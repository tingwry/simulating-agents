## Demographic Prediction Accuracy Summary

*(Excluding cases where actual value is 'Unknown')*

| Field          |   Correct |   Total | Accuracy   | Error Rate   |   Excluded (Unknown) |
|:---------------|----------:|--------:|:-----------|:-------------|---------------------:|
| education      |       568 |     758 | 74.9%      | 25.1%        |                  112 |
| marital_status |       682 |     868 | 78.6%      | 21.4%        |                    2 |
| occupation     |       467 |     870 | 53.7%      | 46.3%        |                    0 |
| num_children   |       791 |     870 | 90.9%      | 9.1%         |                    0 |
| region         |       650 |     870 | 74.7%      | 25.3%        |                    0 |


## F1 Scores Evaluation

| Field          |   F1 Macro |   F1 Micro |   Num Classes |
|:---------------|-----------:|-----------:|--------------:|
| education      |      0.716 |      0.749 |             7 |
| marital_status |      0.514 |      0.786 |             6 |
| occupation     |      0.221 |      0.537 |            11 |
| num_children   |      0.589 |      0.909 |             5 |
| region         |      0.707 |      0.747 |             6 |

- **F1 Macro**: Treats all classes equally (good for imbalanced data)

- **F1 Micro**: Aggregates across all classes (weights by class size)


## Error Analysis by Field


### Education (190 errors)

**Most common prediction errors:**

- Predicted 'bachelor's degree': 69 cases
- Predicted 'vocational certificate': 46 cases
- Predicted 'high school': 31 cases

**Example cases:**

- **Customer 1508**, Confidence: 85%)
  - Predicted: `bachelor's degree`
  - Actual: `others`
  - Reasoning: The customer is 24 years old and currently listed as a student. Observing similar customers, particularly those who transitioned to entrepreneurship or professional roles, it is likely that they are nearing the completion of their current education or pursuing a bachelor's degree. Given the customer's age and potential career aspirations, advancing to a bachelor's degree aligns with patterns seen in similar customers (e.g., Similar Customer 2).

- **Customer 1789**, Confidence: 85%)
  - Predicted: `master's degree`
  - Actual: `others`
  - Reasoning: Similar Customer 5 upgraded their education from bachelor's to master's degree, reflecting a strategic investment in personal growth and professional advancement. This customer’s financial inflow and outflow patterns, combined with her entrepreneurial occupation, suggest she may pursue higher education to enhance skills or business opportunities. The customer's age (37) also aligns with the likelihood of pursuing advanced education for career-driven reasons.

- **Customer 3963**, Confidence: 85%)
  - Predicted: `bachelor's degree`
  - Actual: `vocational certificate`
  - Reasoning: The customer is 28 years old and currently holds a vocational certificate. Given her age and the financial inflows of $10,730.43 per month, there is a moderate likelihood that she may pursue higher education (e.g., a bachelor's degree) to enhance her career prospects. The financial discipline seen in her cash flow further supports the possibility of self-investment in education. Similar Customer 3 and 5, who focused on career advancement, also suggest education may align with her career goals.

### Marital_Status (186 errors)

**Most common prediction errors:**

- Predicted 'single': 80 cases
- Predicted 'married': 72 cases
- Predicted 'others': 22 cases

**Example cases:**

- **Customer 1628**, Confidence: 95%)
  - Predicted: `single`
  - Actual: `married`
  - Reasoning: The customer is currently single at T0, and there are no observable financial or lifestyle indicators suggesting relationship-related changes, such as increased shared expenses or family formation. Similar Customer 1 and Similar Customer 3, who are in stable situations, remained unchanged in marital status, reinforcing this prediction.

- **Customer 599**, Confidence: 85%)
  - Predicted: `separate`
  - Actual: `single`
  - Reasoning: Similar Customer 1 experienced a marital status change from married to single, likely due to life events such as divorce or separation. Given the customer's age (43), lack of financial engagement, and freelancer occupation, there is a moderate likelihood of marital status transitioning to 'separated,' especially if financial or personal instability arises.

- **Customer 2922**, Confidence: 100%)
  - Predicted: `married`
  - Actual: `single`
  - Reasoning: The customer is already married at T0, and no financial indicators or life changes suggest a transition to another marital status within the next year. Married status is the most stable and likely to remain unchanged.

### Occupation (403 errors)

**Most common prediction errors:**

- Predicted 'Corporate Employee': 172 cases
- Predicted 'Entrepreneur': 83 cases
- Predicted 'Freelancer': 64 cases

**Example cases:**

- **Customer 2995**, Confidence: 90%)
  - Predicted: `Entrepreneur`
  - Actual: `Other`
  - Reasoning: The customer's occupation as an entrepreneur aligns with their financial behavior, which shows a high deposit balance and stable cash flow patterns. There is no evidence of financial distress or career instability that would suggest a shift to freelancing or another occupation, as observed in Similar Customers 4 and 5. Given the customer's age and financial stability, they are predicted to remain an entrepreneur.

- **Customer 1628**, Confidence: 85%)
  - Predicted: `Homemaker`
  - Actual: `Entrepreneur`
  - Reasoning: The customer is unemployed at T0, but her consistent financial behavior and cash flow suggest she may take on household responsibilities rather than re-entering the workforce. Similar Customer 3 transitioned from unemployed to homemaker in a similar situation, indicating this is a plausible outcome.

- **Customer 904**, Confidence: 90%)
  - Predicted: `Entrepreneur`
  - Actual: `Unemployed`
  - Reasoning: The customer is an entrepreneur with stable inflows and outflows ($2,000 per month). No signs of significant financial stress or diversification in transactions indicate a change in occupational status. Similar customers with entrepreneurial roles and steady cash flows (e.g., Similar Customer 1) maintained their occupations over a one-year period. Given the customer’s age and financial behavior, entrepreneurial continuity is the most likely scenario.

### Num_Children (79 errors)

**Most common prediction errors:**

- Predicted '0': 45 cases
- Predicted '1': 26 cases
- Predicted '2': 6 cases

**Example cases:**

- **Customer 4050**, Confidence: 80%)
  - Predicted: `1`
  - Actual: `0.0`
  - Reasoning: The customer currently has no children, but his age and marital status suggest a possibility of family formation. Financial stability, coupled with high inflows and controlled outflows, indicates that he may be preparing for life changes such as welcoming a child within the next year. Similar customers with similar profiles have reported this transition. Therefore, a prediction of 1 child is made with moderate confidence.

- **Customer 2808**, Confidence: 95%)
  - Predicted: `1`
  - Actual: `0.0`
  - Reasoning: The customer currently has 1 child, and age-related constraints make it highly unlikely for her to have additional children. Similar Customers show no changes in the number of children for individuals over 50, reinforcing stability in this demographic.

- **Customer 3431**, Confidence: 99%)
  - Predicted: `0`
  - Actual: `1.0`
  - Reasoning: The customer is 55 years old, widowed, and currently has no children. There are no financial patterns suggesting family-related expenses or changes indicating plans for adoption or other child-related activities. Similar customers in similar life stages and financial situations have not exhibited changes in the number of children. Thus, the number of children is predicted to remain at 0.

### Region (220 errors)

**Most common prediction errors:**

- Predicted 'Central': 84 cases
- Predicted 'Eastern': 47 cases
- Predicted 'Northeastern': 43 cases

**Example cases:**

- **Customer 4050**, Confidence: 98%)
  - Predicted: `Northeastern`
  - Actual: `Central`
  - Reasoning: The customer's financial and transaction patterns show no indications of geographic mobility. His consistent inflows and outflows align with a stable lifestyle in the Northeastern region. Similar customers with stable careers and no indicators of relocation remain in their current region. Thus, 'Northeastern' is predicted with very high confidence.

- **Customer 3604**, Confidence: 95%)
  - Predicted: `Central`
  - Actual: `Eastern`
  - Reasoning: The customer is located in the Central region, and there are no transaction patterns, inflows/outflows, or financial behaviors suggesting geographic mobility. Similar customers with stable financial profiles and no regional indicators remained in their current region, so this status is likely unchanged.

- **Customer 3963**, Confidence: 75%)
  - Predicted: `Northeastern`
  - Actual: `Central`
  - Reasoning: The customer currently resides in the Central region. However, relocation to the Northeastern region is plausible, as seen in Similar Customers 2 and 4, who moved for career or personal reasons. Given the prediction of a career transition to entrepreneurship, she may seek opportunities in a region with better business support or market access. The confidence score is moderate due to the absence of direct relocation indicators in her transaction patterns.