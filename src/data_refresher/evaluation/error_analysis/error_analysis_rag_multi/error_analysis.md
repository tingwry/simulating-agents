## Demographic Prediction Accuracy Summary

*(Excluding cases where actual value is 'Unknown')*

| Field          |   Correct |   Total | Accuracy   | Error Rate   |   Excluded (Unknown) |
|:---------------|----------:|--------:|:-----------|:-------------|---------------------:|
| education      |       641 |     758 | 84.6%      | 15.4%        |                  112 |
| marital_status |       672 |     868 | 77.4%      | 22.6%        |                    2 |
| occupation     |       500 |     870 | 57.5%      | 42.5%        |                    0 |
| num_children   |       802 |     870 | 92.2%      | 7.8%         |                    0 |
| region         |       678 |     870 | 77.9%      | 22.1%        |                    0 |


## F1 Scores Evaluation

| Field          |   F1 Macro |   F1 Micro |   Num Classes |
|:---------------|-----------:|-----------:|--------------:|
| education      |      0.811 |      0.846 |             7 |
| marital_status |      0.444 |      0.774 |             7 |
| occupation     |      0.228 |      0.575 |            11 |
| num_children   |      0.603 |      0.922 |             5 |
| region         |      0.557 |      0.779 |             8 |

- **F1 Macro**: Treats all classes equally (good for imbalanced data)

- **F1 Micro**: Aggregates across all classes (weights by class size)


## Error Analysis by Field


### Education (117 errors)

**Most common prediction errors:**

- Predicted 'others': 34 cases
- Predicted 'bachelor's degree': 28 cases
- Predicted 'high school': 23 cases

**Example cases:**

- **Customer 1789**, Confidence: 95%)
  - Predicted: `bachelor's degree`
  - Actual: `others`
  - Reasoning: The customer has not shown any financial behavior indicative of pursuing further education, such as increased inflows or outflows for tuition-related expenses. At age 37, many individuals tend to stabilize their education level unless there is a strong career-driven need, which is not evident here. Additionally, the customer’s entrepreneurial focus suggests she is prioritizing business growth over academic advancement.

- **Customer 150**, Confidence: 85%)
  - Predicted: `high school`
  - Actual: `less than high school`
  - Reasoning: Given the predicted action to 'pursue further studies,' it is likely the customer will complete high school within the next year. The age of 23 and the current status as a student support this possibility, as similar customers often aim to complete this level of education before progressing further.

- **Customer 3766**, Confidence: 85%)
  - Predicted: `vocational certificate`
  - Actual: `bachelor's degree`
  - Reasoning: The customer is 52 years old and already holds a vocational certificate. Given the predicted action of 'no change,' combined with the customer's life stage and lack of financial activity suggesting career advancement, it is unlikely that they will pursue further education.

### Marital_Status (196 errors)

**Most common prediction errors:**

- Predicted 'married': 84 cases
- Predicted 'single': 69 cases
- Predicted 'others': 30 cases

**Example cases:**

- **Customer 1628**, Confidence: 90%)
  - Predicted: `single`
  - Actual: `married`
  - Reasoning: The customer is single, and there are no financial or lifestyle indicators (e.g., significant changes in spending patterns or shared financial goals) suggesting a shift towards marriage. Similar customers in this age range and financial situation have shown a tendency to maintain their marital status.

- **Customer 599**, Confidence: 70%)
  - Predicted: `married`
  - Actual: `single`
  - Reasoning: The customer is currently married, and there are no indicators of household dissolution or financial strain that would suggest a change in marital status. Her profile aligns with maintaining her current marital status.

- **Customer 2922**, Confidence: 95%)
  - Predicted: `married`
  - Actual: `single`
  - Reasoning: The action 'remain married' aligns with the customer's current status and age. At 19, married individuals typically maintain their marital status without significant changes unless financial or other stressors are evident, which are absent here.

### Occupation (370 errors)

**Most common prediction errors:**

- Predicted 'Corporate Employee': 182 cases
- Predicted 'Entrepreneur': 57 cases
- Predicted 'Freelancer': 36 cases

**Example cases:**

- **Customer 2995**, Confidence: 80%)
  - Predicted: `Entrepreneur`
  - Actual: `Other`
  - Reasoning: The customer is an entrepreneur, and their financial patterns (e.g., substantial current balance, consistent inflows, and outflows) suggest stability in their business. There is no indication of a need to shift to another occupation or become unemployed. Similar customers also show stability in occupational status when they are entrepreneurs at this stage of life.

- **Customer 1628**, Confidence: 80%)
  - Predicted: `Unemployed`
  - Actual: `Entrepreneur`
  - Reasoning: The customer remains unemployed, and her financial patterns (e.g., low transaction amounts, stable inflows) do not indicate a significant push towards entering the workforce. Her financial stability, supported by inflows, suggests she may not feel immediate pressure to seek employment. Similar customers have shown a tendency to remain in their current occupational status.

- **Customer 904**, Confidence: 90%)
  - Predicted: `Entrepreneur`
  - Actual: `Unemployed`
  - Reasoning: The customer’s financial inflows and outflows are consistent, suggesting that their entrepreneurial activities are stable. There is no evidence of financial distress, such as increased borrowing or reduced inflows, that would force a career change. At 54 years old, it is likely that the customer will continue with their current occupation rather than seeking employment or retiring. Similar customers in entrepreneurial roles have shown a tendency to maintain their business activities unless faced with external disruptions.

### Num_Children (68 errors)

**Most common prediction errors:**

- Predicted '0': 47 cases
- Predicted '1': 13 cases
- Predicted '2': 6 cases

**Example cases:**

- **Customer 2808**, Confidence: 90%)
  - Predicted: `1`
  - Actual: `0.0`
  - Reasoning: The predicted action for the number of children is 'no_change,' and due to natural aging factors, having additional children at age 57 is highly improbable. Customers with similar profiles typically have no changes in the number of children at this stage.

- **Customer 3431**, Confidence: 98%)
  - Predicted: `0`
  - Actual: `1.0`
  - Reasoning: The customer is highly unlikely to have children or adopt, given her age, widow status, and absence of family-related spending patterns. Her financial behavior further confirms this prediction.

- **Customer 3520**, Confidence: 95%)
  - Predicted: `0`
  - Actual: `1.0`
  - Reasoning: The customer is married but has no children and is aged 44, which is late for starting a family. Her financial behavior does not indicate preparation for family-related expenses, making it highly likely she will continue to have no children.

### Region (192 errors)

**Most common prediction errors:**

- Predicted 'Central': 71 cases
- Predicted 'Northeastern': 48 cases
- Predicted 'Eastern': 30 cases

**Example cases:**

- **Customer 4050**, Confidence: 90%)
  - Predicted: `Northeastern`
  - Actual: `Central`
  - Reasoning: The customer resides in the Northeastern region, and there are no transactional patterns or financial behaviors (e.g., high moving-related expenses or career-driven inflows from other regions) to suggest they are planning to relocate. Their financial discipline and stability align with remaining in their current region.

- **Customer 3604**, Confidence: 90%)
  - Predicted: `Central`
  - Actual: `Eastern`
  - Reasoning: The customer's stagnant financial behavior and lack of mobility-related expenses (e.g., relocation or career-driven moves) indicate stability in their geographic location. Similar customers have remained in the same region under comparable financial and demographic circumstances.

- **Customer 150**, Confidence: 80%)
  - Predicted: `Northern`
  - Actual: `Central`
  - Reasoning: The prediction suggests the customer will 'remain in Northern,' which is supported by the absence of financial activity related to relocation. Similar customers without moving expenses or indications of educational relocation typically stay in their current region.