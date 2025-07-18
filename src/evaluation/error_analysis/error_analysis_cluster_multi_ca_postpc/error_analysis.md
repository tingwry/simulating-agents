## Demographic Prediction Accuracy Summary

*(Excluding cases where actual value is 'Unknown')*

| Field          |   Correct |   Total | Accuracy   | Error Rate   |   Excluded (Unknown) |
|:---------------|----------:|--------:|:-----------|:-------------|---------------------:|
| education      |       637 |     758 | 84.0%      | 16.0%        |                  112 |
| marital_status |       688 |     868 | 79.3%      | 20.7%        |                    2 |
| occupation     |       532 |     870 | 61.1%      | 38.9%        |                    0 |
| num_children   |       803 |     870 | 92.3%      | 7.7%         |                    0 |
| region         |       696 |     870 | 80.0%      | 20.0%        |                    0 |


## F1 Scores Evaluation

| Field          |   F1 Macro |   F1 Micro |   Num Classes |
|:---------------|-----------:|-----------:|--------------:|
| education      |      0.808 |      0.84  |             7 |
| marital_status |      0.517 |      0.793 |             6 |
| occupation     |      0.202 |      0.611 |            12 |
| num_children   |      0.604 |      0.923 |             5 |
| region         |      0.764 |      0.8   |             6 |

- **F1 Macro**: Treats all classes equally (good for imbalanced data)

- **F1 Micro**: Aggregates across all classes (weights by class size)


## Error Analysis by Field


### Education (121 errors)

**Most common prediction errors:**

- Predicted 'others': 35 cases
- Predicted 'bachelor's degree': 30 cases
- Predicted 'high school': 30 cases

**Example cases:**

- **Customer 1789** (Cluster 4, Confidence: 95%)
  - Predicted: `bachelor's degree`
  - Actual: `others`
  - Reasoning: The customer's current educational level is a bachelor's degree, and the predicted action indicates no plans for further education. At age 37, with stable entrepreneurial income, there are no significant indicators suggesting a return to academia. Cluster 4 customers often prioritize career and financial growth over further formal education.

- **Customer 150** (Cluster 6, Confidence: 80%)
  - Predicted: `high school`
  - Actual: `less than high school`
  - Reasoning: Given the customer's age and current occupation as a student, pursuing higher education is probable. Since the predicted action aligns with achieving educational progression, 'high school' is likely the next step toward career improvement.

- **Customer 3766** (Cluster 8, Confidence: 85%)
  - Predicted: `vocational certificate`
  - Actual: `bachelor's degree`
  - Reasoning: Given the customer's age (52 years) and current status, the likelihood of pursuing additional education is low. Most individuals above 50 with a vocational certificate tend to focus on leveraging existing skills rather than pursuing further studies unless there is a strong career incentive, which is not evident in the financial data. The customer's inactivity in financial accounts and lack of career-related changes also suggest no educational pursuits. This prediction aligns with the cluster trend, where education levels remain stable.

### Marital_Status (180 errors)

**Most common prediction errors:**

- Predicted 'single': 82 cases
- Predicted 'married': 68 cases
- Predicted 'others': 15 cases

**Example cases:**

- **Customer 1628** (Cluster 10, Confidence: 90%)
  - Predicted: `single`
  - Actual: `married`
  - Reasoning: The customer is 40 years old, single, and exhibits conservative financial habits with minimal discretionary spending. There is no indication of relationship-related spending or household formation, which aligns with the cluster's trend of stable marital status. Therefore, it is likely the customer will remain single.

- **Customer 599** (Cluster 9, Confidence: 95%)
  - Predicted: `married`
  - Actual: `single`
  - Reasoning: The customer is currently married, and there are no signs of potential changes in marital status. The action 'remain married' and the absence of life events or financial changes support this stable prediction.

- **Customer 2922** (Cluster 7, Confidence: 95%)
  - Predicted: `married`
  - Actual: `single`
  - Reasoning: The customer is young and married, with no financial or behavioral indicators suggesting a change in marital status. The prediction of 'married' is highly confident due to stability in their current situation and alignment with typical demographic trends for this age group in the cluster.

### Occupation (338 errors)

**Most common prediction errors:**

- Predicted 'Corporate Employee': 215 cases
- Predicted 'Entrepreneur': 55 cases
- Predicted 'Professional': 22 cases

**Example cases:**

- **Customer 2995** (Cluster 4, Confidence: 80%)
  - Predicted: `Entrepreneur`
  - Actual: `Other`
  - Reasoning: The customer is an entrepreneur, and there is no indication of career-related transitions or financial instability. Cluster 4's profile also emphasizes financial stability and minimal career shifts, making it unlikely the customer will change their occupation within the next year.

- **Customer 1628** (Cluster 10, Confidence: 85%)
  - Predicted: `Unemployed`
  - Actual: `Entrepreneur`
  - Reasoning: The customer is unemployed, and her financial behavior does not suggest an imminent career change, such as increased savings for skill development or significant inflows that might indicate a new job. Cluster trends also suggest moderate financial engagement with limited career-driven changes, making it likely she remains unemployed.

- **Customer 904** (Cluster 5, Confidence: 85%)
  - Predicted: `Entrepreneur`
  - Actual: `Unemployed`
  - Reasoning: The customer is an entrepreneur, and the predicted action of 'continue as an entrepreneur' aligns with their financial profile, which shows stable cash flow patterns and no signs of job loss or career transition. Age and cluster behavior further suggest career stability.

### Num_Children (67 errors)

**Most common prediction errors:**

- Predicted '0': 47 cases
- Predicted '1': 12 cases
- Predicted '2': 6 cases

**Example cases:**

- **Customer 2808** (Cluster 9, Confidence: 95%)
  - Predicted: `1`
  - Actual: `0.0`
  - Reasoning: At age 57 and with one child already, it is highly unlikely the customer will have additional children. This aligns with life stage expectations and cluster trends.

- **Customer 3431** (Cluster 1, Confidence: 95%)
  - Predicted: `0`
  - Actual: `1.0`
  - Reasoning: At age 55 and as a widow with no prior children, it is highly unlikely the customer will have children or adopt in the next year. Cluster 1 profiles also show limited family formation activities, aligning with her current status.

- **Customer 3520** (Cluster 10, Confidence: 95%)
  - Predicted: `0`
  - Actual: `1.0`
  - Reasoning: The customer has no children at T0, and the predicted action is 'no children' at T1. At age 44, biological and lifestyle factors, combined with financial stability and no indications of family expansion, strongly support this prediction.

### Region (174 errors)

**Most common prediction errors:**

- Predicted 'Central': 67 cases
- Predicted 'Northeastern': 49 cases
- Predicted 'Eastern': 25 cases

**Example cases:**

- **Customer 599** (Cluster 9, Confidence: 80%)
  - Predicted: `Northeastern`
  - Actual: `Central`
  - Reasoning: The customer resides in the Northeastern region, and the action 'remain in the Northeastern region' aligns with the prediction. There are no financial or occupational indicators suggesting a move, and geographic stability is common in middle age for Cluster 9.

- **Customer 4050** (Cluster 7, Confidence: 85%)
  - Predicted: `Northeastern`
  - Actual: `Central`
  - Reasoning: The customer resides in the Northeastern region. There are no significant transactional patterns or financial indicators suggesting geographic mobility (e.g., large one-time expenses or changes in inflow/outflow patterns). Cluster 7 also tends to show limited geographic mobility, aligning with the prediction of 'no_change.'

- **Customer 3604** (Cluster 1, Confidence: 90%)
  - Predicted: `Central`
  - Actual: `Eastern`
  - Reasoning: The customer shows no signs of geographic mobility in their financial patterns, such as relocation-related transactions or changes in cash flow. Remaining in the Central region is consistent with the behavior of Cluster 1 customers who exhibit regional stability.