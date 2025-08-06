## Demographic Prediction Accuracy Summary

*(Excluding cases where actual value is 'Unknown')*

| Field          |   Correct |   Total | Accuracy   | Error Rate   |   Excluded (Unknown) |
|:---------------|----------:|--------:|:-----------|:-------------|---------------------:|
| education      |       620 |     758 | 81.8%      | 18.2%        |                  112 |
| marital_status |       674 |     868 | 77.6%      | 22.4%        |                    2 |
| occupation     |       493 |     870 | 56.7%      | 43.3%        |                    0 |
| num_children   |       804 |     870 | 92.4%      | 7.6%         |                    0 |
| region         |       697 |     870 | 80.1%      | 19.9%        |                    0 |


## F1 Scores Evaluation

| Field          |   F1 Macro |   F1 Micro |   Num Classes |
|:---------------|-----------:|-----------:|--------------:|
| education      |      0.777 |      0.818 |             7 |
| marital_status |      0.528 |      0.776 |             6 |
| occupation     |      0.283 |      0.567 |            10 |
| num_children   |      0.605 |      0.924 |             5 |
| region         |      0.766 |      0.801 |             6 |

- **F1 Macro**: Treats all classes equally (good for imbalanced data)

- **F1 Micro**: Aggregates across all classes (weights by class size)


## Error Analysis by Field


### Education (138 errors)

**Most common prediction errors:**

- Predicted 'bachelor's degree': 41 cases
- Predicted 'high school': 35 cases
- Predicted 'others': 21 cases

**Example cases:**

- **Customer 1508** (Cluster 3, Confidence: 80%)
  - Predicted: `bachelor's degree`
  - Actual: `others`
  - Reasoning: The customer is 24 years old and currently a student. It is highly likely that they will complete their bachelor's degree within the next year, especially given their age and the typical educational trajectory of this demographic cluster.

- **Customer 1789** (Cluster 4, Confidence: 95%)
  - Predicted: `bachelor's degree`
  - Actual: `others`
  - Reasoning: The customer is 37 years old, making it unlikely for them to pursue additional formal education, particularly since they already hold a bachelor's degree. There were no financial or behavioral indicators suggesting career-driven education, so the level remains stable.

- **Customer 150** (Cluster 6, Confidence: 85%)
  - Predicted: `high school`
  - Actual: `less than high school`
  - Reasoning: The customer is 23 years old and currently has less than a high school education. Given their age and status as a student, there is a high likelihood they will complete high school within the next year. This aligns with typical educational progression for individuals in this age range.

### Marital_Status (194 errors)

**Most common prediction errors:**

- Predicted 'single': 110 cases
- Predicted 'married': 52 cases
- Predicted 'others': 25 cases

**Example cases:**

- **Customer 1628** (Cluster 10, Confidence: 99%)
  - Predicted: `single`
  - Actual: `married`
  - Reasoning: The customer is 40 years old, single, and exhibits no significant financial behavior changes indicative of household formation or relationship-related expenses. Cluster 10 has shown stability in marital status historically.

- **Customer 599** (Cluster 9, Confidence: 99%)
  - Predicted: `married`
  - Actual: `single`
  - Reasoning: The customer is currently married, and there are no indications of financial or life changes (e.g., increased household expenses or separation-related financial patterns) that would suggest a marital status change.

- **Customer 2922** (Cluster 7, Confidence: 95%)
  - Predicted: `married`
  - Actual: `single`
  - Reasoning: The customer is already married, and given the stable nature of marital status in a one-year timeframe, it is highly likely they will remain married.

### Occupation (377 errors)

**Most common prediction errors:**

- Predicted 'Corporate Employee': 137 cases
- Predicted 'Student': 68 cases
- Predicted 'Freelancer': 56 cases

**Example cases:**

- **Customer 2995** (Cluster 4, Confidence: 97%)
  - Predicted: `Entrepreneur`
  - Actual: `Other`
  - Reasoning: The customer is an entrepreneur, and their financial behavior aligns with stable business activity. There is no indication of a career shift or transition to another occupation in the short term.

- **Customer 1628** (Cluster 10, Confidence: 85%)
  - Predicted: `Unemployed`
  - Actual: `Entrepreneur`
  - Reasoning: The customer is currently unemployed and shows very low financial activity, with no indicators of career re-entry or entrepreneurial behavior. Cluster 10’s conservative financial patterns suggest continued unemployment is likely.

- **Customer 904** (Cluster 5, Confidence: 98%)
  - Predicted: `Entrepreneur`
  - Actual: `Unemployed`
  - Reasoning: The customer is an entrepreneur, consistent with the cluster profile. Their financial behavior, including modest transaction activity and stable cash flows, suggests no significant career changes.

### Num_Children (66 errors)

**Most common prediction errors:**

- Predicted '0': 47 cases
- Predicted '1': 11 cases
- Predicted '2': 6 cases

**Example cases:**

- **Customer 2808** (Cluster 9, Confidence: 99%)
  - Predicted: `1`
  - Actual: `0.0`
  - Reasoning: The customer has 1 child, and her age and marital status do not suggest an increase in the number of children. This aligns with typical cluster patterns.

- **Customer 3431** (Cluster 1, Confidence: 100%)
  - Predicted: `0`
  - Actual: `1.0`
  - Reasoning: The customer is 55 years old, single (widow), and shows no financial behavior indicative of family planning or child-related expenses. Therefore, the number of children will remain at 0.

- **Customer 3520** (Cluster 10, Confidence: 100%)
  - Predicted: `0`
  - Actual: `1.0`
  - Reasoning: The customer is married but has no children at age 44. There are no financial behaviors indicating family planning or child-related expenses, and it is unlikely that children will be introduced at this stage.

### Region (173 errors)

**Most common prediction errors:**

- Predicted 'Central': 67 cases
- Predicted 'Northeastern': 49 cases
- Predicted 'Eastern': 25 cases

**Example cases:**

- **Customer 599** (Cluster 9, Confidence: 90%)
  - Predicted: `Northeastern`
  - Actual: `Central`
  - Reasoning: The customer resides in the Northeastern region. There are no transaction patterns or financial behaviors indicating geographic mobility or relocation, so she is likely to remain in this region.

- **Customer 4050** (Cluster 7, Confidence: 92%)
  - Predicted: `Northeastern`
  - Actual: `Central`
  - Reasoning: Despite being located in the Northeastern region, transaction behaviors do not indicate any signs of relocation or geographic mobility. The cluster also shows stability in regional distribution.

- **Customer 3604** (Cluster 1, Confidence: 90%)
  - Predicted: `Central`
  - Actual: `Eastern`
  - Reasoning: The customer resides in the central region, and there are no geographic or financial indicators suggesting relocation within the next year. Cluster patterns also show low geographic mobility.