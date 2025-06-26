## Demographic Prediction Accuracy Summary

| Field          |   Correct |   Total | Accuracy   | Error Rate   |
|:---------------|----------:|--------:|:-----------|:-------------|
| education      |        29 |      44 | 65.9%      | 34.1%        |
| marital_status |        37 |      44 | 84.1%      | 15.9%        |
| occupation     |        19 |      44 | 43.2%      | 56.8%        |
| num_children   |        39 |      44 | 88.6%      | 11.4%        |
| region         |        38 |      44 | 86.4%      | 13.6%        |


## Error Analysis by Field


### Education (15 errors)

**Most common prediction errors:**

- Predicted 'vocational certificate/diploma': 4 cases
- Predicted 'bachelor's degree': 3 cases
- Predicted 'master's degree': 3 cases

**Example cases:**

- **Customer 4050** (Cluster 6, Confidence: 85%)
  - Predicted: `bachelor's degree`
  - Actual: `others`
  - Reasoning: The customer is a financially stable middle-aged professional, and while their current education level is 'others,' cluster 6 trends and age suggest a likelihood of pursuing or holding a bachelor's degree for career progression. Financial behavior indicates high inflows and outflows consistent with someone leveraging financial sophistication, which aligns with a bachelor's degree. Confidence is high given their age and corporate role.

- **Customer 2922** (Cluster 8, Confidence: 85%)
  - Predicted: `high school`
  - Actual: `Unknown`
  - Reasoning: The customer is 19 years old and is currently a student. It is reasonable to assume that they are still pursuing their education at the high school level or equivalent, as there is no evidence of financial or academic indicators suggesting advancement beyond this level.

- **Customer 1508** (Cluster 8, Confidence: 85%)
  - Predicted: `bachelor's degree`
  - Actual: `others`
  - Reasoning: At age 24 and currently listed as a student, it is highly likely that the customer is pursuing higher education and will advance to complete a bachelor's degree within the next year. There is no financial activity to suggest a deviation from this path.

### Marital_Status (7 errors)

**Most common prediction errors:**

- Predicted 'single': 3 cases
- Predicted 'married': 3 cases
- Predicted 'others': 1 cases

**Example cases:**

- **Customer 1628** (Cluster 7, Confidence: 100%)
  - Predicted: `single`
  - Actual: `married`
  - Reasoning: The customer is currently single, and there are no financial patterns indicative of a potential marriage or change in marital status within the year. Cluster 7's historical trends also show stable marital status over time.

- **Customer 599** (Cluster 4, Confidence: 98%)
  - Predicted: `married`
  - Actual: `single`
  - Reasoning: The customer is married at T0, and there are no indicators of financial instability, relationship-related expenses, or changes suggesting marital status alterations. Therefore, she is likely to remain married.

- **Customer 2922** (Cluster 8, Confidence: 95%)
  - Predicted: `married`
  - Actual: `single`
  - Reasoning: The customer is already married, and there is no indication of a change in marital status given their age and lack of financial patterns suggesting separation or divorce.

### Occupation (25 errors)

**Most common prediction errors:**

- Predicted 'Corporate Employee': 12 cases
- Predicted 'Student': 4 cases
- Predicted 'Entrepreneur': 3 cases

**Example cases:**

- **Customer 2995** (Cluster 6, Confidence: 90%)
  - Predicted: `Entrepreneur`
  - Actual: `Other`
  - Reasoning: The customer is currently an entrepreneur with no financial behavior indicating a shift to a corporate role or other occupation. Given his age and stability, he is likely to continue in his current occupation.

- **Customer 1628** (Cluster 7, Confidence: 85%)
  - Predicted: `Unemployed`
  - Actual: `Entrepreneur`
  - Reasoning: The customer is currently unemployed, and while there is a slight chance of re-entering the workforce, the limited financial activity and lack of career advancement indicators suggest that their occupation status is likely to remain unchanged.

- **Customer 904** (Cluster 1, Confidence: 90%)
  - Predicted: `Entrepreneur`
  - Actual: `Unemployed`
  - Reasoning: The customer is an entrepreneur, and their financial behavior (moderate inflows and outflows without significant lending activity) aligns with entrepreneurial stability. At their age, major career changes are less likely.

### Num_Children (5 errors)

**Most common prediction errors:**

- Predicted '0': 3 cases
- Predicted '1': 1 cases
- Predicted '2': 1 cases

**Example cases:**

- **Customer 2808** (Cluster 4, Confidence: 100%)
  - Predicted: `1`
  - Actual: `0.0`
  - Reasoning: The customer already has one child, and at age 57, it is improbable she will have additional children. Her marital status of 'married' supports the prediction of maintaining one child.

- **Customer 3431** (Cluster 0, Confidence: 95%)
  - Predicted: `0`
  - Actual: `1.0`
  - Reasoning: The customer currently has zero children and is a widow at the age of 55. There is no financial evidence or life stage indicator suggesting family expansion. Therefore, the number of children is predicted to remain unchanged at 0.

- **Customer 3520** (Cluster 7, Confidence: 99%)
  - Predicted: `0`
  - Actual: `1.0`
  - Reasoning: The customer has no children and is part of a cluster with low indicators of family expansion at this life stage. Financial patterns do not suggest expenses related to children, so the number of children is expected to remain 0.

### Region (6 errors)

**Most common prediction errors:**

- Predicted 'Northeastern': 2 cases
- Predicted 'Central': 2 cases
- Predicted 'Northern': 1 cases

**Example cases:**

- **Customer 599** (Cluster 4, Confidence: 90%)
  - Predicted: `Northeastern`
  - Actual: `Central`
  - Reasoning: The customer resides in the Northeastern region, and there are no financial indicators (e.g., relocation expenses or new regional transaction activity) suggesting geographic mobility. She is likely to remain in the same region.

- **Customer 4050** (Cluster 6, Confidence: 95%)
  - Predicted: `Northeastern`
  - Actual: `Central`
  - Reasoning: The customer resides in the Northeastern region, and there are no financial indicators suggesting relocation. Cluster 6 customers tend to exhibit geographic stability, especially in central or urban regions. Confidence is high based on the lack of mobility signals.

- **Customer 3604** (Cluster 5, Confidence: 90%)
  - Predicted: `Central`
  - Actual: `Eastern`
  - Reasoning: There is no evidence of financial behavior suggesting geographic mobility, such as relocation-related expenses. The customer already resides in the central region, which is characteristic of Cluster 5, making 'Central' the most likely region.