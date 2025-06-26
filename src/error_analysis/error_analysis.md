## Demographic Prediction Accuracy Summary

| Field          |   Correct |   Total | Accuracy   | Error Rate   |
|:---------------|----------:|--------:|:-----------|:-------------|
| education      |        14 |      44 | 31.8%      | 68.2%        |
| marital_status |        27 |      44 | 61.4%      | 38.6%        |
| occupation     |        21 |      44 | 47.7%      | 52.3%        |
| num_children   |        34 |      44 | 77.3%      | 22.7%        |
| region         |        39 |      44 | 88.6%      | 11.4%        |


## Error Analysis by Field


### Education (30 errors)

**Most common prediction errors:**

- Predicted 'bachelor's degree': 9 cases
- Predicted 'master's degree': 8 cases
- Predicted 'vocational certificate': 6 cases

**Example cases:**

- **Customer 2995** (Cluster 6, Confidence: 85%)
  - Predicted: `master's degree`
  - Actual: `bachelor's degree`
  - Reasoning: Given the customer's financial stability and entrepreneurial background, it is likely they may pursue advanced education such as a master's degree to enhance their professional opportunities. This aligns with Cluster 6's financially sophisticated profile and focus on personal financial growth.

- **Customer 599** (Cluster 4, Confidence: 80%)
  - Predicted: `high school`
  - Actual: `less than high school`
  - Reasoning: The customer currently has less than high school education and belongs to a cluster where modest educational progression is common. It is likely that they will attain a high school education level over time given typical life stage progression.

- **Customer 4050** (Cluster 6, Confidence: 85%)
  - Predicted: `bachelor's degree`
  - Actual: `others`
  - Reasoning: Currently, the customer has an 'others' educational level, but based on the financial stability and professional occupation characteristic of Cluster 6, it’s plausible that they may pursue formal education to align with the cluster's typical profile, which often includes a bachelor's degree.

### Marital_Status (17 errors)

**Most common prediction errors:**

- Predicted 'married': 5 cases
- Predicted 'single': 5 cases
- Predicted 'married - registered': 4 cases

**Example cases:**

- **Customer 2995** (Cluster 6, Confidence: 75%)
  - Predicted: `married`
  - Actual: `single`
  - Reasoning: At age 45, the likelihood of marriage increases, especially for financially stable individuals. The customer’s savings and health insurance usage may indicate preparation for family formation, which is common in this cluster.

- **Customer 1628** (Cluster 7, Confidence: 95%)
  - Predicted: `single`
  - Actual: `married - non registered`
  - Reasoning: The customer's financial patterns do not suggest significant changes linked to household formation or dissolution. Based on age and cluster data, remaining single is most likely.

- **Customer 599** (Cluster 4, Confidence: 95%)
  - Predicted: `married`
  - Actual: `single`
  - Reasoning: The customer is already married, and there are no financial indications suggesting a change in marital status, such as separation or divorce. This prediction aligns with the cluster's behavioral norms.

### Occupation (23 errors)

**Most common prediction errors:**

- Predicted 'Corporate Employee': 11 cases
- Predicted 'Freelancer': 5 cases
- Predicted 'Entrepreneur': 2 cases

**Example cases:**

- **Customer 2995** (Cluster 6, Confidence: 80%)
  - Predicted: `Corporate Employee`
  - Actual: `Other`
  - Reasoning: The customer's financial patterns suggest a shift toward more stable and lucrative opportunities, potentially transitioning from entrepreneurship to a corporate role. This aligns with Cluster 6's characteristics of middle-aged professionals in corporate roles.

- **Customer 1628** (Cluster 7, Confidence: 85%)
  - Predicted: `Unemployed`
  - Actual: `Entrepreneur`
  - Reasoning: The customer is currently unemployed, and there are no financial indicators (e.g., increased transaction amounts or inflows) suggesting imminent career advancement or new employment.

- **Customer 904** (Cluster 1, Confidence: 90%)
  - Predicted: `Entrepreneur`
  - Actual: `Unemployed`
  - Reasoning: The customer is currently an entrepreneur, and their financial patterns (low transaction volume and moderate inflows/outflows) suggest stability in this occupation rather than a career shift.

### Num_Children (10 errors)

**Most common prediction errors:**

- Predicted '1': 6 cases
- Predicted '0': 3 cases
- Predicted '2': 1 cases

**Example cases:**

- **Customer 2995** (Cluster 6, Confidence: 65%)
  - Predicted: `1`
  - Actual: `0.0`
  - Reasoning: The customer's financial stability and potential marriage suggest a possibility of starting a family, with one child being a realistic expectation within the given timeframe.

- **Customer 4050** (Cluster 6, Confidence: 75%)
  - Predicted: `1`
  - Actual: `0.0`
  - Reasoning: Given their age (35), marital status, and financial stability, it is likely they may begin family planning within the timeframe, leading to the addition of one child.

- **Customer 1392** (Cluster 7, Confidence: 70%)
  - Predicted: `1`
  - Actual: `0.0`
  - Reasoning: Given her age, marital status, and financial patterns, it is plausible that the customer may begin family planning, resulting in her having one child within the timeframe.

### Region (5 errors)

**Most common prediction errors:**

- Predicted 'Central': 2 cases
- Predicted 'Northeastern': 1 cases
- Predicted 'Northern': 1 cases

**Example cases:**

- **Customer 599** (Cluster 4, Confidence: 85%)
  - Predicted: `Northeastern`
  - Actual: `Central`
  - Reasoning: The customer resides in the Northeastern region, and there are no financial patterns suggesting relocation, such as increased inflow/outflow transactions or adoption of products indicating travel or mobility. This prediction aligns with their current status and cluster norms.

- **Customer 3604** (Cluster 5, Confidence: 90%)
  - Predicted: `Central`
  - Actual: `Eastern`
  - Reasoning: The customer resides in the central region and shows no financial activity indicating relocation. Cluster 5 customers tend to remain in urban central areas due to career opportunities and lifestyle preferences.

- **Customer 150** (Cluster 1, Confidence: 90%)
  - Predicted: `Northern`
  - Actual: `Central`
  - Reasoning: There is no evidence of geographic mobility in the customer’s financial behavior. Staying in the Northern region aligns with their current status and the cluster's typical patterns.