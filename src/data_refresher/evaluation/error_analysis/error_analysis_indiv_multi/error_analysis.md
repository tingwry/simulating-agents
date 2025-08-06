## Demographic Prediction Accuracy Summary

*(Excluding cases where actual value is 'Unknown')*

| Field          |   Correct |   Total | Accuracy   | Error Rate   |   Excluded (Unknown) |
|:---------------|----------:|--------:|:-----------|:-------------|---------------------:|
| education      |       620 |     758 | 81.8%      | 18.2%        |                  112 |
| marital_status |       692 |     868 | 79.7%      | 20.3%        |                    2 |
| occupation     |       503 |     870 | 57.8%      | 42.2%        |                    0 |
| num_children   |       804 |     870 | 92.4%      | 7.6%         |                    0 |
| region         |       697 |     870 | 80.1%      | 19.9%        |                    0 |


## F1 Scores Evaluation

| Field          |   F1 Macro |   F1 Micro |   Num Classes |
|:---------------|-----------:|-----------:|--------------:|
| education      |      0.779 |      0.818 |             7 |
| marital_status |      0.553 |      0.797 |             6 |
| occupation     |      0.253 |      0.578 |            11 |
| num_children   |      0.605 |      0.924 |             5 |
| region         |      0.766 |      0.801 |             6 |

- **F1 Macro**: Treats all classes equally (good for imbalanced data)

- **F1 Micro**: Aggregates across all classes (weights by class size)


## Error Analysis by Field


### Education (138 errors)

**Most common prediction errors:**

- Predicted 'bachelor's degree': 33 cases
- Predicted 'others': 32 cases
- Predicted 'vocational certificate': 31 cases

**Example cases:**

- **Customer 1789** (Confidence: 85%)
  - Predicted: `bachelor's degree`
  - Actual: `others`
  - Reasoning: The predicted action indicates no further education, and the customer already holds a bachelor's degree. At 37 years old, the likelihood of pursuing additional formal education is low, especially considering their entrepreneurial role and stable financial patterns, which suggest focused business activity rather than academic pursuits.

- **Customer 110** (Confidence: 85%)
  - Predicted: `master's degree`
  - Actual: `bachelor's degree`
  - Reasoning: Given the predicted action to pursue further studies and the customer’s current status as a student with a bachelor's degree, it is reasonable to expect that the customer would aim to advance their education by pursuing a master's degree. Confidence is high due to the alignment of the predicted action and current life stage.

- **Customer 150** (Confidence: 80%)
  - Predicted: `high school`
  - Actual: `less than high school`
  - Reasoning: The predicted action indicates the customer will pursue further studies. At age 24, completing high school is plausible given their current educational level and demographic progression. This aligns with the typical trajectory for individuals aiming to improve career readiness.

### Marital_Status (176 errors)

**Most common prediction errors:**

- Predicted 'single': 81 cases
- Predicted 'married': 71 cases
- Predicted 'others': 20 cases

**Example cases:**

- **Customer 1628** (Confidence: 90%)
  - Predicted: `single`
  - Actual: `married`
  - Reasoning: The customer is single at the age of 40, with no financial behaviors suggesting relationship or household formation activities. There are no expenses that hint at engagement, marriage, or cohabitation. Additionally, the lack of children and absence of related financial behaviors reinforces the prediction of remaining single.

- **Customer 599** (Confidence: 90%)
  - Predicted: `married`
  - Actual: `single`
  - Reasoning: The customer is currently married and there are no financial behaviors or life stage indicators (e.g., increased household expenses, transaction activity) suggesting marital changes like divorce. At age 43, stability in marital status is more likely unless significant life events occur, which are not evident in this case. The confidence score is 0.90.

- **Customer 2922** (Confidence: 95%)
  - Predicted: `married`
  - Actual: `single`
  - Reasoning: Given the young age of 20 and a lack of any financial or behavioral indicators suggesting instability in the relationship, the customer is likely to remain married. There are no signs of household dissolution or major life disruptions.

### Occupation (367 errors)

**Most common prediction errors:**

- Predicted 'Corporate Employee': 149 cases
- Predicted 'Freelancer': 65 cases
- Predicted 'Entrepreneur': 55 cases

**Example cases:**

- **Customer 2995** (Confidence: 95%)
  - Predicted: `Entrepreneur`
  - Actual: `Other`
  - Reasoning: The customer is an entrepreneur, and their financial patterns reflect stability in this role. There is no indication of financial distress, job-seeking, or career changes. Therefore, it is expected they will remain an entrepreneur.

- **Customer 1628** (Confidence: 85%)
  - Predicted: `Unemployed`
  - Actual: `Entrepreneur`
  - Reasoning: The customer is unemployed, and there is no evidence in financial patterns (such as increased inflows, higher transaction volumes, or job-seeking-related spending) to suggest a transition to employment. The current financial stability and low activity levels indicate that the customer is likely to remain unemployed.

- **Customer 904** (Confidence: 95%)
  - Predicted: `Entrepreneur`
  - Actual: `Unemployed`
  - Reasoning: The customer is an entrepreneur, and their financial behavior shows consistent cash flow stability. There is no indication of career dissatisfaction or financial hardship that would prompt a change in occupation. Remaining an entrepreneur is the most likely scenario.

### Num_Children (66 errors)

**Most common prediction errors:**

- Predicted '0': 47 cases
- Predicted '1': 11 cases
- Predicted '2': 6 cases

**Example cases:**

- **Customer 2808** (Confidence: 95%)
  - Predicted: `1`
  - Actual: `0.0`
  - Reasoning: The customer is 57 years old, married, and already has one child. Age-related factors and the absence of financial activity indicating family planning suggest no change in the number of children.

- **Customer 3431** (Confidence: 100%)
  - Predicted: `0`
  - Actual: `1.0`
  - Reasoning: At 55 years old, the customer has no children and is widowed. Their age and financial patterns strongly indicate they will not have or adopt children in the future.

- **Customer 3520** (Confidence: 90%)
  - Predicted: `0`
  - Actual: `1.0`
  - Reasoning: The customer is unlikely to have children within the next year due to age (44 years old), stable financial patterns, and no current indicators of family planning, such as increased health or childcare-related expenses.

### Region (173 errors)

**Most common prediction errors:**

- Predicted 'Central': 67 cases
- Predicted 'Northeastern': 49 cases
- Predicted 'Eastern': 25 cases

**Example cases:**

- **Customer 599** (Confidence: 90%)
  - Predicted: `Northeastern`
  - Actual: `Central`
  - Reasoning: The customer resides in the Northeastern region. With no financial transactions, inflows, or outflows, there are no indicators of geographic mobility (e.g., job relocation, significant lifestyle changes). At 43, mobility is less likely without career-related reasons. The confidence score is 0.90 due to the stability of the customer’s current lifestyle.

- **Customer 4050** (Confidence: 75%)
  - Predicted: `Northeastern`
  - Actual: `Central`
  - Reasoning: The customer resides in the Northeastern region, and there are no financial behaviors indicating relocation, such as moving-related expenses. Their occupation and financial stability suggest geographic stability for the foreseeable future.

- **Customer 3604** (Confidence: 80%)
  - Predicted: `Central`
  - Actual: `Eastern`
  - Reasoning: The absence of financial transactions tied to relocation and the stability of the customer's occupation in the Central region strongly suggest that the customer will remain in their current region.