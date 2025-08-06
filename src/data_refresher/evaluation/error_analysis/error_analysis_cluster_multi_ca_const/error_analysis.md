## Demographic Prediction Accuracy Summary

*(Excluding cases where actual value is 'Unknown')*

| Field          |   Correct |   Total | Accuracy   | Error Rate   |   Excluded (Unknown) |
|:---------------|----------:|--------:|:-----------|:-------------|---------------------:|
| education      |       618 |     758 | 81.5%      | 18.5%        |                  112 |
| marital_status |       678 |     868 | 78.1%      | 21.9%        |                    2 |
| occupation     |       497 |     870 | 57.1%      | 42.9%        |                    0 |
| num_children   |       801 |     870 | 92.1%      | 7.9%         |                    0 |
| region         |       697 |     870 | 80.1%      | 19.9%        |                    0 |


## F1 Scores Evaluation

| Field          |   F1 Macro |   F1 Micro |   Num Classes |
|:---------------|-----------:|-----------:|--------------:|
| education      |      0.776 |      0.815 |             7 |
| marital_status |      0.386 |      0.781 |             8 |
| occupation     |      0.23  |      0.571 |            10 |
| num_children   |      0.58  |      0.921 |             5 |
| region         |      0.765 |      0.801 |             6 |

- **F1 Macro**: Treats all classes equally (good for imbalanced data)

- **F1 Micro**: Aggregates across all classes (weights by class size)


## Error Analysis by Field


### Education (140 errors)

**Most common prediction errors:**

- Predicted 'bachelor's degree': 41 cases
- Predicted 'others': 31 cases
- Predicted 'high school': 26 cases

**Example cases:**

- **Customer 1508** (Cluster 3, Confidence: 85%)
  - Predicted: `master's degree`
  - Actual: `others`
  - Reasoning: The customer is 24 years old and currently a student. Pursuing advanced studies, such as a master's degree, aligns with the prediction for higher education. This is supported by their current age, educational trajectory, and typical behavior for individuals in this demographic cluster.

- **Customer 1789** (Cluster 4, Confidence: 85%)
  - Predicted: `bachelor's degree`
  - Actual: `others`
  - Reasoning: The customer is 37 years old, and based on the cluster's characteristics and historical evolution, there is a low likelihood of further formal education at this stage unless there are strong financial or career-driven indicators, which are absent here. The customer's entrepreneurial occupation does not strongly suggest a need for additional education.

- **Customer 110** (Cluster 6, Confidence: 80%)
  - Predicted: `master's degree`
  - Actual: `bachelor's degree`
  - Reasoning: The predicted action suggests pursuing further studies. Given the customer's age (34) and current status as a student, it is reasonable to predict that the customer will aim for a master's degree to advance career opportunities. This aligns with educational progression trends for their demographic group, especially in career-focused clusters.

### Marital_Status (190 errors)

**Most common prediction errors:**

- Predicted 'single': 82 cases
- Predicted 'married': 55 cases
- Predicted 'others': 29 cases

**Example cases:**

- **Customer 1628** (Cluster 10, Confidence: 90%)
  - Predicted: `single`
  - Actual: `married`
  - Reasoning: At age 40, the customer remains single, with no observable relationship-related spending patterns or household formation signals. Cluster trends for this demographic indicate stability in marital status, making 'single' the most likely prediction.

- **Customer 599** (Cluster 9, Confidence: 95%)
  - Predicted: `married`
  - Actual: `single`
  - Reasoning: The predicted action 'remain married' aligns with the customer's current status. Given the stable life stage and absence of significant financial or demographic indicators that suggest a change, the marital status is predicted to remain 'married' with a high confidence score of 0.95.

- **Customer 2922** (Cluster 7, Confidence: 95%)
  - Predicted: `married`
  - Actual: `single`
  - Reasoning: The predicted action 'remain_married' and the lack of any financial or transactional indicators suggesting household changes support the continuation of the current marital status. At this life stage, stability in marital status is expected.

### Occupation (373 errors)

**Most common prediction errors:**

- Predicted 'Corporate Employee': 187 cases
- Predicted 'Professional': 67 cases
- Predicted 'Entrepreneur': 55 cases

**Example cases:**

- **Customer 2995** (Cluster 4, Confidence: 90%)
  - Predicted: `Entrepreneur`
  - Actual: `Other`
  - Reasoning: The predicted action for business growth aligns with the customer's entrepreneurial status. Financial patterns such as high balances and moderate transaction volumes suggest stability and focus on business growth rather than transitioning to other occupations.

- **Customer 1628** (Cluster 10, Confidence: 85%)
  - Predicted: `Unemployed`
  - Actual: `Entrepreneur`
  - Reasoning: The customer is unemployed, with no financial indicators like increased inflows or spending on job-seeking activities suggesting a change in occupation. Cluster trends for this demographic show limited occupational shifts, supporting the prediction of 'Unemployed.'

- **Customer 904** (Cluster 5, Confidence: 85%)
  - Predicted: `Entrepreneur`
  - Actual: `Unemployed`
  - Reasoning: The customer is an entrepreneur, and there is no indication of career transition or retirement preparation in financial patterns. Entrepreneurial roles tend to exhibit stability at this age, particularly given the customer's modest financial activity and lack of lending product use.

### Num_Children (69 errors)

**Most common prediction errors:**

- Predicted '0': 52 cases
- Predicted '1': 9 cases
- Predicted '2': 6 cases

**Example cases:**

- **Customer 2808** (Cluster 9, Confidence: 95%)
  - Predicted: `1`
  - Actual: `0.0`
  - Reasoning: At age 57, the likelihood of having more children is extremely low due to biological factors. The customer already has one child, and financial patterns do not suggest any planning for family expansion. Cluster-level data also does not indicate a trend of increasing family size at this life stage.

- **Customer 3431** (Cluster 1, Confidence: 98%)
  - Predicted: `0`
  - Actual: `1.0`
  - Reasoning: The customer has no children and is 55 years old. Biological factors and her current marital status make it highly unlikely for her to have children in the future. This prediction aligns with demographic norms and the absence of relevant financial indicators.

- **Customer 3520** (Cluster 10, Confidence: 95%)
  - Predicted: `0`
  - Actual: `1.0`
  - Reasoning: The customer is married with no children, and their financial behaviors do not indicate preparation for family expansion. The predicted action 'no change' aligns with the assumption that they will remain childless.

### Region (173 errors)

**Most common prediction errors:**

- Predicted 'Central': 66 cases
- Predicted 'Northeastern': 49 cases
- Predicted 'Eastern': 25 cases

**Example cases:**

- **Customer 599** (Cluster 9, Confidence: 85%)
  - Predicted: `Northeastern`
  - Actual: `Central`
  - Reasoning: The predicted action 'remain in the Northeastern region' aligns with the customer's current location. There are no indications of geographic mobility, and cluster data reflects a tendency for individuals in this profile to remain in their existing region. The confidence score is 0.85.

- **Customer 4050** (Cluster 7, Confidence: 90%)
  - Predicted: `Northeastern`
  - Actual: `Central`
  - Reasoning: The customer resides in the Northeastern region, and the predicted action is to remain in the same region. Financial patterns do not indicate geographic mobility, such as changes in inflows or outflows. Cluster 7 is characterized by low geographic mobility, supporting this prediction.

- **Customer 3604** (Cluster 1, Confidence: 85%)
  - Predicted: `Central`
  - Actual: `Eastern`
  - Reasoning: The predicted action 'no relocation' aligns with the customer's current residence in the Central region. There are no financial or behavioral indicators suggesting relocation, and Cluster 1 members typically exhibit geographic stability.