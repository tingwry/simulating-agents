## Demographic Prediction Accuracy Summary

*(Excluding cases where actual value is 'Unknown')*

| Field          |   Correct |   Total | Accuracy   | Error Rate   |   Excluded (Unknown) |
|:---------------|----------:|--------:|:-----------|:-------------|---------------------:|
| education      |       633 |     758 | 83.5%      | 16.5%        |                  112 |
| marital_status |       694 |     868 | 80.0%      | 20.0%        |                    2 |
| occupation     |       509 |     870 | 58.5%      | 41.5%        |                    0 |
| num_children   |       804 |     870 | 92.4%      | 7.6%         |                    0 |
| region         |       694 |     870 | 79.8%      | 20.2%        |                    0 |


## F1 Scores Evaluation

| Field          |   F1 Macro |   F1 Micro |   Num Classes |
|:---------------|-----------:|-----------:|--------------:|
| education      |      0.702 |      0.835 |             8 |
| marital_status |      0.553 |      0.8   |             6 |
| occupation     |      0.24  |      0.585 |            10 |
| num_children   |      0.605 |      0.924 |             5 |
| region         |      0.762 |      0.798 |             6 |

- **F1 Macro**: Treats all classes equally (good for imbalanced data)

- **F1 Micro**: Aggregates across all classes (weights by class size)


## Error Analysis by Field


### Education (125 errors)

**Most common prediction errors:**

- Predicted 'bachelor's degree': 37 cases
- Predicted 'others': 31 cases
- Predicted 'high school': 28 cases

**Example cases:**

- **Customer 1789** (Cluster 4, Confidence: 85%)
  - Predicted: `bachelor's degree`
  - Actual: `others`
  - Reasoning: The customer is already 37 years old and holds a bachelor's degree. Her financial stability, entrepreneurial focus, and cluster characteristics suggest prioritizing career stability and wealth preservation over pursuing additional formal education.

- **Customer 3963** (Cluster 8, Confidence: 70%)
  - Predicted: `bachelor's degree`
  - Actual: `vocational certificate`
  - Reasoning: The customer is predicted to pursue further education, likely transitioning from a vocational certificate to a bachelor's degree. This aligns with their age and stable financial situation, which allows for upskilling. The moderate confidence score reflects the possibility of pursuing education, though not guaranteed.

- **Customer 150** (Cluster 6, Confidence: 85%)
  - Predicted: `high school`
  - Actual: `less than high school`
  - Reasoning: The customer has expressed intent to pursue higher education, suggesting they may complete high school first before advancing further. As a student, this aligns with their current life stage and the cluster's preference for educational progression.

### Marital_Status (174 errors)

**Most common prediction errors:**

- Predicted 'single': 80 cases
- Predicted 'married': 78 cases
- Predicted 'others': 12 cases

**Example cases:**

- **Customer 1628** (Cluster 10, Confidence: 90%)
  - Predicted: `single`
  - Actual: `married`
  - Reasoning: The customer’s age (40) and financial behavior (conservative inflows and outflows, no indicators of household formation) suggest stability in their marital status. There is no evidence of significant relationship-related changes, which aligns with Cluster 10’s low likelihood of marriage or divorce at this stage.

- **Customer 599** (Cluster 9, Confidence: 90%)
  - Predicted: `married`
  - Actual: `single`
  - Reasoning: The customer is currently married, and there are no financial or demographic indicators (e.g., increased spending, household changes) suggesting divorce or remarriage. Cluster 9 generally represents a stable household structure, and this aligns with no expected change in marital status.

- **Customer 2922** (Cluster 7, Confidence: 95%)
  - Predicted: `married`
  - Actual: `single`
  - Reasoning: The predicted action to remain married and the absence of indicators of relational instability, coupled with the customer's young age, strongly suggest the marital status will remain 'married.'

### Occupation (361 errors)

**Most common prediction errors:**

- Predicted 'Corporate Employee': 180 cases
- Predicted 'Freelancer': 66 cases
- Predicted 'Entrepreneur': 54 cases

**Example cases:**

- **Customer 2995** (Cluster 4, Confidence: 95%)
  - Predicted: `Entrepreneur`
  - Actual: `Other`
  - Reasoning: The customer is predicted to continue their entrepreneurial occupation based on stable financial behaviors, including high balances and consistent inflows. There are no indicators for a career shift, and Cluster 4’s profile emphasizes occupational stability.

- **Customer 1628** (Cluster 10, Confidence: 75%)
  - Predicted: `Corporate Employee`
  - Actual: `Entrepreneur`
  - Reasoning: The customer is predicted to start a job, likely in a corporate environment, as they are unemployed but exhibit financial inflows and outflows that may indicate readiness for employment. Cluster 10 characteristics suggest mid-career individuals may pursue moderate employment opportunities to enhance financial stability, and starting a job is a logical progression given their age and financial patterns.

- **Customer 904** (Cluster 5, Confidence: 85%)
  - Predicted: `Entrepreneur`
  - Actual: `Unemployed`
  - Reasoning: The customer is currently an entrepreneur, and neither their financial inflows nor their spending patterns suggest a shift toward employment or unemployment. Their cluster's profile as financially stable and cautious entrepreneurs supports maintaining their current occupation.

### Num_Children (66 errors)

**Most common prediction errors:**

- Predicted '0': 47 cases
- Predicted '1': 11 cases
- Predicted '2': 6 cases

**Example cases:**

- **Customer 2808** (Cluster 9, Confidence: 95%)
  - Predicted: `1`
  - Actual: `0.0`
  - Reasoning: The customer is 57 years old with one child, and there are no financial behaviors indicating family planning or child-related expenses. Age-related factors and marital stability suggest that the likelihood of having more children is extremely low. Confidence in no change is very high.

- **Customer 3431** (Cluster 1, Confidence: 95%)
  - Predicted: `0`
  - Actual: `1.0`
  - Reasoning: The predicted action 'no children' aligns with the customer's current demographic status of having no children. At age 55, biological factors and the absence of family-related financial behaviors strongly support this prediction.

- **Customer 3520** (Cluster 10, Confidence: 90%)
  - Predicted: `0`
  - Actual: `1.0`
  - Reasoning: The customer remains child-free, with no financial or behavioral patterns indicating family expansion. Their age and life-stage suggest a lower likelihood of first-time parenthood, and Cluster 10 customers often remain child-free.

### Region (176 errors)

**Most common prediction errors:**

- Predicted 'Central': 74 cases
- Predicted 'Northeastern': 48 cases
- Predicted 'Eastern': 23 cases

**Example cases:**

- **Customer 599** (Cluster 9, Confidence: 75%)
  - Predicted: `Northeastern`
  - Actual: `Central`
  - Reasoning: The customer resides in the Northeastern region, and there are no financial or occupational indicators suggesting relocation (e.g., increased transactions, large cash outflows for moving-related expenses). While freelancers may have some mobility, the cluster profile suggests a tendency toward stability in residence. Thus, no regional change is anticipated.

- **Customer 4050** (Cluster 7, Confidence: 80%)
  - Predicted: `Northeastern`
  - Actual: `Central`
  - Reasoning: Transaction patterns do not suggest relocation, and the customer’s cluster is characterized by stability in their geographic region. Remaining in the Northeastern region aligns with their current life stage and financial patterns.

- **Customer 3604** (Cluster 1, Confidence: 80%)
  - Predicted: `Central`
  - Actual: `Eastern`
  - Reasoning: The customer's current region is Central, and there are no financial indicators suggesting relocation or geographic mobility. Cluster 1's stability in residence supports the prediction of no change in region.