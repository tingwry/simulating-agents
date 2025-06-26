## Demographic Prediction Accuracy Summary

| Field          |   Correct |   Total | Accuracy   | Error Rate   |
|:---------------|----------:|--------:|:-----------|:-------------|
| education      |        27 |      44 | 61.4%      | 38.6%        |
| marital_status |        37 |      44 | 84.1%      | 15.9%        |
| occupation     |        20 |      44 | 45.5%      | 54.5%        |
| num_children   |        38 |      44 | 86.4%      | 13.6%        |
| region         |        38 |      44 | 86.4%      | 13.6%        |


## Error Analysis by Field


### Education (17 errors)

**Most common prediction errors:**

- Predicted 'bachelor's degree': 3 cases
- Predicted 'high school': 3 cases
- Predicted 'others': 3 cases

**Example cases:**

- **Customer 4050** (Cluster 6, Confidence: 80%)
  - Predicted: `bachelor's degree`
  - Actual: `others`
  - Reasoning: The customer is 35 years old and financially stable, suggesting potential motivation to pursue a bachelor's degree to align with the cluster's typical educational profile. However, given his current education level is 'others,' this change is moderately likely.

- **Customer 2922** (Cluster 8, Confidence: 90%)
  - Predicted: `high school`
  - Actual: `Unknown`
  - Reasoning: The customer is 19 years old and likely in the process of completing or has recently completed high school. No financial indicators suggest advanced education at this time.

- **Customer 1508** (Cluster 8, Confidence: 90%)
  - Predicted: `bachelor's degree`
  - Actual: `others`
  - Reasoning: At age 24, it is highly likely the customer is pursuing higher education. As a student, progressing from an 'others' educational level to a bachelor's degree aligns with age-related norms and cluster characteristics.

### Marital_Status (7 errors)

**Most common prediction errors:**

- Predicted 'single': 3 cases
- Predicted 'married': 3 cases
- Predicted 'others': 1 cases

**Example cases:**

- **Customer 1628** (Cluster 7, Confidence: 99%)
  - Predicted: `single`
  - Actual: `married`
  - Reasoning: The customer is currently single, which is consistent with Cluster 7's demographic profile. There are no financial indicators (such as significant lifestyle or household spending changes) that suggest a transition to a different marital status within one year.

- **Customer 599** (Cluster 4, Confidence: 99%)
  - Predicted: `married`
  - Actual: `single`
  - Reasoning: The customer is currently married, and there are no financial indicators suggesting relationship changes or household dissolution. Cluster 4 typically exhibits stable marital statuses for married individuals.

- **Customer 2922** (Cluster 8, Confidence: 100%)
  - Predicted: `married`
  - Actual: `single`
  - Reasoning: The customer is already married at T0. Based on the rules, marital status is unlikely to change within the one-year time frame.

### Occupation (24 errors)

**Most common prediction errors:**

- Predicted 'Corporate Employee': 10 cases
- Predicted 'Freelancer': 4 cases
- Predicted 'Entrepreneur': 3 cases

**Example cases:**

- **Customer 2995** (Cluster 6, Confidence: 95%)
  - Predicted: `Entrepreneur`
  - Actual: `Other`
  - Reasoning: Given their stable cash flow and entrepreneurial activities, it is highly likely that the customer will continue as an entrepreneur rather than transitioning to a corporate role.

- **Customer 1628** (Cluster 7, Confidence: 90%)
  - Predicted: `Unemployed`
  - Actual: `Entrepreneur`
  - Reasoning: The customer is unemployed at T0, and their financial patterns do not indicate significant career-related spending, such as job-related training or relocation expenses. This suggests they are likely to remain unemployed within the next year.

- **Customer 904** (Cluster 1, Confidence: 90%)
  - Predicted: `Entrepreneur`
  - Actual: `Unemployed`
  - Reasoning: The customer's financial behavior, including entrepreneurial cash flow patterns, supports the continuation of their current occupation as an entrepreneur. Limited financial growth suggests stability rather than career advancement or retirement.

### Num_Children (6 errors)

**Most common prediction errors:**

- Predicted '0': 3 cases
- Predicted '1': 2 cases
- Predicted '2': 1 cases

**Example cases:**

- **Customer 4050** (Cluster 6, Confidence: 70%)
  - Predicted: `1`
  - Actual: `0.0`
  - Reasoning: Given the customer's financial stability, marital status, and age, family planning is plausible within the next year. The prediction assumes the addition of one child based on typical life stage progression in this cluster.

- **Customer 2808** (Cluster 4, Confidence: 99%)
  - Predicted: `1`
  - Actual: `0.0`
  - Reasoning: The customer currently has one child, and given their age and financial inactivity, it is unlikely they will have additional children. This prediction aligns with typical life stage and family formation trends for this cluster.

- **Customer 3431** (Cluster 0, Confidence: 100%)
  - Predicted: `0`
  - Actual: `1.0`
  - Reasoning: The customer has no children currently, and there are no financial patterns or demographic indicators suggesting family expansion at this stage of life.

### Region (6 errors)

**Most common prediction errors:**

- Predicted 'Northeastern': 2 cases
- Predicted 'Central': 2 cases
- Predicted 'Northern': 1 cases

**Example cases:**

- **Customer 599** (Cluster 4, Confidence: 90%)
  - Predicted: `Northeastern`
  - Actual: `Central`
  - Reasoning: The customer resides in the Northeastern region, and there are no financial indicators suggesting relocation or geographic mobility. Cluster 4 typically remains geographically stable due to constrained finances and limited transaction activity.

- **Customer 4050** (Cluster 6, Confidence: 90%)
  - Predicted: `Northeastern`
  - Actual: `Central`
  - Reasoning: There are no transaction indicators or financial behavior patterns suggesting geographic relocation. The customer is predicted to remain in the Northeastern region.

- **Customer 3604** (Cluster 5, Confidence: 90%)
  - Predicted: `Central`
  - Actual: `Eastern`
  - Reasoning: The customer resides in the Central region, and there is no transactional activity indicating relocation. Geographic stability is characteristic of this cluster, especially given a lack of financial patterns suggesting mobility.