

# clean predictions 
# import pandas as pd

# pred_rag = pd.read_csv('src/similar_indiv/rag/predictions/v2/predictions_v1.csv')

# pred_rag['PRED_marital_status'] = pred_rag['PRED_marital_status'].replace(
#     'married - non registered',
#     'married'
# )
# pred_rag['PRED_marital_status'] = pred_rag['PRED_marital_status'].replace(
#     'married - registered',
#     'married'
# )
# pred_rag['PRED_marital_status'] = pred_rag['PRED_marital_status'].replace(
#     'separated',
#     'separate'
# )
# pred_rag['PRED_marital_status'] = pred_rag['PRED_marital_status'].replace(
#     'widowed',
#     'widow'
# )
# pred_rag['PRED_marital_status'] = pred_rag['PRED_marital_status'].replace(
#     'divorced',
#     'divorce'
# )


# pred_rag.to_csv('src/similar_indiv/rag/predictions/v2/predictions_v1.csv', index=False)


# clean predictions

import pandas as pd

pred_rag = pd.read_csv('src/prediction/pred_results/predictions_cluster_multi_no_ca.csv')

# pred_rag['PRED_education'] = pred_rag['PRED_education'].replace(
#     'vocational certificate/diploma',
#     'vocational certificate'
# )
pred_rag['PRED_marital_status'] = pred_rag['PRED_marital_status'].replace(
    'married - non registered',
    'married'
)
pred_rag['PRED_marital_status'] = pred_rag['PRED_marital_status'].replace(
    'married - registered',
    'married'
)
# pred_rag['PRED_marital_status'] = pred_rag['PRED_marital_status'].replace(
#     'separated',
#     'separate'
# )
# pred_rag['PRED_marital_status'] = pred_rag['PRED_marital_status'].replace(
#     'widowed',
#     'widow'
# )
# pred_rag['PRED_marital_status'] = pred_rag['PRED_marital_status'].replace(
#     'divorced',
#     'divorce'
# )

pred_rag.to_csv('src/prediction/pred_results/predictions_cluster_multi_no_ca.csv', index=False)

