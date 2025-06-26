# from src.client.llm import get_aoi_client
# from src.variables import *
# from openai import AzureOpenAI
# from typing import Optional, Dict, Any
# import os
# from dotenv import load_dotenv

# load_dotenv()


# def reflection_generator(demographics: str, actions: str, environment: str) -> Optional[Dict[str, Any]]:
#     # try:
#         # Initialize client (newer OpenAI SDK style)
#     client = get_aoi_client()

#     # observations_str = "\n".join([f"- {obs}" for obs in observations])

#     prompt = f"""Below are customer's information at a specific time
#     Customer's Demographics:
#     {demographics}
#     Customer's Actions:
#     {actions}

#     Environment:
#     {environment}

#     What 5 high-level insights can you infer from
#     the above statements, including preferences, patterns, goals, influence of environment on behavior?
# """
    
#     response = client.chat.completions.create(
#         model="gpt-4o",
#         messages=[
#             {"role": "system", "content": prompt},
#             {"role": "user", "content": ""}
#         ],
#         temperature=0.7,
#         max_tokens=1000
#     )
    
#     return {
#         "content": response.choices[0].message.content,
#         "usage": dict(response.usage),
#         # "id": response.id
#     }


# def data_refresher(
#         demographics: str,
#         actions: str,
#         environment: str,
#         reflections: str,
#         time_yr: int
#         ) -> Optional[Dict[str, Any]]:
#     # try:
#         # Initialize client (newer OpenAI SDK style)
#     client = get_aoi_client()

#     prompt = f"""
# You are an expert in customer behavior modeling for a financial bank.

# You are given the following profile of a customer at time T0:
# {demographics}
# {actions}
# {reflections}

# Given that we have no new explicit data about this customer at time T1 (now), represented as {time_yr} years after.
# Also given that the environment at time T1 is:
# {environment}
# Please infer their most likely current demographics. Clearly indicate any assumptions or inferences you make.

#     """
    
#     response = client.chat.completions.create(
#         model="gpt-4o",
#         messages=[
#             {"role": "system", "content": prompt},
#             {"role": "user", "content": ""}
#         ],
#         temperature=0.7,
#         max_tokens=1000
#     )
    
#     return {
#         "content": response.choices[0].message.content,
#         "usage": dict(response.usage),
#         # "id": response.id
#     }
        

# def action_predictor(
#         demographics: str,
#         reflections: str,
#         environment: str,
#         ) -> Optional[Dict[str, Any]]:
#     # try:
#         # Initialize client (newer OpenAI SDK style)
#     client = get_aoi_client()

#     prompt = f"""
# You are an expert in customer behavior modeling for a financial bank.

# You are given the following profile of a customer at a specific time:
# {demographics}
# {reflections}

# Also given that the environment at the time is:
# {environment}
# Please infer their most likely actions relevant to financial products and deposit accounts. Clearly indicate any assumptions or inferences you make.

#     """
    
#     response = client.chat.completions.create(
#         model="gpt-4o",
#         messages=[
#             {"role": "system", "content": prompt},
#             {"role": "user", "content": ""}
#         ],
#         temperature=0.7,
#         max_tokens=1000
#     )
    
#     return {
#         "content": response.choices[0].message.content,
#         "usage": dict(response.usage),
#         # "id": response.id
#     }


# if __name__ == "__main__":
#     print("Reflection \n ---------")
    
#     reflection_response = reflection_generator(ya_demographics, ya_actions, ya_environment)
#     if reflection_response:
#         print(reflection_response["content"])

#     print("--------- \n Refreshed Data \n---------")

#     refreshed_response = data_refresher(ya_demographics, ya_actions, ya_environment_10yr, reflection_response["content"], time_yr)
#     if refreshed_response:
#         print(refreshed_response["content"])

#     print("--------- \n T1 Reflection \n---------")

#     reflection_response = reflection_generator(refreshed_response["content"], "", ya_environment_10yr)
#     if reflection_response:
#         print(reflection_response["content"])

#     print("--------- \n T1 Actions \n---------")

#     actions_response = action_predictor(refreshed_response["content"], reflection_response["content"], ya_environment_10yr)
#     if actions_response:
#         print(actions_response["content"])




# import pandas as pd

# train_df = pd.read_csv('src/clustering/data_v2/train_df.csv')

# for col in train_df.columns:
#     print(train_df[col].unique())




# clean test T1 actual
import pandas as pd

test_t1_actual = pd.read_csv('src/test_T1_actual/test_T1_actual.csv')

# test_t1_actual['Education level'] = test_t1_actual['Education level'].replace(
#     'vocational certificate/ diploma',
#     'vocational certificate/diploma'
# )

test_t1_actual['Marital status'] = test_t1_actual['Marital status'].replace(
    'married - non registered',
    'married'
)
test_t1_actual['Marital status'] = test_t1_actual['Marital status'].replace(
    'married - registered',
    'married'
)


test_t1_actual.to_csv('src/test_T1_actual/test_T1_actual.csv', index=False)




# clean train T1
import pandas as pd

train_T1 = pd.read_csv('src/train_T1/train_T1.csv')

train_T1['Education level'] = train_T1['Education level'].replace(
    'vocational certificate/ diploma',
    'vocational certificate/diploma'
)

train_T1['Marital status'] = train_T1['Marital status'].replace(
    'married - non registered',
    'married'
)
train_T1['Marital status'] = train_T1['Marital status'].replace(
    'married - registered',
    'married'
)


train_T1.to_csv('src/train_T1/train_T1.csv', index=False)