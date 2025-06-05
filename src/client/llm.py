from openai import AzureOpenAI
from src.client.config import Config


config = Config()

def get_aoi_client(**kwargs) -> AzureOpenAI:
    llm_client = AzureOpenAI(
        azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
        api_key=config.AZURE_OPENAI_API_KEY,
        api_version=config.AZURE_OPENAI_API_VERSION,
        **kwargs,
    )
    return llm_client
