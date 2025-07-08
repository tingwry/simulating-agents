from qdrant_client import QdrantClient
from src.client.config import Config


config = Config()

def get_qdrant_client(**kwargs) -> QdrantClient:
    qdrant_client = QdrantClient(
        url=config.QDRANT_URL, 
        api_key=config.QDRANT_API_KEY,
        **kwargs,
    )
    return qdrant_client


# print(qdrant_client.get_collections())