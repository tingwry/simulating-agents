import os
from pathlib import Path
from dotenv import load_dotenv


class Config:
    def __init__(self, env_file: str = None):
        self.env_file = env_file or Path(__file__).resolve().parents[2] / ".env"

        # load environment variables from .env file
        if self.env_file.exists():
            load_dotenv(self.env_file, override=True)

        # read environment variables
        self.AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
        self.AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

        # validate that all required environment variables are set
        self.validate()

    def validate(self):
        required_vars = [
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_API_VERSION",
        ]
        missing_vars = [
            var
            for var in required_vars
            if not getattr(self, var)
        ]

        if missing_vars:
            raise ValueError(f"The following environment variables are missing: {', '.join(missing_vars)}")
        