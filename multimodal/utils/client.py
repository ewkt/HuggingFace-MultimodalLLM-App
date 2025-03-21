import os
import dotenv
from huggingface_hub import InferenceClient

class LLMClient:
    def __init__(self):
        dotenv.load_dotenv()
        api_key = os.getenv('HF_TOKEN')
        if not api_key:
            raise ValueError("HF_TOKEN is not set in the environment variables")
        self.client = InferenceClient(
            provider="hf-inference",
            api_key=api_key,
        )

    def __getattr__(self, attr):
        return getattr(self.client, attr)