import os
import dotenv
from huggingface_hub import InferenceClient

class LLMClient:
    """
    This class is used to authenticate the HF client.
    """
    
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
        """
        This function is used to pass the attributes of the HF client to the LLMClient class.
        """
        return getattr(self.client, attr)