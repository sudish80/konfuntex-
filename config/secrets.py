import os
import logging

logger = logging.getLogger(__name__)

class SecretsManager:
    """
    Interface for retrieving secrets. Default implementation looks at env vars.
    Extend this class to integrate with HashiCorp Vault, AWS Secrets Manager, etc.
    """
    
    @staticmethod
    def get_secret(key: str, default: str = None) -> str:
        # 1. Try environment variables
        val = os.environ.get(f"COLAB_AGENT_{key.upper()}")
        if val:
            return val
            
        # 2. Add placeholders here for Vault/AWS integration
        # if settings.use_vault:
        #    return vault_client.get(key)
            
        return default

    @staticmethod
    def get_all_hf_token() -> str:
        return SecretsManager.get_secret("HF_TOKEN")

    @staticmethod
    def get_openai_api_key() -> str:
        return SecretsManager.get_secret("OPENAI_API_KEY")
