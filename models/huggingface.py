import os
from config.settings import settings


class HuggingFaceManager:
    def __init__(self):
        self.token = settings.hf_token
        self.cache_dir = settings.hf_cache_dir or os.path.expanduser("~/.cache/huggingface")

    def ensure_login(self) -> str:
        """Returns code snippet to authenticate with HuggingFace in Colab."""
        if self.token:
            return f"""
import os
os.environ['HF_TOKEN'] = '{self.token}'
from huggingface_hub import login
login(token='{self.token}')
"""
        return "# No HF token configured. Using public models only."

    def download_model_code(self, model_name: str, use_4bit: bool = True, use_8bit: bool = False) -> str:
        """Generate Colab code to download and load a model."""
        quant_config = ""
        if use_4bit:
            quant_config = """
from transformers import BitsAndBytesConfig
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
"""
        elif use_8bit:
            quant_config = """
quant_config = BitsAndBytesConfig(
    load_in_8bit=True,
)
"""
        return f"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
{quant_config}
model_name = "{model_name}"
print(f"Loading model: {{model_name}}")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_config if {'True' if use_4bit or use_8bit else 'False'} else None,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16,
)
print(f"Model loaded! Parameters: {{model.num_parameters() / 1e9:.2f}}B")
print(f"Device map: {{model.hf_device_map}}")
"""

    def list_models_code(self, task: str = "text-generation") -> str:
        return f"""
from huggingface_hub import HfApi
api = HfApi()
models = api.list_models(task="{task}", sort="downloads", direction=-1, limit=20)
for m in models:
    print(f"{{m.id:50s}} | downloads: {{m.downloads:,}}")
"""

    def push_to_hub_code(self, local_path: str, repo_id: str) -> str:
        return f"""
from huggingface_hub import HfApi, upload_folder
api = HfApi()
api.create_repo(repo_id="{repo_id}", exist_ok=True)
upload_folder(
    folder_path="{local_path}",
    repo_id="{repo_id}",
    commit_message="Upload fine-tuned model",
)
print(f"Model pushed to: https://huggingface.co/{{repo_id}}")
"""
