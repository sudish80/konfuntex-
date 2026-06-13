from typing import Optional, Literal


class FinetuneCodeGenerator:
    def __init__(self, hf_token: Optional[str] = None):
        self.hf_token = hf_token

    def generate(self, base_model: str, method: Literal["lora", "qlora", "full"] = "qlora",
                 dataset: str = None, hf_token: str = None,
                 dataset_config: str = None, dataset_split: str = "train",
                 text_column: str = "text", max_length: int = 512,
                 num_epochs: int = 3, batch_size: int = 4,
                 learning_rate: float = 2e-4, output_dir: str = "./finetuned_model",
                 push_to_hub: bool = False, hub_repo_id: str = None,
                 lora_rank: int = 16, lora_alpha: int = 32) -> str:
        """Generate complete Colab code for fine-tuning."""
        token = hf_token or self.hf_token or "YOUR_HF_TOKEN"



        if method == "qlora":
            quant_setup = """
from transformers import BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
"""
            load_kwargs = "quantization_config=bnb_config,"
        elif method == "lora":
            quant_setup = ""
            load_kwargs = "torch_dtype=torch.float16,"
        else:
            quant_setup = ""
            load_kwargs = "torch_dtype=torch.float16,"

        dataset_load = f"""
from datasets import load_dataset
dataset = load_dataset("{dataset}", split="{dataset_split}")
if "{text_column}" in dataset.column_names:
    dataset = dataset.select_columns(["{text_column}"])
print(f"Dataset loaded: {{len(dataset)}} samples")
print(f"Sample: {{dataset[0]}}")
"""

        peft_config = f"""
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
peft_config = LoraConfig(
    r={lora_rank},
    lora_alpha={lora_alpha},
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)
"""

        return f"""
# ===== Colab Agent: Fine-tuning Setup =====
# Model: {base_model} | Method: {method.upper()} | Dataset: {dataset}

!pip install -q transformers datasets accelerate peft trl bitsandbytes

import os
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
)
{quant_setup}
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

# HuggingFace Login
from huggingface_hub import login
login(token="{token}")

# Load Model
model_name = "{base_model}"
print(f"Loading model: {{model_name}}")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    {load_kwargs}
    device_map="auto",
    trust_remote_code=True,
)
print(f"Model loaded! Params: {{model.num_parameters() / 1e9:.2f}}B")

# Prepare for k-bit training if needed
{"model = prepare_model_for_kbit_training(model)" if method in ("qlora", "lora") else ""}

# LoRA config
{peft_config if method != "full" else ""}
{"model = get_peft_model(model, peft_config)" if method != "full" else ""}
model.print_trainable_parameters()

# Load Dataset
{dataset_load}

# Tokenize
def tokenize_fn(examples):
    texts = examples["{text_column}"] if "{text_column}" in examples else examples["text"]
    return tokenizer(texts, truncation=True, padding="max_length", max_length={max_length})

tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)

# Training Args
training_args = TrainingArguments(
    output_dir="{output_dir}",
    num_train_epochs={num_epochs},
    per_device_train_batch_size={batch_size},
    gradient_accumulation_steps=4,
    learning_rate={learning_rate},
    warmup_steps=100,
    logging_steps=25,
    save_steps=500,
    save_total_limit=2,
    fp16={"True" if method != "full" else "False"},
    bf16=False,
    max_grad_norm=0.3,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit" if {"True" if method == "qlora" else "False"} else "adamw_torch",
    lr_scheduler_type="cosine",
    report_to="none",
    remove_unused_columns=False,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8),
)

# Train
print("Starting training...")
trainer.train()

# Save
trainer.save_model("{output_dir}")
tokenizer.save_pretrained("{output_dir}")
print(f"Model saved to {{output_dir}}")

# Push to Hub
{"from huggingface_hub import HfApi, upload_folder\\napi = HfApi()\\napi.create_repo(repo_id=\\\"{hub_repo_id}\\\", exist_ok=True)\\nupload_folder(folder_path=\\\"{output_dir}\\\", repo_id=\\\"{hub_repo_id}\\\")\\nprint(f\\\"Pushed to: https://huggingface.co/{{hub_repo_id}}\\\")" if push_to_hub else ""}

print("=== FINE-TUNING COMPLETE ===")
"""
