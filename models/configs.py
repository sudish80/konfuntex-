

FINETUNE_METHODS = {
    "lora": {
        "description": "Low-Rank Adaptation - efficient, adds trainable adapters",
        "memory_efficient": True,
        "recommended_rank": 16,
        "recommended_alpha": 32,
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
        "supports_4bit": False,
        "supports_8bit": True,
    },
    "qlora": {
        "description": "QLoRA - 4-bit quantized LoRA, most memory efficient",
        "memory_efficient": True,
        "recommended_rank": 8,
        "recommended_alpha": 16,
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "supports_4bit": True,
        "supports_8bit": True,
    },
    "full": {
        "description": "Full fine-tuning - updates all parameters, requires most memory",
        "memory_efficient": False,
        "recommended_rank": None,
        "recommended_alpha": None,
        "target_modules": None,
        "supports_4bit": False,
        "supports_8bit": False,
    },
}


def estimate_memory(model_name: str) -> dict:
    model_size_map = {
        "phi-2": (2.7, "2.7B"),
        "phi-1_5": (1.4, "1.4B"),
        "phi-1": (1.3, "1.3B"),
        "llama-7b": (7, "7B"),
        "llama-13b": (13, "13B"),
        "llama-70b": (70, "70B"),
        "mistral-7b": (7, "7B"),
        "mixtral-8x7b": (47, "47B"),
        "codellama-7b": (7, "7B"),
        "codellama-13b": (13, "13B"),
        "codellama-34b": (34, "34B"),
        "gemma-2b": (2, "2B"),
        "gemma-7b": (7, "7B"),
        "qwen-1_8b": (1.8, "1.8B"),
        "qwen-7b": (7, "7B"),
        "qwen-14b": (14, "14B"),
        "qwen-72b": (72, "72B"),
        "deepseek-7b": (7, "7B"),
        "deepseek-67b": (67, "67B"),
        "falcon-7b": (7, "7B"),
        "falcon-40b": (40, "40B"),
        "stablelm-3b": (3, "3B"),
        "stablelm-7b": (7, "7B"),
        "stablelm-12b": (12, "12B"),
        "opt-1.3b": (1.3, "1.3B"),
        "opt-6.7b": (6.7, "6.7B"),
        "opt-13b": (13, "13B"),
        "opt-30b": (30, "30B"),
        "opt-66b": (66, "66B"),
        "bloom-7b": (7, "7B"),
        "bloom-176b": (176, "176B"),
        "starcoder-1b": (1, "1B"),
        "starcoder-3b": (3, "3B"),
        "starcoder-15b": (15, "15B"),
        "starcoder2-3b": (3, "3B"),
        "starcoder2-7b": (7, "7B"),
        "starcoder2-15b": (15, "15B"),
        "phi-3-mini": (3.8, "3.8B"),
        "phi-3-small": (7, "7B"),
        "phi-3-medium": (14, "14B"),
        "llama-3-8b": (8, "8B"),
        "llama-3-70b": (70, "70B"),
        "llama-3.1-8b": (8, "8B"),
        "llama-3.1-70b": (70, "70B"),
        "llama-3.1-405b": (405, "405B"),
        "codestral-22b": (22, "22B"),
        "deepseek-coder-1.3b": (1.3, "1.3B"),
        "deepseek-coder-6.7b": (6.7, "6.7B"),
        "deepseek-coder-33b": (33, "33B"),
        "codegemma-2b": (2, "2B"),
        "codegemma-7b": (7, "7B"),
        "yi-6b": (6, "6B"),
        "yi-34b": (34, "34B"),
    }
    name_lower = model_name.lower().replace("/", "-")
    for key, (params, label) in model_size_map.items():
        if key in name_lower:
            return {"parameters_b": params, "label": label}
    return {"parameters_b": 7, "label": "~7B (estimated)"}


def recommend_method(model_params_b: float, vram_gb: float = 0) -> tuple:
    if vram_gb >= 40 and model_params_b < 7:
        return ("full", "Full fine-tuning recommended. Sufficient VRAM and model is manageable.")
    if vram_gb >= 16 and model_params_b < 7:
        return ("lora", "LoRA recommended. Good balance of performance and memory.")
    if vram_gb >= 16 and model_params_b < 20:
        return ("lora", "LoRA recommended. Model fits in VRAM with LoRA.")
    if vram_gb >= 40 and model_params_b < 70:
        return ("qlora", "QLoRA recommended. Large model needs 4-bit quantization.")
    if vram_gb >= 80 and model_params_b >= 70:
        return ("qlora", "QLoRA recommended. Very large model, 4-bit quantization necessary.")
    if vram_gb < 16 and model_params_b < 3:
        return ("qlora", "QLoRA recommended. Limited VRAM, 4-bit necessary.")
    return ("qlora", "QLoRA (4-bit) recommended safest option for Colab.")


def get_lora_config(
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    target_modules: list = None,
    use_4bit: bool = True,
    use_8bit: bool = False,
    task_type: str = "CAUSAL_LM",
):
    from peft import LoraConfig
    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules or ["q_proj", "v_proj"],
        task_type=task_type,
        bias="none",
    )


def get_training_args(
    output_dir: str = "./results",
    num_epochs: int = 3,
    per_device_batch: int = 4,
    gradient_accum: int = 4,
    learning_rate: float = 2e-4,
    warmup_steps: int = 100,
    logging_steps: int = 25,
    save_steps: int = 500,
    eval_steps: int = 500,
    save_total_limit: int = 2,
    fp16: bool = True,
    bf16: bool = False,
    max_grad_norm: float = 0.3,
    max_steps: int = -1,
    gradient_checkpointing: bool = True,
    optim: str = "paged_adamw_8bit",
    lr_scheduler: str = "cosine",
    packing: bool = False,
):
    from transformers import TrainingArguments
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=gradient_accum,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        save_total_limit=save_total_limit,
        fp16=fp16,
        bf16=bf16,
        max_grad_norm=max_grad_norm,
        max_steps=max_steps,
        gradient_checkpointing=gradient_checkpointing,
        optim=optim,
        lr_scheduler_type=lr_scheduler,
        packing=packing,
        report_to="none",
    )
