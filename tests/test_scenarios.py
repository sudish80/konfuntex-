"""
Phase 7 — Test scenarios for the autonomous agent.

Each scenario documents:
  - Goal (user input)
  - Expected agent decision trace
  - Code the agent would generate
  - Expected outcome
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ------------------------------------------------------------------ #
#  Scenario 1 : 7B model on T4 -> QLoRA
# ------------------------------------------------------------------ #

SCENARIO_1 = {
    "goal": "Fine-tune Mistral 7B on OpenAssistant for chat",
    "expected_trace": [
        ("plan", "Analyze model -> 7B, runtime T4 (16 GB). Recommend QLoRA."),
        ("step", "setup_environment — install deps, login HF"),
        ("step", "check_runtime — verify T4, 16 GB VRAM"),
        ("step", "download_model — load mistralai/Mistral-7B-v0.1 with 4-bit"),
        ("step", "load_dataset — OpenAssistant/oasst1"),
        ("step", "configure_training — QLoRA rank=8, lr=2e-4"),
        ("step", "train — 2-3 epochs, gradient_checkpointing=True"),
        ("step", "save_and_evaluate — save to Drive, log metrics"),
    ],
    "expected_code_snippets": [
        'BitsAndBytesConfig(load_in_4bit=True)',
        'peft.LoraConfig(r=8)',
        'TrainingArguments(fp16=True, gradient_checkpointing=True)',
        'device_map="auto"',
    ],
    "expected_outcome": "Fine-tuned adapter saved. Loss < 0.8 after 3 epochs.",
}


# ------------------------------------------------------------------ #
#  Scenario 2 : 350M model full on A100
# ------------------------------------------------------------------ #

SCENARIO_2 = {
    "goal": "Full fine-tune distilbert-base-uncased on IMDB for sentiment",
    "expected_trace": [
        ("plan", "Model 67M. A100 (40 GB) has ample VRAM for full FT."),
        ("step", "setup_environment"),
        ("step", "check_runtime — A100, >40 GB"),
        ("step", "download_model — distilbert-base-uncased (no quant)"),
        ("step", "load_dataset — imdb"),
        ("step", "configure_training — full FT, batch=16, lr=5e-5"),
        ("step", "train"),
        ("step", "save_and_evaluate"),
    ],
    "expected_code_snippets": [
        'AutoModelForSequenceClassification',
        'TrainingArguments(per_device_train_batch_size=16)',
        'load_dataset("imdb")',
    ],
    "expected_outcome": "Full model fine-tuned. Accuracy > 92% on IMDB.",
}


# ------------------------------------------------------------------ #
#  Scenario 3 : Dataset not found -> search alternatives
# ------------------------------------------------------------------ #

SCENARIO_3_ERROR_TRACE = {
    "trigger": 'load_dataset("my-nonexistent-dataset")',
    "error": "DatasetNotFound: my-nonexistent-dataset",
    "agent_response": [
        ("action", "Search HF Hub for alternatives: 'my' or 'nonexistent'"),
        ("action", "Fall back to generic dataset: databricks/databricks-dolly-15k"),
        ("action", "Proceed with fallback dataset"),
    ],
    "expected_recovery_code": """
from huggingface_hub import HfApi
api = HfApi()
alternatives = api.list_models(task="text-generation", search="my")
if not alternatives:
    dataset_name = "databricks/databricks-dolly-15k"
else:
    dataset_name = alternatives[0].id
""",
}


# ------------------------------------------------------------------ #
#  Scenario 4 : CUDA OOM -> reduce batch size -> switch runtime
# ------------------------------------------------------------------ #

SCENARIO_4_OOM_RECOVERY = {
    "trigger": "CUDA out of memory on T4 (16 GB)",
    "agent_trace": [
        ("retry_1", "Reduce batch_size 4->2, enable gradient_accum 4->8"),
        ("retry_2", "Switch runtime T4 -> V100"),
        ("retry_3", "Resume from last checkpoint on V100"),
    ],
    "expected_code": """
# Attempt 1: reduce memory
per_device_train_batch_size = 2
gradient_accumulation_steps = 8
model.gradient_checkpointing_enable()

# Attempt 2: switch runtime
! Runtime -> V100

# Attempt 3: resume
trainer.train(resume_from_checkpoint=CHECKPOINT_PATH)
""",
}


# ------------------------------------------------------------------ #
#  Scenario 5 : Resume after crash
# ------------------------------------------------------------------ #

SCENARIO_5_RESUME = {
    "goal": "Resume fine-tuning Phi-2 after Colab runtime crash",
    "state_before_crash": {
        "model": "microsoft/phi-2",
        "checkpoint_path": "/content/drive/MyDrive/colab-checkpoints/phi-2/checkpoint-1500",
        "completed_steps": 1500,
        "loss": 0.45,
    },
    "agent_resume_trace": [
        ("detect", "Checkpoint found at {checkpoint_path}"),
        ("action", "Load model + tokenizer from base, PEFT from checkpoint"),
        ("action", "Set resume_from_checkpoint=True in Trainer"),
        ("action", "Continue training for remaining steps"),
    ],
    "expected_code_snippet": """
trainer.train(resume_from_checkpoint="/content/drive/MyDrive/colab-checkpoints/phi-2/checkpoint-1500")
""",
}


# ------------------------------------------------------------------ #
#  Scenario 6 : GitHub error retrieval -> apply fix
# ------------------------------------------------------------------ #

SCENARIO_6_GITHUB_ERROR = {
    "trigger": "ModuleNotFoundError: No module named 'bitsandbytes'",
    "past_error_preview": """
## Error
ModuleNotFoundError: No module named 'bitsandbytes'
## Fix
!pip install bitsandbytes -q
# Then restart runtime
""",
    "agent_action": [
        ("retrieve", "GitHubLogger.retrieve_similar_errors('ModuleNotFoundError')"),
        ("find", "Matched 1 past error with score 0.85"),
        ("apply", "Install missing package: !pip install bitsandbytes -q"),
        ("apply", "Restart runtime and retry"),
    ],
}


def print_scenario(n: int, scenario: dict):
    print(f"\n{'='*60}")
    print(f"SCENARIO {n}")
    print(f"{'='*60}")
    print(f"Goal: {scenario.get('goal', scenario.get('trigger', 'N/A'))}")
    if "expected_trace" in scenario:
        print("\nExpected trace:")
        for kind, desc in scenario["expected_trace"]:
            print(f"  [{kind:10s}] {desc}")
    if "expected_code_snippets" in scenario:
        print("\nExpected code patterns:")
        for s in scenario["expected_code_snippets"]:
            print(f"  - {s}")
    if "expected_outcome" in scenario:
        print(f"\nOutcome: {scenario['expected_outcome']}")
    if "agent_trace" in scenario:
        print("\nOOM recovery trace:")
        for kind, desc in scenario["agent_trace"]:
            print(f"  [{kind:10s}] {desc}")
    if "agent_action" in scenario:
        print("\nActions:")
        for kind, desc in scenario["agent_action"]:
            print(f"  [{kind:10s}] {desc}")


if __name__ == "__main__":
    print_scenario(1, SCENARIO_1)
    print_scenario(2, SCENARIO_2)
    print("\n--- Scenario 3: Dataset Not Found ---")
    print(f"Trigger: {SCENARIO_3_ERROR_TRACE['trigger']}")
    print(f"Error: {SCENARIO_3_ERROR_TRACE['error']}")
    for kind, desc in SCENARIO_3_ERROR_TRACE["agent_response"]:
        print(f"  [{kind}] {desc}")
    print("\n--- Scenario 4: OOM Recovery ---")
    print(f"Trigger: {SCENARIO_4_OOM_RECOVERY['trigger']}")
    for kind, desc in SCENARIO_4_OOM_RECOVERY["agent_trace"]:
        print(f"  [{kind}] {desc}")
    print("\n--- Scenario 5: Resume After Crash ---")
    cs = SCENARIO_5_RESUME["state_before_crash"]
    print(f"State: {cs}")
    for kind, desc in SCENARIO_5_RESUME["agent_resume_trace"]:
        print(f"  [{kind}] {desc}")
    print(f"  Code: {SCENARIO_5_RESUME['expected_code_snippet'].strip()}")
    print("\n--- Scenario 6: GitHub Error Retrieval ---")
    print(f"Trigger: {SCENARIO_6_GITHUB_ERROR['trigger']}")
    for kind, desc in SCENARIO_6_GITHUB_ERROR["agent_action"]:
        print(f"  [{kind}] {desc}")
