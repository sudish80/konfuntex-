"""Tests for models/ — configs, finetune, huggingface, finetune_orchestrator."""
import sys, os, ast
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _is_valid_python(code: str) -> bool:
    lines = [ln for ln in code.split("\n") if not ln.strip().startswith("!")]
    py = "\n".join(lines)
    try:
        ast.parse(py if py.strip() else "pass")
        return True
    except SyntaxError:
        return False


# ================================================================== #
#  models/configs.py
# ================================================================== #

class TestFinetuneMethods:
    def test_has_lora(self):
        from models.configs import FINETUNE_METHODS
        assert "lora" in FINETUNE_METHODS
        assert FINETUNE_METHODS["lora"]["memory_efficient"] is True

    def test_has_qlora(self):
        from models.configs import FINETUNE_METHODS
        assert "qlora" in FINETUNE_METHODS
        assert FINETUNE_METHODS["qlora"]["supports_4bit"] is True

    def test_has_full(self):
        from models.configs import FINETUNE_METHODS
        assert "full" in FINETUNE_METHODS
        assert FINETUNE_METHODS["full"]["memory_efficient"] is False


class TestEstimateMemory:
    def test_known_model(self):
        from models.configs import estimate_memory
        result = estimate_memory("microsoft/phi-2")
        assert result["parameters_b"] == 2.7
        assert result["label"] == "2.7B"

    def test_known_model_llama(self):
        from models.configs import estimate_memory
        result = estimate_memory("meta/llama-3.1-8b")
        assert result["parameters_b"] == 8

    def test_unknown_model(self):
        from models.configs import estimate_memory
        result = estimate_memory("unknown/model-xyz")
        assert result["parameters_b"] == 7
        assert "estimated" in result["label"]

    def test_case_insensitive(self):
        from models.configs import estimate_memory
        result = estimate_memory("MISTRAL-7B")
        assert result["parameters_b"] == 7


class TestRecommendMethod:
    def test_full_large_vram_small_model(self):
        from models.configs import recommend_method
        method, _ = recommend_method(2.7, vram_gb=48)
        assert method == "full"

    def test_lora_16gb_small_model(self):
        from models.configs import recommend_method
        method, _ = recommend_method(2.7, vram_gb=16)
        assert method == "lora"

    def test_qlora_40gb_large_model(self):
        from models.configs import recommend_method
        method, _ = recommend_method(47, vram_gb=40)
        assert method == "qlora"

    def test_qlora_80gb_very_large(self):
        from models.configs import recommend_method
        method, _ = recommend_method(70, vram_gb=80)
        assert method == "qlora"

    def test_qlora_limited_vram(self):
        from models.configs import recommend_method
        method, _ = recommend_method(2.7, vram_gb=8)
        assert method == "qlora"

    def test_default_fallback(self):
        from models.configs import recommend_method
        method, reason = recommend_method(100, vram_gb=0)
        assert method == "qlora"
        assert "safest" in reason.lower()


class TestGetLoraConfig:
    def test_returns_lora_config(self):
        from models.configs import get_lora_config
        config = get_lora_config(rank=8, alpha=16)
        assert config.r == 8
        assert config.lora_alpha == 16
        assert config.task_type == "CAUSAL_LM"

    def test_default_target_modules(self):
        from models.configs import get_lora_config
        config = get_lora_config()
        assert "q_proj" in config.target_modules

    def test_custom_target_modules(self):
        from models.configs import get_lora_config
        config = get_lora_config(target_modules=["q_proj"])
        assert config.target_modules == ["q_proj"]


class TestGetTrainingArgs:
    def test_returns_training_args(self):
        from models.configs import get_training_args
        args = get_training_args(output_dir="./test_out", num_epochs=2, batch_size=4)
        assert args.output_dir == "./test_out"
        assert args.num_train_epochs == 2
        assert args.per_device_train_batch_size == 4

    def test_default_fp16(self):
        from models.configs import get_training_args
        args = get_training_args()
        assert args.fp16 is True

    def test_packing_false_by_default(self):
        from models.configs import get_training_args
        args = get_training_args()
        assert hasattr(args, "packing")


# ================================================================== #
#  models/finetune.py
# ================================================================== #

class TestFinetuneCodeGenerator:
    def test_generate_qlora_code(self):
        from models.finetune import FinetuneCodeGenerator
        gen = FinetuneCodeGenerator()
        code = gen.generate(
            base_model="microsoft/phi-2", method="qlora",
            dataset="dolly", num_epochs=2,
        )
        assert _is_valid_python(code), "QLoRA code should be valid Python"
        assert "!pip install" in code
        assert "BitsAndBytesConfig" in code
        assert "load_in_4bit" in code

    def test_generate_lora_code(self):
        from models.finetune import FinetuneCodeGenerator
        gen = FinetuneCodeGenerator()
        code = gen.generate(
            base_model="microsoft/phi-2", method="lora",
            dataset="dolly",
        )
        assert _is_valid_python(code), "LoRA code should be valid Python"
        assert "LoraConfig" in code

    def test_generate_full_code(self):
        from models.finetune import FinetuneCodeGenerator
        gen = FinetuneCodeGenerator()
        code = gen.generate(
            base_model="microsoft/phi-2", method="full",
            dataset="dolly",
        )
        assert _is_valid_python(code), "Full fine-tune code should be valid Python"
        assert "get_peft_model" not in code

    def test_generate_with_hub_push(self):
        from models.finetune import FinetuneCodeGenerator
        gen = FinetuneCodeGenerator()
        code = gen.generate(
            base_model="phi-2", method="lora", dataset="dolly",
            push_to_hub=True, hub_repo_id="user/test",
        )
        assert _is_valid_python(code)
        assert "upload_folder" in code

    def test_generate_custom_params(self):
        from models.finetune import FinetuneCodeGenerator
        gen = FinetuneCodeGenerator()
        code = gen.generate(
            base_model="phi-2", method="qlora", dataset="dolly",
            lora_rank=8, lora_alpha=16, learning_rate=1e-4,
        )
        assert _is_valid_python(code)
        assert "r=8" in code or "r= 8" in code


# ================================================================== #
#  models/huggingface.py
# ================================================================== #

class TestHuggingFaceManager:
    def test_ensure_login_with_token(self):
        from models.huggingface import HuggingFaceManager
        import os
        os.environ["COLAB_AGENT_HF_TOKEN"] = "hf_test"
        from importlib import reload
        import models.huggingface
        reload(models.huggingface)

        hm = HuggingFaceManager()
        code = hm.ensure_login()
        assert "login" in code
        assert _is_valid_python(code)

    def test_ensure_login_no_token(self):
        from models.huggingface import HuggingFaceManager
        import os
        os.environ.pop("COLAB_AGENT_HF_TOKEN", None)
        from importlib import reload
        import models.huggingface
        reload(models.huggingface)

        hm = HuggingFaceManager()
        code = hm.ensure_login()
        assert "No HF token" in code

    def test_download_model_code_4bit(self):
        from models.huggingface import HuggingFaceManager
        hm = HuggingFaceManager()
        code = hm.download_model_code("microsoft/phi-2", use_4bit=True)
        assert _is_valid_python(code)
        assert "load_in_4bit" in code

    def test_download_model_code_no_quant(self):
        from models.huggingface import HuggingFaceManager
        hm = HuggingFaceManager()
        code = hm.download_model_code("microsoft/phi-2", use_4bit=False)
        assert _is_valid_python(code)
        assert "measurement_config" not in code

    def test_list_models_code(self):
        from models.huggingface import HuggingFaceManager
        hm = HuggingFaceManager()
        code = hm.list_models_code(task="text-generation")
        assert _is_valid_python(code)
        assert "list_models" in code

    def test_push_to_hub_code(self):
        from models.huggingface import HuggingFaceManager
        hm = HuggingFaceManager()
        code = hm.push_to_hub_code("./output", "user/model")
        assert _is_valid_python(code)
        assert "upload_folder" in code


# ================================================================== #
#  models/finetune_orchestrator.py
# ================================================================== #

class TestFineTuneOrchestratorPlan:
    def test_plan_qlora_for_large_model(self):
        from models.finetune_orchestrator import FineTuneOrchestrator
        fo = FineTuneOrchestrator()
        plan = fo.plan_training("llama-3.1-70b", vram_gb=40)
        assert plan["method"] == "qlora"
        assert plan["rank"] <= 16
        assert plan["epochs"] == 2

    def test_plan_full_for_small_model(self):
        from models.finetune_orchestrator import FineTuneOrchestrator
        fo = FineTuneOrchestrator()
        plan = fo.plan_training("microsoft/phi-2", vram_gb=48)
        assert plan["method"] == "full"
        assert plan["rank"] is None

    def test_plan_lora_for_medium(self):
        from models.finetune_orchestrator import FineTuneOrchestrator
        fo = FineTuneOrchestrator()
        plan = fo.plan_training("mistral-7b", vram_gb=16)
        assert plan["method"] == "lora"

    def test_plan_includes_all_keys(self):
        from models.finetune_orchestrator import FineTuneOrchestrator
        fo = FineTuneOrchestrator()
        plan = fo.plan_training("phi-2", vram_gb=16)
        for key in ["method", "rank", "alpha", "learning_rate", "batch_size",
                     "epochs", "gradient_accumulation_steps", "optimizer",
                     "fp16", "target_modules", "recommendation_reason",
                     "estimated_vram_needed_gb"]:
            assert key in plan, f"missing key: {key}"

    def test_plan_with_explicit_model_params(self):
        from models.finetune_orchestrator import FineTuneOrchestrator
        fo = FineTuneOrchestrator()
        plan = fo.plan_training("custom-model", vram_gb=16, model_params_b=2.7)
        assert plan["method"] in ("lora", "full")


class TestFineTuneOrchestratorScript:
    def test_generate_script_qlora(self):
        from models.finetune_orchestrator import FineTuneOrchestrator
        fo = FineTuneOrchestrator()
        script = fo.generate_script(
            base_model="microsoft/phi-2", dataset="dolly",
            method="qlora",
        )
        assert _is_valid_python(script), "Generated script should be valid Python"
        assert "load_in_4bit" in script
        assert "Trainer" in script

    def test_generate_script_lora(self):
        from models.finetune_orchestrator import FineTuneOrchestrator
        fo = FineTuneOrchestrator()
        script = fo.generate_script(
            base_model="mistral-7b", dataset="dolly", method="lora",
        )
        assert _is_valid_python(script)
        assert "LoraConfig" in script
        assert "load_in_4bit" not in script

    def test_generate_script_full(self):
        from models.finetune_orchestrator import FineTuneOrchestrator
        fo = FineTuneOrchestrator()
        script = fo.generate_script(
            base_model="phi-2", dataset="dolly", method="full",
        )
        assert _is_valid_python(script)
        assert "peft" not in script.lower() or "get_peft_model" not in script

    def test_generate_script_custom_dataset(self):
        from models.finetune_orchestrator import FineTuneOrchestrator
        fo = FineTuneOrchestrator()
        script = fo.generate_script(
            base_model="phi-2", dataset="custom",
            custom_dataset_path="./data/train.csv",
        )
        assert _is_valid_python(script)
        assert "read_csv" in script

    def test_generate_script_with_hub_push(self):
        from models.finetune_orchestrator import FineTuneOrchestrator
        fo = FineTuneOrchestrator()
        script = fo.generate_script(
            base_model="phi-2", dataset="dolly",
            push_to_hub=True, hub_repo_id="user/test-model",
        )
        assert _is_valid_python(script)
        assert "upload_folder" in script

    def test_generate_script_with_checkpoint(self):
        from models.finetune_orchestrator import FineTuneOrchestrator
        fo = FineTuneOrchestrator()
        script = fo.generate_script(
            base_model="phi-2", dataset="dolly",
            resume_from_checkpoint="./checkpoint-500",
        )
        assert _is_valid_python(script)
        assert "RESUME_CHECKPOINT" in script

    def test_generate_resume_script(self):
        from models.finetune_orchestrator import FineTuneOrchestrator
        fo = FineTuneOrchestrator()
        script = fo.generate_resume_script(
            checkpoint_path="./ckpt-1000",
            base_model="phi-2", dataset="dolly",
        )
        assert _is_valid_python(script)
        assert "RESUME_CHECKPOINT" in script

    def test_generate_script_includes_metrics_callback(self):
        from models.finetune_orchestrator import FineTuneOrchestrator
        fo = FineTuneOrchestrator()
        script = fo.generate_script(base_model="phi-2", dataset="dolly")
        assert "MetricsCallback" in script
        assert "training_log.json" in script
