"""Verify all prompt templates are valid strings and contain expected content."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agent.prompts import (
    SYSTEM_PROMPT, PLANNING_PROMPT, CODE_GENERATION_PROMPT,
    RESULT_PARSING_PROMPT, ERROR_ANALYSIS_PROMPT, SUMMARY_PROMPT,
)


def _check_format(prompt: str, name: str, expected_keywords: list[str]):
    assert prompt, f"{name}: empty"
    assert isinstance(prompt, str), f"{name}: not a string"
    for kw in expected_keywords:
        assert kw in prompt, f"{name}: missing keyword '{kw}'"


class TestSystemPrompt:
    def test_non_empty(self):
        assert SYSTEM_PROMPT
        assert len(SYSTEM_PROMPT) > 200

    def test_contains_key_concepts(self):
        _check_format(SYSTEM_PROMPT, "SYSTEM_PROMPT", [
            "Google Colab", "HuggingFace", "PLAN", "EXECUTE", "DECIDE",
            "LoRA", "QLoRA", "A100",
        ])


class TestPlanningPrompt:
    def test_non_empty(self):
        assert PLANNING_PROMPT
        assert len(PLANNING_PROMPT) > 300

    def test_has_goal_placeholder(self):
        assert "{goal}" in PLANNING_PROMPT

    def test_has_json_schema(self):
        assert "JSON" in PLANNING_PROMPT
        assert "steps" in PLANNING_PROMPT
        assert "model" in PLANNING_PROMPT

    def test_valid_f_string(self):
        filled = PLANNING_PROMPT.format(goal="test goal", model_hint="", dataset_hint="", method_hint="")
        assert "test goal" in filled

    def test_escaped_braces(self):
        assert "{{" in PLANNING_PROMPT or "}}" in PLANNING_PROMPT


class TestCodeGenerationPrompt:
    def test_non_empty(self):
        assert CODE_GENERATION_PROMPT
        assert len(CODE_GENERATION_PROMPT) > 200

    def test_has_placeholders(self):
        for ph in ["{goal}", "{step_id}", "{step_description}", "{action}"]:
            assert ph in CODE_GENERATION_PROMPT, f"missing {ph}"

    def test_valid_f_string(self):
        filled = CODE_GENERATION_PROMPT.format(
            goal="test", step_id=1, step_description="desc", action="run"
        )
        assert "test" in filled
        assert "1" in filled

    def test_keywords(self):
        _check_format(CODE_GENERATION_PROMPT, "CODE_GENERATION_PROMPT", [
            "Colab", "pip install", "device_map", "BitsAndBytesConfig",
            "torch.cuda.empty_cache",
        ])


class TestResultParsingPrompt:
    def test_non_empty(self):
        assert RESULT_PARSING_PROMPT
        assert len(RESULT_PARSING_PROMPT) > 200

    def test_has_placeholders(self):
        assert "{goal}" in RESULT_PARSING_PROMPT
        assert "{output}" in RESULT_PARSING_PROMPT

    def test_json_schemas_present(self):
        assert "status" in RESULT_PARSING_PROMPT
        assert "next_action" in RESULT_PARSING_PROMPT
        assert "retry" in RESULT_PARSING_PROMPT
        assert "switch_runtime" in RESULT_PARSING_PROMPT
        assert "change_model" in RESULT_PARSING_PROMPT
        assert "change_dataset" in RESULT_PARSING_PROMPT
        assert "abort" in RESULT_PARSING_PROMPT

    def test_error_types(self):
        for t in ["runtime_oom", "syntax_error", "import_error", "api_error", "training_diverged", "wrong_model", "wrong_dataset"]:
            assert t in RESULT_PARSING_PROMPT


class TestErrorAnalysisPrompt:
    def test_non_empty(self):
        assert ERROR_ANALYSIS_PROMPT
        assert len(ERROR_ANALYSIS_PROMPT) > 100

    def test_has_placeholders(self):
        for ph in ["{error}", "{runtime}", "{context}", "{attempts}"]:
            assert ph in ERROR_ANALYSIS_PROMPT, f"missing {ph}"


class TestSummaryPrompt:
    def test_non_empty(self):
        assert SUMMARY_PROMPT
        assert len(SUMMARY_PROMPT) > 100

    def test_has_placeholders(self):
        for ph in ["{goal}", "{plan}", "{results}", "{final_status}"]:
            assert ph in SUMMARY_PROMPT, f"missing {ph}"

    def test_keywords(self):
        _check_format(SUMMARY_PROMPT, "SUMMARY_PROMPT", [
            "summary", "metrics", "model", "dataset",
        ])


class TestAllPrompts:
    def test_no_prompt_is_none(self):
        prompts = [
            ("SYSTEM", SYSTEM_PROMPT),
            ("PLANNING", PLANNING_PROMPT),
            ("CODE_GENERATION", CODE_GENERATION_PROMPT),
            ("RESULT_PARSING", RESULT_PARSING_PROMPT),
            ("ERROR_ANALYSIS", ERROR_ANALYSIS_PROMPT),
            ("SUMMARY", SUMMARY_PROMPT),
        ]
        for name, prompt in prompts:
            assert prompt is not None, f"{name} is None"
            assert len(prompt) >= 50, f"{name} too short ({len(prompt)} chars)"
