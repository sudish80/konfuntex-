"""Shared fixtures and configuration for all tests."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ.setdefault("COLAB_AGENT_DB_URL", "sqlite://")
os.environ.setdefault("COLAB_AGENT_DATA_DIR", "/tmp")
os.environ.setdefault("COLAB_AGENT_GITHUB_TOKEN", "ghp_test_mock_token_12345")
os.environ.setdefault("COLAB_AGENT_GITHUB_REPO", "test-user/test-repo")
os.environ.setdefault("COLAB_AGENT_HF_TOKEN", "hf_test_mock_token_12345")
os.environ.setdefault("COLAB_AGENT_OPENAI_BASE_URL", "https://integrate.api.nvidia.com/v1")
os.environ.setdefault("COLAB_AGENT_LLM_MODEL", "test-model")
os.environ.setdefault("COLAB_AGENT_LLM_PROVIDER", "openai")
os.environ.setdefault("COLAB_AGENT_OPENAI_API_KEY", "sk-test-key")
os.environ.setdefault("COLAB_AGENT_ENV", "dev")
os.environ.setdefault("COLAB_AGENT_BUDGET_MAX_UNITS", "999999")
os.environ.setdefault("COLAB_AGENT_BUDGET_WARN_THRESHOLD", "0.9")

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "live_api: marks tests that need a real LLM API key (skipped in CI)")


@pytest.fixture
def mock_llm_client(monkeypatch):
    """Replace LLMClient with a mock returning canned responses."""
    import agent.llm_client as llm

    class MockLLMClient:
        provider = "openai"
        model = "test-model"
        temperature = 0.0

        def chat(self, messages, tools=None, tool_choice=None):
            msgs_text = str(messages)
            if "Return ONLY Python code" in msgs_text:
                return {"role": "assistant", "content": "print('Mock execution successful')"}
            if "Summarize the fine-tuning" in msgs_text:
                return {"role": "assistant", "content": "Mock summary"}
            return {"role": "assistant", "content": "Mock response"}

        def safe_json_chat(self, messages, parser="auto"):
            msgs_text = str(messages)
            if "Analyze the user's goal" in msgs_text:
                return {
                    "goal": "List 3 Python ML libraries",
                    "analysis": {},
                    "steps": [
                        {"id": 1, "action": "setup_environment", "description": "Install Python ML libraries", "expected_duration": "1 minute"},
                        {"id": 2, "action": "execute", "description": "List libraries", "expected_duration": "1 minute"},
                    ],
                }
            if "You are analyzing the output" in msgs_text:
                return {"status": "success", "summary": "Mock success", "key_values": {"result": "ok"}, "next_action": "proceed"}
            return {"status": "ok", "value": 42}

        def extract_json_from_response(self, content):
            return None

        def chat_stream(self, messages, on_chunk=None):
            if on_chunk:
                on_chunk("Mock stream")
            return "Mock stream"

    monkeypatch.setattr(llm, "LLMClient", MockLLMClient)
    # Also patch modules that already imported LLMClient
    import agent.core as core_mod
    monkeypatch.setattr(core_mod, "LLMClient", MockLLMClient)


@pytest.fixture
def isolated_db():
    """Provide an isolated in-memory SQLite database per test."""
    from storage.database import reset_session
    reset_session()
    yield
    reset_session()
