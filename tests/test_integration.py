"""Integration tests — now pass without live APIs via mocks + env defaults from conftest.py."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_01_llm_client(mock_llm_client):
    """Test LLM client returns a response (mocked)."""
    from agent.llm_client import LLMClient
    client = LLMClient()
    assert client.provider == "openai"
    assert client.model is not None and len(client.model) > 0
    resp = client.chat([{"role": "user", "content": "Say hello in one word."}])
    content = resp.get("content", "")
    assert content, f"Empty response: {resp}"
    assert "Error" not in content


def test_02_llm_json(mock_llm_client):
    """Test LLM can return structured JSON (mocked)."""
    from agent.llm_client import LLMClient
    client = LLMClient()
    result = client.safe_json_chat([
        {"role": "system", "content": "Return JSON only."},
        {"role": "user", "content": 'Return {"status": "ok", "value": 42}'},
    ])
    assert result is not None
    assert result.get("status") == "ok"
    assert result.get("value") == 42


def test_03_agent_basic(mock_llm_client):
    """Test agent runs a simple simulation plan (mocked LLM)."""
    from agent.core import OrchestratorAgent
    from storage.database import init_db
    init_db()
    agent = OrchestratorAgent()
    result = agent.run("List 3 Python ML libraries")
    assert result is not None
    assert result.get("status") in ("completed", "partial", "aborted")


def test_04_github_token():
    """Test GitHub token is loadable (from conftest env default)."""
    from config.settings import settings
    assert settings.github_token is not None
    assert settings.github_token.startswith("ghp_")


def test_05_hf_token():
    """Test HF token is loadable (from conftest env default)."""
    from config.settings import settings
    assert settings.hf_token is not None
    assert settings.hf_token.startswith("hf_")


def test_06_config():
    """Test settings loaded correctly (from conftest env defaults)."""
    from config.settings import settings
    assert settings.openai_base_url == "https://integrate.api.nvidia.com/v1"
    assert settings.llm_model is not None and len(settings.llm_model) > 0
    assert settings.llm_provider == "openai"


def test_07_memory_store():
    """Test memory store works end-to-end (no external deps)."""
    from agent.memory import MemoryStore
    ms = MemoryStore(max_turns=5)
    ms.add("user", "Hello")
    ms.add("assistant", "Hi there!")
    assert len(ms.get_context()) == 2
    ms.add("user", "How are you?")
    ms.add("assistant", "I'm good!")
    ms.add("user", "What's up?")
    ms.add("assistant", "Not much!")
    ms.add("user", "Too many messages now")
    assert len(ms.get_context()) <= 5
