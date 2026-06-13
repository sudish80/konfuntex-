"""Tests for config/settings.py — env profiles, validation, defaults."""
import pytest


class TestSettingsDefaults:
    def test_env_defaults_to_dev(self, monkeypatch):
        monkeypatch.delenv("COLAB_AGENT_ENV", raising=False)
        from config.settings import Settings
        s = Settings()
        assert s.env == "dev"

    def test_data_dir_default(self, monkeypatch):
        monkeypatch.delenv("COLAB_AGENT_DATA_DIR", raising=False)
        from config.settings import Settings
        s = Settings()
        assert "colab-agent" in s.data_dir

    def test_db_url_format(self, monkeypatch):
        monkeypatch.delenv("COLAB_AGENT_DB_URL", raising=False)
        from config.settings import Settings
        s = Settings()
        url = s.get_db_url()
        assert url.startswith("sqlite")

    def test_budget_defaults(self, monkeypatch):
        monkeypatch.delenv("COLAB_AGENT_BUDGET_MAX_UNITS", raising=False)
        monkeypatch.delenv("COLAB_AGENT_BUDGET_WARN_THRESHOLD", raising=False)
        from config.settings import Settings
        s = Settings()
        assert s.budget_max_units == 100.0
        assert s.budget_warn_threshold == 0.8


class TestSettingsValidation:
    def test_invalid_env_raises(self):
        from config.settings import Settings
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            Settings(env="invalid")

    def test_invalid_temperature_raises(self):
        from config.settings import Settings
        with pytest.raises(ValueError):
            Settings(agent_temperature=5.0)

    def test_negative_budget_raises(self):
        from config.settings import Settings
        with pytest.raises(ValueError):
            Settings(budget_max_units=-1)

    def test_prod_requires_api_key(self):
        with pytest.raises(ValueError, match="API_KEY is required"):
            from config.settings import Settings
            Settings(env="prod", db_url="postgresql://localhost/db")

    def test_prod_requires_postgres(self):
        with pytest.raises(ValueError, match="PostgreSQL"):
            from config.settings import Settings
            Settings(env="prod", api_key="test-key")

    def test_prod_valid(self):
        from config.settings import Settings
        s = Settings(env="prod", api_key="test-key", db_url="postgresql://localhost/db")
        assert s.env == "prod"
        assert s.api_key == "test-key"

    def test_profile_config_dev(self):
        from config.settings import Settings
        s = Settings(env="dev")
        cfg = s.profile_config()
        assert cfg["agent_verbose"] is True

    def test_profile_config_prod(self):
        from config.settings import Settings
        s = Settings(env="prod", api_key="k", db_url="postgresql://localhost/db")
        cfg = s.profile_config()
        assert cfg["agent_verbose"] is False


class TestSettingsLoad:
    def test_load_settings_singleton(self, monkeypatch):
        monkeypatch.delenv("COLAB_AGENT_ENV", raising=False)
        from config.settings import load_settings
        s1 = load_settings()
        s2 = load_settings()
        assert s1 is s2

    def test_load_settings_exits_on_error(self, monkeypatch):
        monkeypatch.setenv("COLAB_AGENT_ENV", "nope")
        import config.settings as cs
        if hasattr(cs, "_settings"):
            cs._settings = None
        # Should print error and exit
        with pytest.raises(SystemExit):
            cs.load_settings()
