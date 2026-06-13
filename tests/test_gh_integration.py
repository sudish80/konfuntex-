"""Tests for gh_integration/ — GitHubIntegration and GitHubLogger with mocks."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ["COLAB_AGENT_GITHUB_TOKEN"] = "ghp_test_mock_token_12345"
os.environ["COLAB_AGENT_GITHUB_REPO"] = "test-user/test-repo"

from unittest.mock import MagicMock, patch
from gh_integration.integration import GitHubIntegration
from gh_integration.logger import GitHubLogger


# ------------------------------------------------------------------ #
#  GitHubIntegration tests
# ------------------------------------------------------------------ #

class TestGitHubIntegrationInit:
    def test_from_env(self):
        gi = GitHubIntegration()
        assert gi.token == "ghp_test_mock_token_12345"
        assert gi.repo == "test-user/test-repo"

    def test_explicit_values(self):
        gi = GitHubIntegration(token="abc", repo="my/repo")
        assert gi.token == "abc"
        assert gi.repo == "my/repo"

    def test_client_lazy_init(self):
        gi = GitHubIntegration(token="abc")
        assert gi.client is None
        gi._get_client()
        assert gi.client is not None


class TestGitHubIntegrationPushError:
    @patch("gh_integration.integration.settings")
    def test_no_token(self, mock_settings):
        mock_settings.github_token = None
        mock_settings.github_repo = None
        gi = GitHubIntegration()
        result = gi.push_error_code({"error": "test"})
        assert result["success"] is False
        assert "configured" in result["error"].lower()

    @patch("github.Github")
    def test_success(self, mock_github):
        mock_user = MagicMock()
        mock_gist = MagicMock()
        mock_gist.html_url = "https://gist.github.com/abc123"
        mock_gist.id = "abc123"
        mock_user.create_gist.return_value = mock_gist
        mock_github.return_value.get_user.return_value = mock_user

        gi = GitHubIntegration()
        result = gi.push_error_code(
            {"error": "CUDA OOM", "runtime": "T4", "model": "phi-2"},
            job_id="job-1",
        )
        assert result["success"] is True
        assert result["gist_url"] == "https://gist.github.com/abc123"
        mock_user.create_gist.assert_called_once()

    @patch("github.Github")
    def test_exception_handling(self, mock_github):
        mock_github.return_value.get_user.side_effect = Exception("API error")
        gi = GitHubIntegration()
        result = gi.push_error_code({"error": "test"})
        assert result["success"] is False
        assert "API error" in result["error"]


class TestGitHubIntegrationRepoOps:
    @patch("gh_integration.integration.settings")
    def test_push_code_no_repo(self, mock_settings):
        mock_settings.github_token = "abc"
        mock_settings.github_repo = None
        gi = GitHubIntegration()
        result = gi.push_code_to_repo("test.py", "content")
        assert result["success"] is False

    @patch("github.Github")
    def test_push_code_create(self, mock_github):
        mock_repo = MagicMock()
        mock_repo.get_contents.side_effect = Exception("not found")
        mock_github.return_value.get_repo.return_value = mock_repo

        gi = GitHubIntegration()
        result = gi.push_code_to_repo("test.py", "print('hello')", commit_msg="test")
        assert result["success"] is True
        mock_repo.create_file.assert_called_once()

    @patch("github.Github")
    def test_push_code_update(self, mock_github):
        mock_content = MagicMock()
        mock_content.sha = "abc123"
        mock_repo = MagicMock()
        mock_repo.get_contents.return_value = mock_content
        mock_github.return_value.get_repo.return_value = mock_repo

        gi = GitHubIntegration()
        result = gi.push_code_to_repo("test.py", "new content")
        assert result["success"] is True
        mock_repo.update_file.assert_called_once()

    @patch("github.Github")
    def test_push_code_exception(self, mock_github):
        mock_repo = MagicMock()
        mock_repo.get_contents.side_effect = Exception("not found")
        mock_repo.create_file.side_effect = Exception("create failed")
        mock_github.return_value.get_repo.return_value = mock_repo

        gi = GitHubIntegration()
        result = gi.push_code_to_repo("test.py", "content")
        assert result["success"] is False

    @patch("github.Github")
    def test_pull_code(self, mock_github):
        import base64
        mock_content = MagicMock()
        mock_content.content = base64.b64encode(b"print('hello')").decode()
        mock_content.sha = "sha123"
        mock_repo = MagicMock()
        mock_repo.get_contents.return_value = mock_content
        mock_github.return_value.get_repo.return_value = mock_repo

        gi = GitHubIntegration()
        result = gi.pull_code_from_repo("test.py")
        assert result["success"] is True
        assert result["content"] == "print('hello')"

    @patch("github.Github")
    def test_pull_code_no_repo(self, mock_github):
        gi = GitHubIntegration(token="abc", repo=None)
        result = gi.pull_code_from_repo("test.py")
        assert result["success"] is False

    @patch("github.Github")
    def test_list_files(self, mock_github):
        mock_file = MagicMock()
        mock_file.name = "test.py"
        mock_file.type = "file"
        mock_file.path = "test.py"
        mock_repo = MagicMock()
        mock_repo.get_contents.return_value = [mock_file]
        mock_github.return_value.get_repo.return_value = mock_repo

        gi = GitHubIntegration()
        result = gi.list_repo_files()
        assert result["success"] is True
        assert len(result["files"]) == 1

    @patch("gh_integration.integration.settings")
    def test_list_files_no_repo(self, mock_settings):
        mock_settings.github_token = "abc"
        mock_settings.github_repo = None
        gi = GitHubIntegration()
        result = gi.list_repo_files()
        assert result["success"] is False


class TestGitHubIntegrationCodeGen:
    def test_code_to_push_has_expected_content(self):
        gi = GitHubIntegration(token="tok", repo="user/repo")
        code = gi.code_to_push_colab_script("output/model.pt")
        assert "from github import Github" in code
        assert "tok" in code
        assert "user/repo" in code
        assert "output/model.pt" in code

    def test_code_to_pull_has_expected_content(self):
        gi = GitHubIntegration(token="tok", repo="user/repo")
        code = gi.code_to_pull_colab_script("remote/path.py", output_path="local.py")
        assert "from github import Github" in code
        assert "tok" in code
        assert "local.py" in code


# ------------------------------------------------------------------ #
#  GitHubLogger tests
# ------------------------------------------------------------------ #

class TestGitHubLoggerInit:
    def test_from_env(self):
        gl = GitHubLogger()
        assert gl.token == "ghp_test_mock_token_12345"
        assert gl.repo == "test-user/test-repo"

    def test_explicit(self):
        gl = GitHubLogger(token="abc", repo="x/y")
        assert gl.token == "abc"

    def test_client_lazy(self):
        gl = GitHubLogger(token="abc")
        assert gl._client is None
        gl._get_client()
        assert gl._client is not None


class TestGitHubLoggerPushError:
    @patch("gh_integration.logger.settings")
    def test_no_token(self, mock_settings):
        mock_settings.github_token = None
        mock_settings.github_repo = None
        gl = GitHubLogger()
        result = gl.push_error("error log", job_id="job-1")
        assert result["success"] is False
        assert "configured" in result["error"].lower()

    @patch("github.Github")
    def test_success(self, mock_github):
        mock_repo = MagicMock()
        mock_result = MagicMock()
        mock_result.sha = "sha123"
        mock_result.html_url = "https://github.com/file"
        mock_repo.create_file.return_value = {"content": mock_result}
        mock_repo.get_contents.side_effect = Exception("not found")
        mock_github.return_value.get_repo.return_value = mock_repo

        gl = GitHubLogger()
        result = gl.push_error("CUDA OOM", notebook_cell="train()", job_id="job-1")
        assert result["success"] is True

    @patch("github.Github")
    def test_with_metadata(self, mock_github):
        mock_repo = MagicMock()
        mock_result = MagicMock()
        mock_result.sha = "sha456"
        mock_result.html_url = "https://github.com/file"
        mock_repo.create_file.return_value = {"content": mock_result}
        mock_repo.get_contents.side_effect = Exception("not found")
        mock_github.return_value.get_repo.return_value = mock_repo

        gl = GitHubLogger()
        result = gl.push_error("err", metadata={"gpu": "T4", "vram": 16})
        assert result["success"] is True


class TestGitHubLoggerRetrieveSimilar:
    @patch("github.Github")
    def test_no_token_returns_empty(self, mock_github):
        gl = GitHubLogger(token=None, repo=None)
        assert gl.retrieve_similar_errors("OOM error") == []

    @patch("github.Github")
    def test_returns_matches(self, mock_github):
        mock_file = MagicMock()
        mock_file.type = "file"
        mock_file.path = "errors/job-1/error.md"
        mock_file.sha = "abc"
        mock_file.html_url = "https://url"
        mock_file.decoded_content = b"CUDA out of memory on T4 GPU"

        mock_repo = MagicMock()
        mock_repo.get_contents.return_value = [mock_file]
        mock_github.return_value.get_repo.return_value = mock_repo

        gl = GitHubLogger()
        results = gl.retrieve_similar_errors("CUDA OOM on T4", max_results=5)
        assert len(results) >= 1
        assert results[0]["score"] > 0

    @patch("github.Github")
    def test_empty_when_no_match(self, mock_github):
        mock_file = MagicMock()
        mock_file.type = "file"
        mock_file.decoded_content = b"zzzzz qqqqq wwwww vvvvv"
        mock_repo = MagicMock()
        mock_repo.get_contents.return_value = [mock_file]
        mock_github.return_value.get_repo.return_value = mock_repo

        gl = GitHubLogger()
        results = gl.retrieve_similar_errors("aaaa bbbb cccc dddd", max_results=5)
        assert len(results) == 0

    @patch("github.Github")
    def test_handles_dir_contents(self, mock_github):
        mock_dir = MagicMock()
        mock_dir.type = "dir"
        mock_dir.path = "errors/job-1"

        mock_subfile = MagicMock()
        mock_subfile.type = "file"
        mock_subfile.decoded_content = b"CUDA error occurred"
        mock_subfile.path = "errors/job-1/error.md"
        mock_subfile.sha = "abc"
        mock_subfile.html_url = "https://url"

        mock_repo = MagicMock()
        mock_repo.get_contents.side_effect = [
            [mock_dir],
            [mock_subfile],
        ]
        mock_github.return_value.get_repo.return_value = mock_repo

        gl = GitHubLogger()
        results = gl.retrieve_similar_errors("CUDA error")
        assert len(results) >= 1

    @patch("github.Github")
    def test_fallback_on_prefix_not_found(self, mock_github):
        mock_repo = MagicMock()
        mock_repo.get_contents.side_effect = [Exception("not found"), Exception("not found")]
        mock_github.return_value.get_repo.return_value = mock_repo

        gl = GitHubLogger()
        results = gl.retrieve_similar_errors("OOM", job_id="job-1")
        assert results == []


class TestGitHubLoggerPushReport:
    @patch("github.Github")
    def test_push_final_report(self, mock_github):
        mock_repo = MagicMock()
        mock_result = MagicMock()
        mock_result.sha = "sha"
        mock_result.html_url = "https://url"
        mock_repo.create_file.return_value = {"content": mock_result}
        mock_repo.get_contents.side_effect = Exception("not found")
        mock_github.return_value.get_repo.return_value = mock_repo

        gl = GitHubLogger()
        result = gl.push_final_report("job-1", metrics={"loss": 0.5}, summary="done")
        assert result["success"] is True

    @patch("github.Github")
    def test_push_metrics_json(self, mock_github):
        mock_repo = MagicMock()
        mock_result = MagicMock()
        mock_result.sha = "sha"
        mock_result.html_url = "https://url"
        mock_repo.create_file.return_value = {"content": mock_result}
        mock_repo.get_contents.side_effect = Exception("not found")
        mock_github.return_value.get_repo.return_value = mock_repo

        gl = GitHubLogger()
        result = gl.push_metrics_json("job-1", [{"loss": 0.5}, {"loss": 0.3}])
        assert result["success"] is True

    @patch("github.Github")
    def test_create_file_exists(self, mock_github):
        mock_existing = MagicMock()
        mock_existing.sha = "existing_sha"
        mock_existing.html_url = "https://existing"
        mock_repo = MagicMock()
        mock_repo.get_contents.return_value = mock_existing
        mock_github.return_value.get_repo.return_value = mock_repo

        gl = GitHubLogger()
        result = gl.push_final_report("existing-job")
        assert result["success"] is True
        assert result["sha"] == "existing_sha"

    @patch("github.Github")
    def test_create_file_exception(self, mock_github_class):
        mock_github_instance = MagicMock()
        mock_repo = MagicMock()
        mock_repo.get_contents.side_effect = Exception("not found")
        mock_repo.create_file.side_effect = Exception("create failed too")
        mock_github_instance.get_repo.return_value = mock_repo
        mock_github_class.return_value = mock_github_instance

        gl = GitHubLogger()
        result = gl.push_final_report("broken-job")
        assert result["success"] is False

    @patch("github.Github")
    def test_generate_error_code(self, mock_github):
        gl = GitHubLogger()
        code = gl.generate_error_code("job-42")
        assert "Github" in code
        assert "job-42" in code
        assert "push" in code.lower()
