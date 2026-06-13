import base64
from config.settings import settings


class GitHubIntegration:
    def __init__(self, token: str = None, repo: str = None):
        self.token = token or settings.github_token
        self.repo = repo or settings.github_repo
        self.client = None

    def _get_client(self):
        if self.client is None and self.token:
            from github import Github, Auth
            self.client = Github(auth=Auth.Token(self.token))
        return self.client

    def push_error_code(self, error_info: dict, job_id: str = None) -> dict:
        """Push error context and code to a GitHub gist or repo."""
        if not self._get_client():
            return {"success": False, "error": "GitHub not configured. Set COLAB_AGENT_GITHUB_TOKEN."}

        try:
            user = self._get_client().get_user()
            content = f"""# Colab Agent Error Report
# Job ID: {job_id or 'N/A'}
# Timestamp: {__import__('datetime').datetime.now().isoformat()}

## Error
{error_info.get('error', 'Unknown error')}

## Context
Runtime: {error_info.get('runtime', 'N/A')}
GPU: {error_info.get('gpu', 'N/A')}
Model: {error_info.get('model', 'N/A')}

## Traceback
```
{error_info.get('traceback', 'N/A')}
```

## Code Context
```python
{error_info.get('code_snippet', 'N/A')}
```
"""
            gist = user.create_gist(
                public=False,
                files={f"colab_agent_error_{job_id or 'unknown'}.md": {"content": content}},
                description=f"Colab Agent Error: {error_info.get('error', 'Unknown')[:100]}"
            )
            return {"success": True, "gist_url": gist.html_url, "gist_id": gist.id}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def push_code_to_repo(self, file_path: str, content: str, commit_msg: str = None,
                          branch: str = "main") -> dict:
        """Push code to the configured GitHub repo."""
        if not self._get_client() or not self.repo:
            return {"success": False, "error": "GitHub repo not configured."}

        try:
            repo = self._get_client().get_repo(self.repo)
            commit_msg = commit_msg or f"Colab Agent: Update {file_path}"

            try:
                contents = repo.get_contents(file_path, ref=branch)
                repo.update_file(
                    path=file_path,
                    message=commit_msg,
                    content=content,
                    sha=contents.sha,
                    branch=branch,
                )
            except Exception:
                repo.create_file(
                    path=file_path,
                    message=commit_msg,
                    content=content,
                    branch=branch,
                )

            raw_url = f"https://raw.githubusercontent.com/{self.repo}/{branch}/{file_path}"
            return {"success": True, "url": raw_url}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def pull_code_from_repo(self, file_path: str, branch: str = "main") -> dict:
        """Pull code from the configured GitHub repo."""
        if not self._get_client() or not self.repo:
            return {"success": False, "error": "GitHub repo not configured."}

        try:
            repo = self._get_client().get_repo(self.repo)
            contents = repo.get_contents(file_path, ref=branch)
            decoded = base64.b64decode(contents.content).decode("utf-8")
            return {"success": True, "content": decoded, "sha": contents.sha}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def list_repo_files(self, path: str = "", branch: str = "main") -> dict:
        if not self._get_client() or not self.repo:
            return {"success": False, "error": "GitHub repo not configured."}
        try:
            repo = self._get_client().get_repo(self.repo)
            contents = repo.get_contents(path, ref=branch)
            files = [{"name": c.name, "type": c.type, "path": c.path} for c in contents]
            return {"success": True, "files": files}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def code_to_push_colab_script(self, file_path: str, branch: str = "main") -> str:
        """Generate Colab code to push results to GitHub."""
        return f"""
# Push to GitHub
from github import Github
import base64

g = Github("{self.token or 'YOUR_TOKEN'}")
repo = g.get_repo("{self.repo or 'username/repo'}")

# Read file
with open("{file_path}", "r") as f:
    content = f.read()

try:
    contents = repo.get_contents("{file_path}", ref="{branch}")
    repo.update_file("{file_path}", "Update from Colab", content, contents.sha)
    print(f"Updated: {{contents.html_url}}")
except:
    repo.create_file("{file_path}", "Create from Colab", content)
    print(f"Created: {file_path}")
"""

    def code_to_pull_colab_script(self, file_path: str, branch: str = "main",
                                  output_path: str = None) -> str:
        """Generate Colab code to pull code from GitHub."""
        output = output_path or file_path.split("/")[-1]
        return f"""
# Pull from GitHub
from github import Github
import base64

g = Github("{self.token or 'YOUR_TOKEN'}")
repo = g.get_repo("{self.repo or 'username/repo'}")

contents = repo.get_contents("{file_path}", ref="{branch}")
decoded = base64.b64decode(contents.content).decode("utf-8")

with open("{output}", "w") as f:
    f.write(decoded)
print(f"Pulled {file_path} -> {output}")
print(f"Content length: {{len(decoded)}} chars")
"""
