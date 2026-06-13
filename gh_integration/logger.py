"""
GitHubLogger — Phase 4
Pushes error logs, retrieves similar past errors, and commits final
reports to a designated GitHub repository.
"""
import re
import json
from datetime import datetime
from typing import Optional
from config.settings import settings


class GitHubLogger:
    """
    Logs agent actions, errors, and final reports to a GitHub repo.

    Repository layout:
        errors/{job_id}/error_{timestamp}.md
        reports/{job_id}/report_{timestamp}.md
        metrics/{job_id}/metrics.json
    """

    def __init__(self, token: Optional[str] = None,
                 repo: Optional[str] = None):
        self.token = token or settings.github_token
        self.repo = repo or settings.github_repo
        self._client = None

    def _get_client(self):
        if self._client is None and self.token:
            from github import Github, Auth
            self._client = Github(auth=Auth.Token(self.token))
        return self._client

    # -------------------------------------------------------------- #
    #  push_error
    # -------------------------------------------------------------- #

    def push_error(self, error_log: str, notebook_cell: str = "",
                   stack_trace: str = "", job_id: Optional[str] = None,
                   metadata: Optional[dict] = None) -> dict:
        """
        Creates a file in errors/{job_id}/error_{timestamp}.md
        containing structured error context.
        """
        if not self._get_client():
            return {"success": False, "error": "GitHub not configured"}

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_id = job_id or "unknown"
        path = f"errors/{safe_id}/error_{ts}.md"

        content = f"""# Error Report
- **Job ID:** {safe_id}
- **Timestamp:** {datetime.now().isoformat()}

## Error Log
```
{error_log}
```

## Notebook Cell
```python
{notebook_cell}
```

## Stack Trace
```
{stack_trace}
```

## Metadata
```json
{json.dumps(metadata or {}, indent=2)}
```
"""
        return self._create_file(path, content,
                                 f"Error report {safe_id} {ts}")

    # -------------------------------------------------------------- #
    #  retrieve_similar_errors
    # -------------------------------------------------------------- #

    def retrieve_similar_errors(self, error_description: str,
                                job_id: Optional[str] = None,
                                max_results: int = 5) -> list[dict]:
        """
        Searches past error reports in the repo for similar
        keywords.  Falls back to simple keyword matching.
        """
        if not self._get_client():
            return []

        repo_obj = self._get_client().get_repo(self.repo)
        prefix = f"errors/{job_id}/" if job_id else "errors/"
        query_words = set(
            w.lower() for w in re.findall(r"\w{4,}", error_description)
        )

        matches = []
        try:
            contents = repo_obj.get_contents(prefix)
        except Exception:
            # Try searching all error dirs
            try:
                contents = repo_obj.get_contents("errors")
            except Exception:
                return []

        for content_file in contents:
            if content_file.type == "dir":
                try:
                    for sub in repo_obj.get_contents(content_file.path):
                        self._score_error_file(sub, query_words, matches)
                except Exception:
                    continue
            else:
                self._score_error_file(content_file, query_words, matches)

        matches.sort(key=lambda x: x["score"], reverse=True)
        return matches[:max_results]

    def _score_error_file(self, file, query_words: set,
                          matches: list):
        try:
            decoded = file.decoded_content.decode("utf-8")
            text_words = set(w.lower() for w in re.findall(r"\w{4,}", decoded))
            overlap = query_words & text_words
            score = len(overlap) / max(len(query_words), 1)
            if score > 0.1:
                matches.append({
                    "path": file.path,
                    "score": round(score, 3),
                    "sha": file.sha,
                    "url": file.html_url,
                    "preview": decoded[:300],
                })
        except Exception:
            pass

    # -------------------------------------------------------------- #
    #  push_final_report
    # -------------------------------------------------------------- #

    def push_final_report(self, job_id: str,
                          metrics: Optional[dict] = None,
                          conversation: Optional[list] = None,
                          summary: str = "") -> dict:
        """
        Commits a full report (metrics + conversation + summary)
        to reports/{job_id}/report_{timestamp}.md.
        """
        if not self._get_client():
            return {"success": False, "error": "GitHub not configured"}

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"reports/{job_id}/report_{ts}.md"

        content = f"""# Fine-Tuning Report
- **Job ID:** {job_id}
- **Timestamp:** {datetime.now().isoformat()}

## Summary
{summary}

## Metrics
```json
{json.dumps(metrics or {}, indent=2)}
```

## Conversation
```json
{json.dumps(conversation or [], indent=2, default=str)}
```
"""
        return self._create_file(path, content,
                                 f"Report {job_id} {ts}")

    # -------------------------------------------------------------- #
    #  push_metrics_json
    # -------------------------------------------------------------- #

    def push_metrics_json(self, job_id: str,
                          metrics_data: list[dict]) -> dict:
        """Push a metrics JSON file to metrics/{job_id}/metrics.json."""
        if not self._get_client():
            return {"success": False, "error": "GitHub not configured"}

        path = f"metrics/{job_id}/metrics.json"
        content = json.dumps(metrics_data, indent=2, default=str)
        return self._create_file(path, content,
                                 f"Metrics {job_id}")

    # -------------------------------------------------------------- #
    #  Internal helpers
    # -------------------------------------------------------------- #

    def _create_file(self, path: str, content: str,
                     commit_msg: str) -> dict:
        try:
            repo_obj = self._get_client().get_repo(self.repo)
            try:
                existing = repo_obj.get_contents(path)
                return {"success": True, "sha": existing.sha,
                        "url": existing.html_url, "updated": True}
            except Exception:
                result = repo_obj.create_file(
                    path=path,
                    message=commit_msg,
                    content=content,
                )
                return {"success": True, "sha": result["content"].sha,
                        "url": result["content"].html_url,
                        "updated": False}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def generate_error_code(self, job_id: str) -> str:
        """Generate Colab code that calls push_error after catching an exception."""
        token = self.token or "YOUR_TOKEN"
        repo = self.repo or "username/repo"
        ts_fmt = datetime.now().strftime("%Y%m%d_%H%M%S")
        return (
            '# === Push error to GitHub ===\n'
            'try:\n'
            '    import traceback, json\n'
            '    from github import Github\n'
            f'    g = Github("{token}")\n'
            f'    repo = g.get_repo("{repo}")\n'
            f'    ts = "{ts_fmt}"\n'
            f'    path = f"errors/{job_id}/error_{{ts}}.md"\n'
            '    err_content = traceback.format_exc()\n'
            f'    content = "# Error\\nJob: {job_id}\\nTime: " + '
            "__import__('datetime').datetime.now().isoformat() + "
            '"\\n```python\\n" + err_content + "\\n```\\n"\n'
            f'    repo.create_file(path=path, message=f"Error {job_id}", content=content)\n'
            '    print(f"Error pushed to {path}")\n'
            'except Exception as e2:\n'
            '    print(f"Could not push error: {e2}")\n'
        )
