"""
Git utilities.
Wrapper around git command line tools to retrieve history.
"""
import subprocess
import os
from datetime import datetime

class GitError(Exception):
    pass


def _build_candidate_paths(file_path_rel: str) -> list[str]:
    """Generate likely git path variants from DB values."""
    raw = (file_path_rel or "").strip()
    if not raw:
        return []

    normalized = raw.replace("\\", "/").strip("/")
    parts = [p for p in normalized.split("/") if p]
    base = parts[-1] if parts else normalized

    candidates: list[str] = []

    def add(path: str) -> None:
        p = path.strip("/")
        if p and p not in candidates:
            candidates.append(p)

    add(normalized)

    # Common mismatch: DB path misses extension while repo stores .sql files.
    if "." not in base:
        add(f"{normalized}.sql")

    # Fallbacks that ignore leading folders that may differ from repo layout.
    add(base)
    if "." not in base:
        add(f"{base}.sql")

    # Try trimmed tails of the original path.
    for i in range(1, len(parts)):
        tail = "/".join(parts[i:])
        add(tail)
        tail_base = parts[-1]
        if "." not in tail_base and not tail.endswith(".sql"):
            add(f"{tail}.sql")

    return candidates

def get_git_history(
    repo_path: str,
    file_path_rel: str,
    limit: int = 5,
    before_datetime: datetime | str | None = None,
) -> list[dict]:
    """
    Get the last N commits for a specific file in the repository.

    Args:
        repo_path: Absolute path to the git repository root
        file_path_rel: Relative path to the file within the repository
        limit: Number of commits to retrieve
        before_datetime: Only return commits at or before this timestamp.

    Returns:
        List of dictionaries with commit details:
        [{
            'hash': 'Abc1234',
            'date': '2024-01-01',
            'author': 'Jane Doe',
            'message': 'Fix bug'
        }]
    """
    if not os.path.exists(repo_path):
        raise GitError(f"Repository path not found: {repo_path}")

    candidate_files = _build_candidate_paths(file_path_rel)
    if not candidate_files:
        return []
    
    # Construct command
    # git log -n 5 --date=short --pretty=format:"%h|%ad|%an|%s" -- path/to/file
    try:
        base_cmd = [
            "git",
            "-c",
            f"safe.directory={repo_path}",
            "-C", repo_path,
            "log",
            "--follow",
            "-n",
            str(limit),
            "--date=short",
            "--pretty=format:%h|%ad|%an|%s",
        ]

        if before_datetime is not None:
            if isinstance(before_datetime, datetime):
                until_value = before_datetime.strftime("%Y-%m-%d %H:%M:%S")
            else:
                until_value = str(before_datetime)
            base_cmd.extend(["--until", until_value])

        for target_file in candidate_files:
            cmd = [*base_cmd, "--", target_file]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                check=False
            )

            if result.returncode != 0:
                err_msg = result.stderr.strip()
                if "did not match any file" in err_msg or "unknown revision" in err_msg:
                    continue
                raise GitError(f"Git command failed: {err_msg}")

            commits = []
            for line in result.stdout.splitlines():
                if not line.strip():
                    continue
                parts = line.split("|", 3)
                if len(parts) == 4:
                    commits.append({
                        "hash": parts[0],
                        "date": parts[1],
                        "author": parts[2],
                        "message": parts[3]
                    })

            if commits:
                return commits

        return []

    except FileNotFoundError:
        raise GitError("Git executable not found. Please ensure git is installed and in PATH.")
    except Exception as e:
        raise GitError(f"Unexpected error retrieving git history: {str(e)}")
