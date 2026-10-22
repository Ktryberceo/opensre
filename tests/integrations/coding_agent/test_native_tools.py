"""Isolation contract of the built-in coding agent's tools: the checkout is the boundary."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import pytest

from integrations.coding_agent.native_tools import WorkspaceTools, refused_git_verb


def _checkout(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "app.py").write_text("value = 1\n")
    return repo


def test_tools_refuse_paths_outside_the_checkout_and_git_history_changes(tmp_path: Path) -> None:
    repo = _checkout(tmp_path)
    (tmp_path / "secret.txt").write_text("token")
    (repo / "link").symlink_to(tmp_path / "secret.txt")
    tools = WorkspaceTools(str(repo), deadline=time.monotonic() + 60)

    assert tools.read_file("../secret.txt")["ok"] is False
    assert tools.read_file(str(tmp_path / "secret.txt"))["ok"] is False
    assert tools.read_file("link")["ok"] is False
    assert tools.write_file("../escape.txt", "x")["ok"] is False
    assert tools.write_file(".git/hooks/pre-commit", "x")["ok"] is False
    assert not (tmp_path / "escape.txt").exists()

    assert tools.edit_file("app.py", "value = 1", "value = 2")["ok"] is False
    assert tools.read_file("app.py")["ok"] is True
    assert tools.edit_file("app.py", "value = 1", "value = 2")["ok"] is True
    assert (repo / "app.py").read_text() == "value = 2\n"

    pushed = tools.bash("git -C . push origin main")
    assert pushed["ok"] is False
    assert "push" in pushed["error"]


@pytest.mark.parametrize(
    ("command", "verb"),
    [
        ("git commit -am fix", "commit"),
        ("make lint && git push", "push"),
        ("sh -c 'git reset --hard'", "reset"),
        ("/usr/bin/git -c user.name=x checkout -- app.py", "checkout"),
        ("git status && git diff", None),
        ("ls .git/hooks && cat .gitignore", None),
    ],
)
def test_refused_git_verb(command: str, verb: str | None) -> None:
    assert refused_git_verb(command) == verb
