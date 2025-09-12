import subprocess
from typing import List

from fastapi import HTTPException

from escargot_review_bot.config.config import REPO_PATH
from escargot_review_bot.config.logging import get_logger


logger = get_logger("review-bot.git")


def run_git_command(command: List[str]) -> str:
    """Run a git subcommand in `REPO_PATH` and return its stdout as text.

    Raises HTTPException(500) on failure so API callers receive a clear error.
    """
    try:
        # Log command for traceability
        logger.debug(f"GIT exec: git {' '.join(command)} (cwd={REPO_PATH})")
        out = subprocess.check_output(["git"] + command, cwd=REPO_PATH, text=True)
        logger.debug(f"GIT ok: len={len(out)}")
        return out
    except subprocess.CalledProcessError as e:
        # Map git failures to HTTP 500 for upstream handlers
        logger.error(f"GIT command failed: git {' '.join(command)} -> {e}")
        raise HTTPException(status_code=500, detail="An internal Git command failed.")