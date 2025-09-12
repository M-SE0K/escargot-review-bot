import os
import runpy
import sys
from pathlib import Path

# Ensure src/ is importable when running via `python main.py`
root = Path(__file__).resolve().parent
src_path = str(root / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Delegate to package entry (keeps single source of truth)
if __name__ == "__main__":
    runpy.run_module("escargot_review_bot.main", run_name="__main__")