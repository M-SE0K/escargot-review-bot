import os
import runpy
import sys
from pathlib import Path

# src 디렉터리를 Python 경로에 추가하여 패키지의 실제 진입점(escargot_review_bot.main)으로 위임
root = Path(__file__).resolve().parent
src_path = str(root / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Delegate to package entry (keeps single source of truth)
if __name__ == "__main__":
    runpy.run_module("escargot_review_bot.main", run_name="__main__")