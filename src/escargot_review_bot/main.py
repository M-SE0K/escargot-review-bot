import os
import sys

from escargot_review_bot.api import app

if __name__ == "__main__":
    try:
        import uvicorn
    except Exception:
        print("uvicorn is required to run the server. Install dependencies first.")
        sys.exit(1)
    uvicorn.run(app, host="127.0.0.1", port=8000)