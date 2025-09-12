import asyncio
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from escargot_review_bot.config.config import REVIEW_MAX_CONCURRENCY
from escargot_review_bot.config.logging import get_logger
from escargot_review_bot.domain.schemas import ReviewRequest
from escargot_review_bot.service import generate_review_comments


app = FastAPI(title="Escargot Review Bot API", version="1.0")
logger = get_logger("review-bot.app")

_review_semaphore = asyncio.Semaphore(REVIEW_MAX_CONCURRENCY)
"""Global semaphore to limit concurrent /review requests."""


@app.middleware("http")
async def _queue_review_requests(request, call_next):
    """Gate /review POST requests under a concurrency semaphore.

    Non-/review routes pass through immediately; /review is throttled by
    `REVIEW_MAX_CONCURRENCY` to avoid resource saturation.
    """
    path = request.url.path.rstrip("/") or "/"
    if path == "/review" and request.method.upper() == "POST":
        async with _review_semaphore:
            return await call_next(request)
    return await call_next(request)


@app.post("/review")
async def handle_review_request(request: ReviewRequest) -> JSONResponse:
    """Handle a code review request and return generated comments as JSON."""
    comments = generate_review_comments(request)
    return JSONResponse(content={"comments": comments})