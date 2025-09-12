import json
import re
import signal
import time
from typing import Any, Dict, List, Optional

import ollama

from escargot_review_bot.config.config import (
    MODEL_NAME,
    OLLAMA_MAX_RETRIES,
    OLLAMA_NUM_BATCH,
    OLLAMA_NUM_CTX,
    OLLAMA_REPEAT_PENALTY,
    OLLAMA_TEMPERATURE,
    OLLAMA_TIMEOUT_SECONDS,
    INTER_REQUEST_DELAY_SECONDS,
)
from escargot_review_bot.config.logging import get_logger


logger = get_logger("review-bot.llm")


JSON_ARRAY_RE = re.compile(r"\[[\s\S]*?\]")


class OllamaTimeoutError(Exception):
    """Raised when an Ollama API call exceeds the configured timeout."""


def _timeout_handler(signum, frame):
    """Signal handler that raises `OllamaTimeoutError` when alarm fires."""
    raise OllamaTimeoutError(f"Ollama API call timed out after {OLLAMA_TIMEOUT_SECONDS} seconds.")


# Install alarm handler for per-request timeout
signal.signal(signal.SIGALRM, _timeout_handler)


def _find_complete_json_array_span(s: str) -> Optional[tuple]:
    """Find the start/end indices of the first complete JSON array in `s`.

    Tracks string escapes and bracket depth so nested arrays are supported.
    Returns (start, end) inclusive indices, or None if no complete array exists.
    """
    if not s:
        return None
    start = s.find("[")
    if start == -1:
        return None
    in_str = False
    esc = False
    depth = 0
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == '[':
            depth += 1
        elif ch == ']':
            depth -= 1
            if depth == 0:
                return (start, i)
    return None


def sanitize_llm_output(raw: str) -> str:
    """Extract the first valid JSON array from raw text.

    Prefer fenced ```json blocks; otherwise scan inline candidates. Returns the
    JSON array string or an empty string when no valid array is found.
    """
    s = raw or ""

    # 1) Code fence first
    m = re.findall(r"```json\s*([\s\S]*?)\s*```", s, flags=re.IGNORECASE)
    for block in m:
        cand = block.strip()
        try:
            obj = json.loads(cand)
            if isinstance(obj, list):
                logger.debug(f"LLM fenced JSON extracted (len={len(cand)})")
                return cand
        except Exception:
            pass

    # 2) If no fence, scan inline candidates and return the first valid JSON array
    candidates = JSON_ARRAY_RE.findall(s)
    for cand in candidates:
        try:
            obj = json.loads(cand)
            if isinstance(obj, list):
                logger.debug(f"LLM inline JSON array extracted (len={len(cand)})")
                return cand
        except Exception:
            continue

    logger.debug("LLM no JSON array could be extracted from response")
    return ""


def chat_and_parse(system_prompt: str, user_prompt: str) -> List[Dict[str, Any]]:
    """Stream a chat completion and parse a JSON array from the output.

    Stops early when a complete array is detected; otherwise falls back to
    sanitization. Enforces a per-request timeout and retry policy.
    """
    for attempt in range(OLLAMA_MAX_RETRIES):
        try:
            logger.info(f"LLM request start model={MODEL_NAME}")
            logger.debug(f"LLM attempt {attempt + 1}/{OLLAMA_MAX_RETRIES} timeout={OLLAMA_TIMEOUT_SECONDS}s")

            # Arm per-request timeout alarm
            signal.alarm(OLLAMA_TIMEOUT_SECONDS)

            stream = ollama.chat(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                options={
                    "temperature": OLLAMA_TEMPERATURE,
                    "num_ctx": OLLAMA_NUM_CTX,
                    "num_batch": OLLAMA_NUM_BATCH,
                    "repeat_penalty": OLLAMA_REPEAT_PENALTY,
                },
                keep_alive="60m",
                stream=True,
            )

            buf_parts: List[str] = []
            parsed: Optional[List[Dict[str, Any]]] = None
            for chunk in stream:
                # Normalize chunk content across possible dict/object shapes
                content = getattr(getattr(chunk, "message", None), "content", None)
                if content is None and isinstance(chunk, dict):
                    msg = chunk.get("message")
                    if isinstance(msg, dict):
                        content = msg.get("content")
                if not isinstance(content, str) or not content:
                    continue
                # Accumulate partial content and check for a complete JSON array
                buf_parts.append(content)
                text = "".join(buf_parts)
                span = _find_complete_json_array_span(text)
                if span is not None:
                    start, end = span
                    array_text = text[start:end + 1]
                    try:
                        raw_comments = json.loads(array_text)
                        if isinstance(raw_comments, list):
                            parsed = [c for c in raw_comments if isinstance(c, dict)]
                            if parsed and len(parsed) > 0:
                                logger.debug("LLM stream-early-stop: json array complete")
                                logger.debug(f"LLM items={len(parsed)}")
                            else:
                                logger.debug("LLM stream-early-stop: empty JSON array []")
                            # Stop streaming once a valid JSON array is parsed
                            break
                    except Exception:
                        pass

            # Disarm alarm now that streaming finished
            signal.alarm(0)

            if parsed is None:
                if not buf_parts:
                    logger.debug("LLM empty response (no content chunks)")
                    return []
                # Fallback: sanitize accumulated text to extract a valid array
                text = "".join(buf_parts)
                cleaned = sanitize_llm_output(text)
                if not cleaned:
                    logger.debug("LLM sanitize produced empty string; returning []")
                    return []
                raw_comments = json.loads(cleaned)
                if not isinstance(raw_comments, list):
                    logger.debug(f"LLM parsed non-list JSON: type={type(raw_comments)}")
                    return []
                parsed = [c for c in raw_comments if isinstance(c, dict)]
                if not parsed:
                    logger.debug("LLM parsed JSON array but contained 0 objects ([])")

            logger.info(f"LLM parsed comments: count={len(parsed)}")
            try:
                logger.debug(f"LLM sample parsed: {parsed[:2]}")
            except Exception:
                pass

            return parsed

        except OllamaTimeoutError as e:
            logger.warning(f"LLM timeout: {e}")
            if attempt + 1 < OLLAMA_MAX_RETRIES:
                logger.info("LLM retrying...")
            else:
                logger.error("LLM max retries reached. Aborting.")
                return []
        except Exception as e:
            logger.error(f"LLM unexpected error: {e}")
            return []
        finally:
            # Ensure alarm is always cleared and apply optional pacing delay
            signal.alarm(0)
            if INTER_REQUEST_DELAY_SECONDS > 0:
                try:
                    time.sleep(INTER_REQUEST_DELAY_SECONDS)
                except Exception:
                    pass

    return []