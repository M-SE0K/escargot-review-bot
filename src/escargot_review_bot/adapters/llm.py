import json
import re
import threading
import time
from typing import Any, Dict, List, Optional

import ollama

from escargot_review_bot.config.config import (
    MODEL_NAME,
    OLLAMA_KEEP_ALIVE,
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
    depth = 0
    in_str = False
    escape = False
    for i, ch in enumerate(s[start:], start):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_str:
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return (start, i)
    return None


def sanitize_llm_output(raw: str) -> str:
    """Extract a JSON array string from raw LLM output.

    Tries fenced code blocks first, then scans inline for the first JSON array.
    """
    # Try fenced blocks
    for fence_start in ("```json", "```"):
        if fence_start in raw:
            after = raw.split(fence_start, 1)[1]
            block = after.split("```", 1)[0].strip()
            if block.startswith("["):
                return block
    # Scan inline candidates
    for m in JSON_ARRAY_RE.finditer(raw):
        candidate = m.group()
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, list):
                return candidate
        except Exception:
            continue
    return ""


def _do_chat_stream(
    use_model: str,
    use_keep_alive: str,
    system_prompt: str,
    user_prompt: str,
) -> List[Dict[str, Any]]:
    """실제 Ollama 스트림 호출 및 JSON 파싱.

    Thread-safe timeout wrapper인 chat_and_parse에서 별도 daemon thread로 호출된다.
    """
    stream = ollama.chat(
        model=use_model,
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
        keep_alive=use_keep_alive,
        stream=True,
    )

    buf_parts: List[str] = []
    parsed: Optional[List[Dict[str, Any]]] = None
    for chunk in stream:
        content = getattr(getattr(chunk, "message", None), "content", None)
        if content is None and isinstance(chunk, dict):
            msg = chunk.get("message")
            if isinstance(msg, dict):
                content = msg.get("content")
        if not isinstance(content, str) or not content:
            continue
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
                    if parsed:
                        logger.debug("LLM stream-early-stop: json array complete")
                        logger.debug(f"LLM items={len(parsed)}")
                    else:
                        logger.debug("LLM stream-early-stop: empty JSON array []")
                    break
            except Exception:
                pass

    if parsed is None:
        if not buf_parts:
            logger.debug("LLM empty response (no content chunks)")
            return []
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

    return parsed or []


def chat_and_parse(
    system_prompt: str,
    user_prompt: str,
    model: Optional[str] = None,
    keep_alive: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Stream a chat completion and parse a JSON array from the output.

    model: Ollama model name. If None, uses config MODEL_NAME.
    keep_alive: e.g. "0" (unload after request), "60m". If None, uses OLLAMA_KEEP_ALIVE.

    Thread-safe timeout: uses threading.Event instead of signal.SIGALRM so that
    multiple hunks can run concurrently in a ThreadPoolExecutor without stomping
    on each other's alarm signal.
    """
    use_model = model if model is not None else MODEL_NAME
    use_keep_alive = keep_alive if keep_alive is not None else OLLAMA_KEEP_ALIVE

    for attempt in range(OLLAMA_MAX_RETRIES):
        result_holder: List[Any] = [None]   # [0] = parsed list or exception
        done_event = threading.Event()

        def _worker():
            try:
                result_holder[0] = _do_chat_stream(
                    use_model, use_keep_alive, system_prompt, user_prompt
                )
            except Exception as exc:
                result_holder[0] = exc
            finally:
                done_event.set()

        try:
            logger.info(f"LLM request start model={use_model} keep_alive={use_keep_alive}")
            logger.debug(
                f"LLM attempt {attempt + 1}/{OLLAMA_MAX_RETRIES} "
                f"timeout={OLLAMA_TIMEOUT_SECONDS}s (thread-safe)"
            )

            worker_thread = threading.Thread(target=_worker, daemon=True)
            worker_thread.start()

            finished = done_event.wait(timeout=OLLAMA_TIMEOUT_SECONDS)

            if not finished:
                raise OllamaTimeoutError(
                    f"Ollama API call timed out after {OLLAMA_TIMEOUT_SECONDS}s "
                    f"(model={use_model})"
                )

            outcome = result_holder[0]
            if isinstance(outcome, Exception):
                raise outcome

            parsed = outcome or []
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
            if INTER_REQUEST_DELAY_SECONDS > 0:
                try:
                    time.sleep(INTER_REQUEST_DELAY_SECONDS)
                except Exception:
                    pass

    return []