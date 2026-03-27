import json
import re
import threading
import time
from typing import Any, Dict, List, Optional

import ollama
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableSerializable

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


def get_chat_ollama(
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    num_ctx: Optional[int] = None,
    num_batch: Optional[int] = None,
    repeat_penalty: Optional[float] = None,
    keep_alive: Optional[str] = None,
) -> ChatOllama:
    """Create a ChatOllama instance with project defaults.
    
    Args:
        model: Ollama model name. Defaults to MODEL_NAME from config.
        temperature: Sampling temperature. Defaults to OLLAMA_TEMPERATURE.
        num_ctx: Context window size. Defaults to OLLAMA_NUM_CTX.
        num_batch: Batch size. Defaults to OLLAMA_NUM_BATCH.
        repeat_penalty: Repeat penalty. Defaults to OLLAMA_REPEAT_PENALTY.
        keep_alive: Keep alive duration. Defaults to OLLAMA_KEEP_ALIVE.
        
    Returns:
        Configured ChatOllama instance.
    """
    return ChatOllama(
        model=model or MODEL_NAME,
        temperature=temperature if temperature is not None else OLLAMA_TEMPERATURE,
        num_ctx=num_ctx if num_ctx is not None else OLLAMA_NUM_CTX,
        num_predict=-1,
        repeat_penalty=repeat_penalty if repeat_penalty is not None else OLLAMA_REPEAT_PENALTY,
        keep_alive=keep_alive if keep_alive is not None else OLLAMA_KEEP_ALIVE,
    )


_llm_cache: Dict[str, ChatOllama] = {}


def get_cached_llm(model: str) -> ChatOllama:
    """Get or create a cached ChatOllama instance for the given model.
    
    Caches instances to avoid repeated initialization overhead.
    
    Args:
        model: Ollama model name.
        
    Returns:
        Cached or newly created ChatOllama instance.
    """
    if model not in _llm_cache:
        logger.debug(f"Creating new ChatOllama instance for model={model}")
        _llm_cache[model] = get_chat_ollama(model=model)
    return _llm_cache[model]


def clear_llm_cache() -> None:
    """Clear the LLM instance cache."""
    _llm_cache.clear()
    logger.debug("LLM cache cleared")


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


def _do_chat_stream(
    use_model: str,
    use_keep_alive: str,
    system_prompt: str,
    user_prompt: str,
) -> List[Dict[str, Any]]:
    """Performs the actual Ollama stream call and JSON parsing (timeout logic is separated).

    This is called in a separate thread by chat_and_parse, a thread-safe timeout wrapper.
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
                    break
            except Exception:
                pass

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

    Thread-safe timeout: uses threading.Timer instead of signal.SIGALRM so that
    multiple passes can run concurrently in a ThreadPoolExecutor without stomping
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
                f"timeout={OLLAMA_TIMEOUT_SECONDS}s"
            )

            worker_thread = threading.Thread(target=_worker, daemon=True)
            worker_thread.start()

            # Wait up to timeout; if the event fires early we continue immediately
            finished = done_event.wait(timeout=OLLAMA_TIMEOUT_SECONDS)

            if not finished:
                # Thread is still running — treat as timeout
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


def build_review_chain(
    pass_type: str,
    model: Optional[str] = None,
) -> RunnableSerializable:
    """Build a LangChain LCEL chain for a review pass.
    
    Constructs: prompt | llm | parser
    
    Args:
        pass_type: One of "defect", "refactor", "compiler", "style"
        model: Ollama model name. Defaults to MODEL_NAME.
        
    Returns:
        LCEL chain that takes {"file_path", "hunk_text", "commentable_catalog"}
        and returns List[LLMReviewComment].
    """
    from escargot_review_bot.prompts import get_prompt
    from escargot_review_bot.adapters.parsers import review_comment_list_parser
    
    prompt = get_prompt(pass_type)
    llm = get_cached_llm(model or MODEL_NAME)
    
    chain = prompt | llm | review_comment_list_parser
    logger.debug(f"Built review chain for pass_type={pass_type}, model={model or MODEL_NAME}")
    return chain


def build_judge_chain(
    model: Optional[str] = None,
) -> RunnableSerializable:
    """Build a LangChain LCEL chain for the judge pass.
    
    Constructs: prompt | llm | parser
    
    Args:
        model: Ollama model name. Defaults to MODEL_NAME.
        
    Returns:
        LCEL chain that takes {"file_path", "target_code", "proposals_text"}
        and returns List[JudgeComment].
    """
    from escargot_review_bot.prompts import get_prompt
    from escargot_review_bot.adapters.parsers import judge_comment_list_parser
    
    prompt = get_prompt("judge")
    llm = get_cached_llm(model or MODEL_NAME)
    
    chain = prompt | llm | judge_comment_list_parser
    logger.debug(f"Built judge chain with model={model or MODEL_NAME}")
    return chain