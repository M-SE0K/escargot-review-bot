from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Set, Tuple

from fastapi import HTTPException
from unidiff import PatchSet, Hunk

from escargot_review_bot.adapters.git import run_git_command
from escargot_review_bot.adapters.llm import chat_and_parse
from escargot_review_bot.config.config import (
    ALIGN_SEARCH_WINDOW,
    CONFIDENCE_THRESHOLD,
    DIFF_CONTEXT,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_MODEL_COMPILER,
    OLLAMA_MODEL_DEFECT,
    OLLAMA_MODEL_REFACTOR,
    REVIEW_INCLUDE_PATHS,
    REVIEW_PARALLEL_WORKERS,
)
from escargot_review_bot.config.logging import get_logger
from escargot_review_bot.domain.schemas import (
    GitHubComment,
    LLMReviewComment,
    ReviewRequest,
)
from escargot_review_bot.prompts.defect import SYSTEM_PROMPT_DEFECT
from escargot_review_bot.prompts.refactor import SYSTEM_PROMPT_REFACTOR
from escargot_review_bot.prompts.compiler import SYSTEM_PROMPT_COMPILER


logger = get_logger("review-bot.service")


class LineMappingLite:
    """Unified diff line mapping with stable `target_id` and side line numbers."""
    def __init__(self, target_id: int, line_type: str, content: str,
                 source_line_no: Optional[int], target_line_no: Optional[int]) -> None:
        self.target_id = target_id
        self.line_type = line_type
        self.content = content
        self.source_line_no = source_line_no
        self.target_line_no = target_line_no


def create_line_mappings_for_hunk(hunk: Hunk) -> List[LineMappingLite]:
    """Build `LineMappingLite` list from a hunk with stable IDs and positions.

    Assigns a monotonic `target_id` to each hunk line and records source/target
    side line numbers alongside raw content for later anchoring.
    """
    # Build stable ID -> line mapping while tracking left/right cursors
    mappings: List[LineMappingLite] = []
    current_id = 1

    right_line = hunk.target_start
    left_line = hunk.source_start

    for line in hunk:
        if line.is_added:
            # Added lines exist only on the right (target) side
            mappings.append(LineMappingLite(
                target_id=current_id,
                line_type='added',
                content=line.value,
                source_line_no=None,
                target_line_no=right_line,
            ))
            right_line += 1
        elif line.is_removed:
            # Removed lines exist only on the left (source) side
            mappings.append(LineMappingLite(
                target_id=current_id,
                line_type='removed',
                content=line.value,
                source_line_no=left_line,
                target_line_no=None,
            ))
            left_line += 1
        else:
            # Context lines exist on both sides and advance both cursors
            mappings.append(LineMappingLite(
                target_id=current_id,
                line_type='context',
                content=line.value,
                source_line_no=left_line,
                target_line_no=right_line,
            ))
            left_line += 1
            right_line += 1
        current_id += 1
    return mappings


def normalize_for_compare(s: str) -> str:
    """Expand tabs(4) and strip to normalize for alignment comparison."""
    return (s or "").expandtabs(4).strip()


def line_without_prefix(raw: str) -> str:
    """Remove leading '+'/'-' from diff line; return empty string if falsy."""
    if not raw:
        return ""
    if raw[0] in {"+", "-"}:
        return raw[1:]
    return raw


def is_meaningful_code(raw: str) -> bool:
    """Heuristic: exclude empty/brace-only lines from review candidates."""
    s = (raw or "").strip()
    if not s:
        return False
    if s in {'{', '}', '};'}:
        return False
    return True


def _collect_target_side_context(
    mappings: List[LineMappingLite],
    center_index: int,
    max_depth: int = 2,
) -> Tuple[List[str], List[str]]:
    """Collect up to `max_depth` normalized target-side neighbor lines.

    Returns (prev_list, next_list), where each element is normalized with
    `line_without_prefix` and `normalize_for_compare`. Only mappings with a
    valid `target_line_no` are considered.
    """
    prev_ctx: List[str] = []
    next_ctx: List[str] = []

    # Walk left for previous target-side lines
    i = center_index - 1
    while i >= 0 and len(prev_ctx) < max_depth:
        mi = mappings[i]
        if mi.target_line_no is not None:
            prev_ctx.append(normalize_for_compare(line_without_prefix(mi.content)))
        i -= 1

    # Walk right for next target-side lines
    i = center_index + 1
    while i < len(mappings) and len(next_ctx) < max_depth:
        mi = mappings[i]
        if mi.target_line_no is not None:
            next_ctx.append(normalize_for_compare(line_without_prefix(mi.content)))
        i += 1

    return prev_ctx, next_ctx


def assert_head_alignment(head_sha: str, path: str, mapping: LineMappingLite,
                          head_cache: Dict[str, List[str]]) -> Optional[bool]:
    """Check exact alignment of an added line at `target_line_no` in HEAD.

    Uses a cached `git show {head_sha}:{path}` blob. Returns True/False for
    match/mismatch, or None if not applicable (non-added or missing position).
    """
    if mapping.line_type != 'added' or mapping.target_line_no is None:
        return None

    # Lazy-load and cache the HEAD blob lines for this file
    key = f"{head_sha}:{path}"
    if key not in head_cache:
        blob_text = run_git_command(["show", f"{head_sha}:{path}"])
        head_cache[key] = blob_text.splitlines()

    lines = head_cache[key]
    idx = mapping.target_line_no - 1
    # Treat out-of-range as invalid expected position against HEAD
    if not (0 <= idx < len(lines)):
        logger.debug(f"Align out-of-range: {path}:{mapping.target_line_no} (len={len(lines)})")
        return False

    # Normalize both sides before equality check to avoid whitespace noise
    expected = normalize_for_compare(line_without_prefix(mapping.content))
    actual = normalize_for_compare(lines[idx])

    if expected == actual:
        return True

    logger.debug(
        f"Align mismatch: {path}:{mapping.target_line_no} expected={expected!r} actual={actual!r}"
    )
    return False


def try_nearby_align(
    head_sha: str,
    path: str,
    mapping: LineMappingLite,
    head_cache: Dict[str, List[str]],
    prev_context: Optional[List[str]] = None,
    next_context: Optional[List[str]] = None,
) -> Optional[int]:
    """Search within +/-`ALIGN_SEARCH_WINDOW` for a nearby normalized match.

    If multiple candidates are found, disambiguate using up to 1-2 lines of
    previous/next target-side context. Only a unique highest-scoring candidate
    is accepted; otherwise return None.
    """
    if mapping.line_type != 'added' or mapping.target_line_no is None:
        return None

    # Reuse cached HEAD blob if already loaded; otherwise load once
    key = f"{head_sha}:{path}"
    if key not in head_cache:
        blob_text = run_git_command(["show", f"{head_sha}:{path}"])
        head_cache[key] = blob_text.splitlines()

    lines = head_cache[key]
    total = len(lines)
    base_idx = mapping.target_line_no - 1
    expected = normalize_for_compare(line_without_prefix(mapping.content))

    # Quick path: current index already matches after normalization
    if 0 <= base_idx < total and normalize_for_compare(lines[base_idx]) == expected:
        return mapping.target_line_no

    # Collect all candidate positions within the search window
    candidates: List[int] = []
    for delta in range(1, ALIGN_SEARCH_WINDOW + 1):
        up = base_idx - delta
        if 0 <= up < total and normalize_for_compare(lines[up]) == expected:
            candidates.append(up)
        down = base_idx + delta
        if 0 <= down < total and normalize_for_compare(lines[down]) == expected:
            candidates.append(down)

    if not candidates:
        return None

    # If single candidate, accept it
    if len(candidates) == 1:
        return candidates[0] + 1

    # Disambiguate using neighbor context if provided
    prev_context = prev_context or []
    next_context = next_context or []

    best_idx: Optional[int] = None
    best_score = -1
    tie = False

    for pos in candidates:
        score = 0
        # Match previous neighbors: prev_context[0] is nearest neighbor
        for offset, txt in enumerate(prev_context, start=1):
            nei = pos - offset
            if 0 <= nei < total and normalize_for_compare(lines[nei]) == txt:
                score += 1
        # Match next neighbors: next_context[0] is nearest neighbor
        for offset, txt in enumerate(next_context, start=1):
            nei = pos + offset
            if 0 <= nei < total and normalize_for_compare(lines[nei]) == txt:
                score += 1

        if score > best_score:
            best_score = score
            best_idx = pos
            tie = False
        elif score == best_score:
            tie = True

    # Accept only a unique highest-scoring candidate with some context support
    if best_idx is not None and not tie and best_score > 0:
        return best_idx + 1

    return None


def build_hunk_based_prompt(path: str, hunk: Hunk, mappings: List[LineMappingLite]) -> str:
    """Compose a hunk review prompt including diff and Commentable Catalog.

    Catalog lists only added lines with `<ID ...>` anchors to constrain output.
    The prompt enforces hunk-locality and a strict JSON-only response.
    """
    # Render raw diff hunk and assemble Commentable Catalog (added lines only)
    hunk_text = str(hunk)
    commentable = [
        m for m in mappings
        if m.line_type == 'added' and is_meaningful_code(line_without_prefix(m.content))
    ]
    if commentable:
        commentable_catalog = [
            f"<ID {m.target_id} | {m.line_type.upper()}>: {line_without_prefix(m.content).strip()}"
            for m in commentable
        ]
        commentable_str = "\n".join(commentable_catalog)
    else:
        commentable_str = "(no added lines)"

    # Compose the strict prompt instructing JSON-only output anchored by IDs
    return f"""
You are a world-class C++ and JavaScript engine reviewer for the Escargot project. Your review must be strict and technically precise.

## Target File: `{path}`
## Review Task
Your task is to review the code changes within the `DIFF HUNK` section only. Use the diff purely; do NOT comment on any line outside of the 'Commentable Catalog'.

### Hard Rules
- Choose "target_id" ONLY from **Commentable Catalog (ADDED lines only)**.
- If no qualifying added line has an issue, return [].
- Do NOT mention or infer any line numbers (e.g., "line 47", "at 115"). Anchor only by exact tokens from the chosen line.

---
### 1. DIFF HUNK
```diff
{hunk_text}
```

---
### 2. Commentable Catalog (ADDED lines only; eligible IDs)
```
{commentable_str}
```

### Output (JSON array only)
Each object: "target_id", "body", "confidence". If none, return [].
""".strip()


def fetch_upstream_with_fallback(pull_request_number: int, base_sha: str, head_sha: str) -> None:
    """Ensure PR refs/SHAs exist locally with pragmatic fallback fetches.

    Prunes/fetches upstream, tries PR ref, then validates and fetches SHAs
    directly if missing; raises HTTPException on final absence.
    """
    # 1) prune and fetch upstream to refresh remote refs
    try:
        logger.debug("Upstream prune fetch start")
        run_git_command(["fetch", "upstream", "--prune"])
        logger.debug("Upstream prune fetch done")
    except Exception as e:
        logger.warning(f"Upstream prune fetch failed (continuing): {e}")

    # 2) attempt to fetch PR head ref; fall back to raw SHAs if missing
    try:
        run_git_command(["fetch", "upstream", f"refs/pull/{pull_request_number}/head"])
    except Exception as e:
        logger.warning(f"PR ref not found (continuing with SHAs): {e}")

    # 3) ensure both base/head SHAs are present; try direct SHA fetch when absent
    for sha in [base_sha, head_sha]:
        try:
            run_git_command(["cat-file", "-e", f"{sha}^{{commit}}"])
            continue
        except Exception:
            pass
        try:
            run_git_command(["fetch", "upstream", sha])
        except Exception as e:
            logger.warning(f"Direct SHA fetch failed (sha={sha}): {e}")
        try:
            run_git_command(["cat-file", "-e", f"{sha}^{{commit}}"])
        except Exception:
            logger.error(f"Missing commit after fetch attempts: {sha}")
            raise HTTPException(status_code=400, detail=f"Missing commit in upstream: {sha}")


def _run_review_pass(
    model_type: str,
    model_name: str,
    system_prompt: str,
    file_path: str,
    hunk: Hunk,
    mappings: List[LineMappingLite],
    mapping_dict: Dict[int, Any],
    head_sha: str,
    head_blob_cache: Dict[str, List[str]],
    skip_ids: Set[int] | None = None,
) -> Tuple[List[Dict[str, Any]], Set[int]]:
    """Run one pass (defect/refactor/compiler): invoke LLM and post-filter results.

    model_name: Ollama model for this pass (multi-model sequential: Load -> Run -> Unload when keep_alive=0).
    """
    prompt = build_hunk_based_prompt(file_path, hunk, mappings)
    try:
        logger.debug(f"{model_type.title()} pass: model={model_name} prompt length={len(prompt)}")
    except Exception:
        logger.debug(f"{model_type.title()} pass: model={model_name}")

    raw = chat_and_parse(system_prompt, prompt, model=model_name)
    logger.info(f"{model_type} pass: LLM returned {len(raw)} raw comment(s)")

    # Filter pipeline: schema -> dedup -> confidence -> mapping -> alignment
    out_comments: List[Dict[str, Any]] = []
    accepted: Set[int] = set()

    for c in raw:
        try:
            llm_comment = LLMReviewComment(**c)
        except Exception as e:
            keys = list(c.keys()) if isinstance(c, dict) else "n/a"
            logger.debug(f"Skip({model_type}): schema invalid -> {e} | keys={keys} | raw={c}")
            continue

        # Skip IDs already accepted in a previous pass (dedup across passes)
        if skip_ids and llm_comment.target_id in skip_ids:
            logger.debug(f"Skip({model_type}): already accepted id={llm_comment.target_id}")
            continue

        # Same-pass dedup: one comment per line per pass
        if llm_comment.target_id in accepted:
            logger.debug(f"Skip({model_type}): duplicate target_id={llm_comment.target_id} in this pass")
            continue

        # Enforce minimum confidence threshold (LLM produces the value)
        if llm_comment.confidence < CONFIDENCE_THRESHOLD:
            logger.debug(f"Skip({model_type}): low confidence {llm_comment.confidence:.2f} < {CONFIDENCE_THRESHOLD}")
            continue

        m = mapping_dict.get(llm_comment.target_id)
        if not m or m.line_type != 'added' or m.target_line_no is None:
            logger.debug(f"Skip({model_type}): invalid target_id={llm_comment.target_id} or not added line")
            continue

        line_no = m.target_line_no
        head_ok = assert_head_alignment(head_sha, file_path, m, head_blob_cache)
        if head_ok is False:
            # Try to locate the same line content nearby if exact slot changed
            logger.debug(f"Align mismatch at ~{line_no}, trying nearby align...")
            # Build small neighbor context around the target line within the hunk
            try:
                center_index = next(i for i, mm in enumerate(mappings) if mm.target_id == m.target_id)
            except StopIteration:
                center_index = None
            prev_ctx: List[str] = []
            next_ctx: List[str] = []
            if center_index is not None:
                prev_ctx, next_ctx = _collect_target_side_context(mappings, center_index, max_depth=2)

            aligned = try_nearby_align(
                head_sha,
                file_path,
                m,
                head_blob_cache,
                prev_context=prev_ctx,
                next_context=next_ctx,
            )
            if aligned is None:
                logger.debug(f"Skip({model_type}): nearby align failed")
                continue
            line_no = aligned

        final_comment = GitHubComment(
            path=file_path,
            body=llm_comment.body,
            commit_id=head_sha,
            line=line_no,
            side="RIGHT"
        )
        out_comments.append(final_comment.model_dump())
        accepted.add(llm_comment.target_id)
        logger.debug(f"Accept({model_type}): id={llm_comment.target_id} -> line={line_no}")

    if len(raw) > 0 and len(out_comments) == 0:
        logger.warning(
            f"{model_type} pass: all {len(raw)} comment(s) dropped by filters "
            "(schema/confidence/target_id/HEAD alignment). Check LOG_LEVEL=DEBUG for Skip reasons."
        )
    return out_comments, accepted


def generate_review_comments(request: ReviewRequest) -> List[Dict[str, Any]]:
    """End-to-end review across diff: three passes per hunk, aggregate comments.

    Fetches upstream, builds unified diff, runs defect, refactor, and compiler
    optimization passes per hunk, applies confidence/alignment checks, and returns
    GitHub comments.
    """
    logger.info(f"Start review PR=#{request.pull_request_number} {request.base_sha}..{request.head_sha}")

    # Fetch upstream refs and ensure base/head SHAs are available locally
    logger.info("Fetching latest data from upstream...")
    fetch_upstream_with_fallback(request.pull_request_number, request.base_sha, request.head_sha)
    logger.info("Fetch complete.")

    # Build unified diff between base..head with configured context lines
    diff_text = run_git_command([
        "diff", "--no-color", "--no-ext-diff", "--text",
        f"-U{DIFF_CONTEXT}", request.base_sha, request.head_sha
    ])
    diff_text = diff_text.replace("\r\n", "\n")
    if not diff_text.endswith("\n"):
        diff_text += "\n"

    try:
        # Parse diff into PatchSet
        patch_set = PatchSet.from_string(diff_text)
        try:
            logger.info(f"Diff created. files={len(patch_set)}")
        except Exception:
            logger.info("Diff created. (could not count files)")
    except Exception as e:
        logger.exception(f"Diff parse failed: {e}")
        return []

    # HEAD blob cache: pre-fill for all included files (read-only in workers)
    head_blob_cache: Dict[str, List[str]] = {}
    for patched_file in patch_set:
        file_path = patched_file.path
        if not any(file_path.startswith(p) for p in REVIEW_INCLUDE_PATHS):
            continue
        key = f"{request.head_sha}:{file_path}"
        if key not in head_blob_cache:
            try:
                blob_text = run_git_command(["show", f"{request.head_sha}:{file_path}"])
                head_blob_cache[key] = blob_text.splitlines()
            except Exception as e:
                logger.debug(f"Could not load blob {key}: {e}")
                head_blob_cache[key] = []

    # Build list of (file_path, hunk, mappings, mapping_dict) for all hunks
    hunk_items: List[Tuple[str, Hunk, List[LineMappingLite], Dict[int, Any]]] = []
    for patched_file in patch_set:
        file_path = patched_file.path
        if not any(file_path.startswith(p) for p in REVIEW_INCLUDE_PATHS):
            continue
        for hunk in patched_file:
            mappings = create_line_mappings_for_hunk(hunk)
            if not mappings:
                continue
            mapping_dict = {m.target_id: m for m in mappings}
            hunk_items.append((file_path, hunk, mappings, mapping_dict))

    if not hunk_items:
        logger.info("No hunks to review.")
        return []

    # OLLAMA_KEEP_ALIVE=0 이면 같은 모델로 동시 요청 시 먼저 끝난 요청이 모델을 내려 나머지가 실패하므로 workers=1로 순차 처리
    workers = max(1, REVIEW_PARALLEL_WORKERS)
    if OLLAMA_KEEP_ALIVE == "0" and workers > 1:
        logger.info(
            f"OLLAMA_KEEP_ALIVE=0: forcing workers=1 to avoid model unload race (was {workers})."
        )
        workers = 1
    logger.info(f"Parallel review: {len(hunk_items)} hunks, {workers} workers per pass.")

    def run_defect(i: int) -> Tuple[List[Dict[str, Any]], Set[int]]:
        fp, h, m, md = hunk_items[i]
        return _run_review_pass(
            model_type="defect",
            model_name=OLLAMA_MODEL_DEFECT,
            system_prompt=SYSTEM_PROMPT_DEFECT,
            file_path=fp,
            hunk=h,
            mappings=m,
            mapping_dict=md,
            head_sha=request.head_sha,
            head_blob_cache=head_blob_cache,
        )

    def run_refactor(i: int, skip_ids: Set[int]) -> Tuple[List[Dict[str, Any]], Set[int]]:
        fp, h, m, md = hunk_items[i]
        return _run_review_pass(
            model_type="refactor",
            model_name=OLLAMA_MODEL_REFACTOR,
            system_prompt=SYSTEM_PROMPT_REFACTOR,
            file_path=fp,
            hunk=h,
            mappings=m,
            mapping_dict=md,
            head_sha=request.head_sha,
            head_blob_cache=head_blob_cache,
            skip_ids=skip_ids,
        )

    def run_compiler(i: int, skip_ids: Set[int]) -> Tuple[List[Dict[str, Any]], Set[int]]:
        fp, h, m, md = hunk_items[i]
        return _run_review_pass(
            model_type="compiler",
            model_name=OLLAMA_MODEL_COMPILER,
            system_prompt=SYSTEM_PROMPT_COMPILER,
            file_path=fp,
            hunk=h,
            mappings=m,
            mapping_dict=md,
            head_sha=request.head_sha,
            head_blob_cache=head_blob_cache,
            skip_ids=skip_ids,
        )

    # Phase 1: Defect (parallel with workers, or sequential when workers=1)
    defect_results: List[Tuple[List[Dict[str, Any]], Set[int]]] = [
        ([], set()) for _ in range(len(hunk_items))
    ]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_idx = {executor.submit(run_defect, i): i for i in range(len(hunk_items))}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                defect_results[idx] = future.result()
            except Exception as e:
                logger.exception(f"Defect pass failed for hunk {idx}: {e}")

    # Phase 2: Refactor
    refactor_results: List[Tuple[List[Dict[str, Any]], Set[int]]] = [
        ([], set()) for _ in range(len(hunk_items))
    ]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_idx = {
            executor.submit(run_refactor, i, defect_results[i][1]): i
            for i in range(len(hunk_items))
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                refactor_results[idx] = future.result()
            except Exception as e:
                logger.exception(f"Refactor pass failed for hunk {idx}: {e}")

    # Phase 3: Compiler
    compiler_results: List[List[Dict[str, Any]]] = [[] for _ in range(len(hunk_items))]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_idx = {
            executor.submit(run_compiler, i, defect_results[i][1] | refactor_results[i][1]): i
            for i in range(len(hunk_items))
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                comments, _ = future.result()
                compiler_results[idx] = comments
            except Exception as e:
                logger.exception(f"Compiler pass failed for hunk {idx}: {e}")

    all_github_comments: List[Dict[str, Any]] = []
    for i in range(len(hunk_items)):
        all_github_comments.extend(defect_results[i][0])
        all_github_comments.extend(refactor_results[i][0])
        all_github_comments.extend(compiler_results[i])

    logger.info(f"Generated {len(all_github_comments)} comments in total.")
    return all_github_comments

