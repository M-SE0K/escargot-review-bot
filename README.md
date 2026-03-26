# Escargot Review Bot
A self‑hosted review server with GitHub Actions integration that generates automated code‑review comments for pull requests in the Escargot (lightweight ECMAScript engine) repository. It provides a Parallel Four-Pass LLM Review (Defects, Compiler, Refactors, Style), intelligent Judge Aggregation, robust line anchoring, and incremental review‑on‑push behavior, all powered by the LangChain framework.

## Purpose
Built to shorten review time and reduce feedback latency so contributors can submit PRs confidently and receive an early quality signal (typically within minutes). When a PR is created or updated, only the changed lines are analyzed to propose concrete defect and refactoring candidates, with comments precisely anchored to newly added lines as evidence.

The pipeline concurrently executes 4 specialized passes—Defect, Compiler, Refactor, and Style—without blocking each other. Redundant or conflicting feedback on the same lines is merged intelligently via a Judge pass to prevent self-confirmation bias. Everything runs on a local LLM and a self-hosted runner for speed and cost efficiency, and integrates cleanly with GitHub Actions.

- Generates reviews tailored specifically for the lightweight JavaScript engine Escargot.
 - Analyze PR diffs to automatically propose defect and improvement candidates across 4 diverse perspectives.
- Generate safe, well-grounded comments anchored to newly added lines.
- **Judge Aggregation:** Merges multiple comments targeting the same line based on priority (Defect ≥ Compiler ≥ Refactor, Style).
- Leverage a local LLM (Ollama) with LangChain for stable, parallel executions and cost-efficient reviews.
- **LangSmith Tracing:** Systematically tracks PR review data to continuously analyze vulnerabilities and improve bot accuracy.
- Integrate seamlessly with GitHub Actions to post comments.

## Key features
- Multiplexed Parallel Review
  - Hunk‑scoped prompts evaluated concurrently across Defect, Compiler, Refactor, and Style passes, significantly reducing overall review time and eliminating legacy `skip_ids` blockers.
- Judge Aggregation & Bias Mitigation
  - If multiple passes target the same line, a dedicated Judge model aggregates them into a single concise comment respecting priority. The Judge uses an independent evaluation structure to mitigate self-confirmation bias.
- Robust line anchoring (against HEAD)
  - Exact match → windowed nearby search (±`ALIGN_SEARCH_WINDOW`) → ±2‑line context tiebreaker; HEAD blobs cached per `{sha}:{path}`
- Incremental review (workflow integration)
  - On `synchronize`, analyze only `before..head`; per‑PR `concurrency` with `cancel-in-progress`; `DIFF_CONTEXT` applied to diffs
- Path scoping
  - Restrict review to prefixes in `REVIEW_INCLUDE_PATHS` (focus on engine‑critical directories)
- LangChain Integration & Tracing
  - LLM orchestration migrated to standard LangChain, enabling detailed LangSmith tracing for vulnerability analysis and easy integration of new LLM providers.
- Git and error handling
  - Map git subprocess failures to HTTP 500; rich debug logs for traceability
- Performance and timeouts
  - Tunable parallel workers (`REVIEW_PARALLEL_WORKERS`), timeouts (`OLLAMA_TIMEOUT_SECONDS`), and `INTER_REQUEST_DELAY_SECONDS`.

## Architecture overview
```text
[GitHub Actions (pull_request_target)]
   ├─ Compute diff range
   │    • opened/reopened  → base = PR base SHA, head = PR head SHA (full review)
   │    • synchronize      → base = before,      head = PR head SHA (incremental)
   ├─ POST /review {base, head, pr}
   ├─ Receive review.json {comments: [...]}
   └─ POST /pulls/{pr}/comments (loop, 200ms interval)

[Review Server (FastAPI + LangChain)]
   ├─ fetch_upstream_with_fallback(upstream, PR ref, SHAs)
   ├─ git diff -U{DIFF_CONTEXT} base head
   ├─ Parse PatchSet (unidiff)
   ├─ For each file filtered by REVIEW_INCLUDE_PATHS
   │    └─ For each hunk
   │         ├─ create_line_mappings_for_hunk → target_id indexing
   │         ├─ build_hunk_based_prompt (added lines only)
   │         ├─ Parallel LLM Executions (LangChain):
   │         │    ├─ Defect Pass
   │         │    ├─ Compiler Pass
   │         │    ├─ Refactor Pass
   │         │    └─ Style Pass
   │         ├─ Judge Pass: Aggregate overlapping comments (Priority: Defect ≥ Compiler ≥ Refactor, Style)
   │         └─ HEAD alignment → nearby search(±ALIGN_SEARCH_WINDOW)
   │              → ±2-line context tiebreaker → GitHubComment(line/side)
   └─ Return {comments: [...]}

[Adapters & Observability]
   • Git: run_git_command (diff/show/fetch), HEAD blob cache
   • LLM Orchestration: LangChain (JSON parsing, parallel retries)
   • Tracing: LangSmith (Performance & Vulnerability Analysis tracking)
```

### Request → response pipeline (summary)
1) Actions computes `base/head/pr` and calls the server (`synchronize` uses `before..head`).
2) The server syncs upstream and runs `git diff -U{DIFF_CONTEXT}` to produce a unified diff.
3) Parse the PatchSet with `unidiff`, then iterate per file/hunk (paths limited by `REVIEW_INCLUDE_PATHS`).
4) For each hunk, collect only `added` lines into a catalog (unique `target_id`) and build the prompt.
5) Call the LLM concurrently for the 4 passes: Defect, Compiler, Refactor, and Style using LangChain.
6) Valid JSON arrays are extracted, parsed, and validated via Pydantic schema models.
7) Judge pass: Comments generated from different passes targeting the same `target_id` are passed to the Judge model. The Judge evaluates and merges them into a highly readable single comment based on predefined priorities, effectively preventing self-confirmation bias.
8) Line anchoring: if exact HEAD alignment fails, search nearby (±`ALIGN_SEARCH_WINDOW`), then use ±2-line context to pick a unique candidate.
9) Convert valid items to `GitHubComment` with `commit_id/path/line/side` and accumulate.
10) Return `{comments:[...]}` to Actions, which posts PR review comments.

### Components
- API: `escargot_review_bot/api.py` (FastAPI, `POST /review`)
- Service: `escargot_review_bot/service.py` (diff/parsing/prompting/Judge aggregation)
- Adapters: `adapters/git.py` (git subprocess), `adapters/llm.py` (LangChain integration)
- Config/Logging: `config/config.py` (env vars), `config/logging.py` (stdout-only logger)
- Schemas/Prompts: `domain/schemas.py` (Pydantic), `prompts/*` (Defect/Compiler/Refactor/Style/Judge prompts)

## Guards and safety checks
- JSON-only output enforcement: LLM responses must be valid JSON arrays; schema-validated via Pydantic models.
- Confidence threshold: suggestions below `CONFIDENCE_THRESHOLD` are dropped.
- Path scoping: only files under `REVIEW_INCLUDE_PATHS` are considered.
- HEAD anchoring: exact match first, then nearby search (±`ALIGN_SEARCH_WINDOW`) with ±2-line context tiebreaker; ambiguous matches are skipped.
- Judge Pass Validation: Replaces legacy cross-pass de-duplication. Prevents redundant feedback and maintains objectivity by using an independent judgment evaluation instead of directly trusting individual pass outputs.
- Git safety: subprocess failures mapped to HTTP 500; upstream fetch with fallbacks; HEAD blobs cached per `{sha}:{path}`.
- Concurrency guard (workflow): per-PR group with `cancel-in-progress: true` cancels older runs on new pushes.
- LLM robustness: LangChain automatically handles parsing failures, retries (`OLLAMA_MAX_RETRIES`), and timeouts (`OLLAMA_TIMEOUT_SECONDS`).

## Requirements
- Python 3.11+ (tested on 3.12)
- Git installed, with network access to fetch from the `upstream` remote
  - `REPO_PATH` must be a valid local clone and have an `upstream` remote configured
- Ollama installed and the target models pulled (e.g., `qwen3-coder:30b`)
- Self-hosted GitHub Runner that can reach the review server (localhost or network)
- OS: Linux recommended

## Installation
```bash
# 1) Create and activate a virtual environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) Configure environment variables (.env)
cp .env.example .env && vi .env

# 3) Run the server (development)
python -m escargot_review_bot.main
# or
uvicorn escargot_review_bot.api:app --host 0.0.0.0 --port 8000
```

## Operations (systemd + journald)

In production, the service is managed by systemd and logs are viewed via journald. Detailed configuration (service user/group, paths, EnvironmentFile, ports) and any security‑sensitive values are maintained in external operations documentation and are not tracked in this repository.

1) Unit file location
- The unit is assumed to be provisioned at `/etc/systemd/system/escargot-review-bot.service`. Its contents are intentionally not documented here.

2) Enable and start
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now escargot-review-bot
```

3) Restart / stop / status
```bash
sudo systemctl restart escargot-review-bot
sudo systemctl stop escargot-review-bot
sudo systemctl status escargot-review-bot --no-pager
```

4) Logs (journald)
```bash
# Jump to the end
journalctl -u escargot-review-bot -e

# Follow live, starting with the last 200 lines
journalctl -u escargot-review-bot -f -n 200

# Current boot only
journalctl -u escargot-review-bot -b
```

5) Applying updates (example)
```bash
cd /ABS/PATH/TO/escargot-review-bot
source /ABS/PATH/TO/venv/bin/activate
pip install -r requirements.txt
sudo systemctl restart escargot-review-bot
```

### Environment variables (.env)
| Key | Default | Description |
|---|---:|---|
| `REPO_PATH` | (required) | Absolute path to the local Escargot clone used as the git working directory. Must exist and contain an `upstream` remote. |
| `LANGCHAIN_TRACING_V2` | `true` | Enables LangSmith tracing to collect and analyze PR review data. |
| `LANGCHAIN_API_KEY` | (required) | LangSmith API Key. |
| `LANGCHAIN_PROJECT` | `escargot-review-bot`| LangSmith project name for traces. |
| `REVIEW_PARALLEL_WORKERS`| `3` | Number of parallel workers for hunk concurrency. |
| `REVIEW_PARALLEL_PASSES` | `true` | Enables parallel execution for Defect, Compiler, Refactor, and Style passes. |
| `OLLAMA_KEEP_ALIVE` | `30m` | Keeps the model loaded in memory for the specified duration. |
| `OLLAMA_MODEL_DEFECT` | `model-name` | Model used for the Defect pass. |
| `OLLAMA_MODEL_REFACTOR` | `model-name` | Model used for the Refactor pass. |
| `OLLAMA_MODEL_COMPILER` | `model-name` | Model used for the Compiler pass. |
| `OLLAMA_MODEL_STYLE` | `model-name` | Model used for the Style pass. |
| `OLLAMA_TEMPERATURE` | `0.1` | Sampling temperature for the LLM. |
| `OLLAMA_NUM_CTX` | `8192` | Context window size passed to Ollama. |
| `OLLAMA_NUM_BATCH` | `256` | Batch size for Ollama generation. |
| `OLLAMA_REPEAT_PENALTY` | `1.1` | Repeat penalty for Ollama generation. |
| `CONFIDENCE_THRESHOLD` | `0.8` | Minimum confidence required for an LLM suggestion to be kept (0.0–1.0). |
| `OLLAMA_TIMEOUT_SECONDS` | `1800` | Per‑request timeout (seconds) for LLM API calls. |
| `OLLAMA_MAX_RETRIES` | `2` | Retry attempts on timeouts/unexpected parsing errors. |
| `INTER_REQUEST_DELAY_SECONDS`| `0` | Delay (seconds) to mitigate rate limits during parallel requests. |


## GitHub Actions integration (incremental review)
Workflow file: `.github/workflows/code-review.yml` in `Samsung/escargot`.

- Triggers: `pull_request_target` on `opened`, `synchronize`, `reopened`.
- Diff range selection:
  - `opened`/`reopened`: `base = pr.base.sha`, `head = pr.head.sha` (full review)
  - `synchronize`: `base = before`, `head = pr.head.sha` (only the latest push)
- Steps:
  1) Compute range (github-script)
  2) POST to review server: `POST $REVIEW_SERVER/review` with `{base_sha, head_sha, pull_request_number}`
  3) Read `review.json` and post PR comments (200 ms spacing)
- Within the same PR: when a new push arrives, any in‑progress older run is canceled (`cancel-in-progress: true`), so only the latest push gets reviewed. Runs for other PRs may still execute in parallel.

## API
- Endpoint: `POST /review`
- Request (JSON):
```json
{
  "base_sha": "<40-hex>",
  "head_sha": "<40-hex>",
  "pull_request_number": 123
}
```
- Response (JSON):
```json
{
  "comments": [
    {
      "path": "src/file.cpp",
      "body": "comment body",
      "commit_id": "<head sha>",
      "line": 42,
      "side": "RIGHT"
    }
  ]
}
```