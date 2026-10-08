# Deep Memory Implementation Plan

Two-tier memory: hot memory (`pletykas-memory.json`, always injected) + deep
memory (`pletykas-deepmemory.json`, same schema, retrieved via embeddings).
Auto-retrieval in code — NO model-facing recall tag. Vision path excluded from
recall in v1.

Decisions locked in prior discussion (do not relitigate):
- Archive threshold: entries older than the configured archive age (state
  setting `deep_archive_days`, default 7 days, `0` disables age candidacy) are
  archive *candidates*; the LLM curator picks during curation (age selects,
  curator decides). Forced eviction only as pathological-growth fallback.
- Reintroduce `updated_at` so refreshed entries reset the archive-age clock.
- Retrieval: embedding similarity (NOT keyword), because chat is Hungarian and
  memories are curated in English — keyword matching across that split only
  fires on proper nouns.
- Embeddings live in a sidecar file `pletykas-deepmemory-embeddings.json`
  (content-hash -> vector cache), NOT inline in the deep JSON. Keeps deep JSON
  byte-schema-identical to hot JSON so `validate_memory_dict` and admin
  upload/download work untouched.
- Vision/photo evaluation paths do NOT get deep-memory recall in v1.
- Embedding route: OpenAI-compatible `POST {api_base}/embeddings` via the
  existing aiohttp session (new `_call_openai_embeddings` in llm.py). Default
  model `google/gemini-embedding-2` (OpenRouter slug) so a future OpenRouter
  switch is config-only (`MODEL_EMBED_*` env). Native genai-SDK embedding is
  explicitly OUT of scope: the route decision is transport-first, not
  provider-first, and the OpenRouter slug is NOT a valid genai SDK model name.
- Retrieval runs once per LLM *evaluation*, never per arrived message (most
  arrived messages are debounced away without evaluation; embedding each would
  waste API calls).
- All embedding failures degrade to no-injection (log + continue). Retrieval
  must NEVER break the reply path.
- HTTP timeouts (user-locked 2026-10-08): embeddings 5s total per request;
  conversational chat/vision completions 10s total per request. Curation +
  forget-analysis calls are EXEMPT (200k-token JSON outputs cannot fit 10s —
  they keep the 120s session default). Image generation/editing is EXEMPT
  (unchanged session default). Rationale: a hung provider must fail fast on
  the interactive path; slow/large background jobs keep headroom.

Conventions each agent MUST follow (from AGENTS.md + codebase):
- Atomic persistence: `tempfile.NamedTemporaryFile` + `os.replace` (see
  `MemoryManager.save` in memory.py:155). Never write JSON files directly.
- Thread-safety: `threading.RLock` around manager data mutations; curation
  runs in a background thread (`_curate_worker`, handlers.py:873).
- Mocks avoid network I/O. Telegram Update/Message mocks need valid attrs.
- Tests run with `.venv/bin/pytest -v`.
- Debug banners to stdout via `self.state.is_debug_mode()` +
  `_log_debug_payload` (llm.py:272) / `_log_debug_group_msg` (handlers.py:282).
- Git commits ONLY with explicit user approval.

Execution order: Phase 1 -> 2 -> 3 -> 4 -> 5 (each phase depends on the
previous; steps WITHIN a phase may run in parallel where noted).
Verification gate after each phase: run `.venv/bin/pytest -v` (must stay
green) plus the phase-specific checks listed.

Live-input protocol (applies to every phase gate involving a RUNNING bot):
whenever a verification step needs the agent to observe a Telegram interaction
it cannot produce itself, the agent MUST ask the user for input instead of
skipping the check. The user will type the requested message into either the
private admin chat (for `/commands` and uploads) or the group chat (for
conversational triggers) on demand. Concretely: the agent states exactly what
to type where (e.g. "send `/curate` in the private admin chat", "send `mizu
Titusznál?` in the group"), then waits for the user to confirm it was sent
before reading logs/files and judging pass/fail. NEVER mark a live gate
passed without either exercising it via user input or explicitly reporting it
as skipped-with-reason in the final message.

---

## Phase 1: Deep store + timestamp plumbing (memory.py, params.py, state.py)

Goal: second MemoryManager exists, `updated_at` is back, deep file has own
params wiring. No behavior change yet.

### 1.1 Reintroduce `updated_at` on fact entries
File: `memory.py`.
Background: `MemoryManager.load()` at memory.py:128-132 currently STRIPS
`updated_at` from loaded fact entries (`m.pop("updated_at", None)`). This was
a prior cleanup; the plan needs it back so `update_memory()` resets the
archive-age clock.
Work:
- Remove the `m.pop("updated_at", None)` line. Keep popping legacy `id`.
- In `add_memory()` (memory.py:206): set BOTH `created_at` and `updated_at`
  to the same `_utc_iso_now()` value.
- In `update_memory()` (memory.py:218): when content is replaced, also set
  `entry["updated_at"] = _utc_iso_now()`.
- Entries loaded from old files may lack `updated_at`: in `load()`, after
  stripping legacy fields, backfill `m.setdefault("updated_at",
  m.get("created_at", ""))` for every fact entry.
- Dynamics and jokes: leave unchanged (no `updated_at`; their archive age
  uses `created_at` only).
- Update `validate_memory_dict()` (memory.py:19) so a fact entry with an
  `updated_at` string passes validation and is preserved in the cleaned
  output (currently it strips unknown/legacy fields — check what it does
  with extra keys and make `updated_at` explicitly allowed, falling back to
  `created_at` when missing).
Tests: extend `tests/test_memory.py`: add entry has both timestamps;
update entry bumps `updated_at` but keeps `created_at`; loading a legacy
file without `updated_at` backfills from `created_at`; validator preserves
`updated_at`.

### 1.2 Add `deepmemory_file` param wiring
File: `params.py`.
Background: `Params.__init__` defines `self.memory_file =
"pletykas-memory.json"` (params.py:20) and `parse()` binds `--memory-file` /
`MEMORY_FILE` (params.py:123-128). Mirror this exactly.
Work:
- Add `self.deepmemory_file: str = "pletykas-deepmemory.json"` in `__init__`.
- Add a `--deepmemory-file` argparse arg with `dest="deepmemory_file"`,
  `default=os.environ.get("DEEPMEMORY_FILE", "pletykas-deepmemory.json")`,
  help "Path to deep memory JSON file (cold tier)".
- Find where `parse()` assigns `self.memory_file` from parsed args (near the
  end of parse, same validation style as `chathistory_file`) and add the
  identical assignment for `deepmemory_file`.
Tests: extend `tests/test_params.py`: default value, CLI flag override, env
var override. Follow existing test patterns in that file.

### 1.3 `deep_archive_days` state setting
File: `state.py`.
Background: tunables live in `StateManager.data` with a default in
`_create_default_state()` (state.py:78) and a `get_`/`set_` pair, e.g.
`cooldown_sec` default 3 (state.py:85) with `get_cooldown_sec`/`set_cooldown_sec`
(state.py:490-496, clamped `max(0, int(sec))`). Old state files missing the key
must still work: every getter uses `self.data.get(KEY, DEFAULT)`, so no
migration is needed.
Work:
- Add `"deep_archive_days": 7` to `_create_default_state()`.
- Add `get_deep_archive_days() -> int`: `return max(0,
  int(self.data.get("deep_archive_days", 7)))`. Wrap in try/except
  (ValueError, TypeError) returning 7 on garbage values — state files are
  hand-editable and a corrupt value must not crash curation.
- Add `set_deep_archive_days(days: int) -> None`: clamp `max(0, int(days))`,
  store, `self.save()`. Semantics: `0` means age-based candidacy is DISABLED
  (no entry becomes an archive candidate by age; curator archive lists and
  forced eviction still apply).
- Document the `0`-disables semantics in both docstrings.
Tests: extend `tests/test_state.py`: default is 7 on fresh state AND on a
legacy state dict lacking the key; setter clamps negatives to 0; garbage value
(e.g. string) falls back to 7. Follow existing getter/setter test patterns.

### 1.4 Instantiate + persist the deep MemoryManager in main.py
File: `main.py`.
Background: hot memory is created at main.py:129-130 (`MemoryManager(
params.memory_file)` + `load()`), passed to `BotHandlers(params, state,
memory, llm)` at main.py:133, and saved in `post_shutdown` at main.py:149-157.
Work:
- After the hot-memory block, create `deep_memory =
  MemoryManager(params.deepmemory_file)` and call `deep_memory.load()`.
  Backup rotation is automatic inside `create_backup()` and is filename-based
  (`deepmemory-*` pattern), so no extra backup code is needed.
- Change the `BotHandlers(...)` constructor call to pass `deep_memory`
  (this requires step 2.1's signature change — coordinate: implement the
  `BotHandlers.__init__` change in this step too, defaulting
  `deep_memory=None` and creating an empty in-memory manager if None, so
  existing tests constructing BotHandlers without it do not break).
- In `post_shutdown`, also call `deep_memory.save()`.
Tests: none new here (covered by integration in Phase 5), but ALL existing
tests must still pass — run `.venv/bin/pytest -v`.

### 1.5 `MODEL_EMBED_*` params (name/base/key/dim)
File: `params.py`.
Background: model families follow the large/image pattern: `__init__` defaults
(params.py:27-38), argparse block with env fallbacks (params.py:156-212),
strip-assignments at end of `parse()` (params.py:252-260), and
`effective_*` fallback properties (params.py:41-69). The large model falls
back to the PRIMARY key/base when unset (`effective_large_api_key`,
`effective_large_api_base`).
Work:
- `__init__` defaults: `self.model_embed_name: str =
  "google/gemini-embedding-2"`, `self.model_embed_api_key: str = ""`,
  `self.model_embed_api_base: str = ""`, `self.model_embed_dim: int = 0`
  (`0` = provider default; only sent when positive).
- Argparse: `--model-embed-name` (`MODEL_EMBED_NAME`, default
  `"google/gemini-embedding-2"`), `--model-embed-api-key`
  (`MODEL_EMBED_API_KEY`, default `""`), `--model-embed-api-base`
  (`MODEL_EMBED_API_BASE`, default `""`), `--model-embed-dim`
  (`MODEL_EMBED_DIM`, default `"0"`).
- End of `parse()`: strip-assign name/key/base; parse dim with
  `int(...)` inside try/except falling back to 0 on garbage, then
  `max(0, dim)`.
- Fallback properties (mirror `effective_large_*` exactly):
  `effective_embed_api_key` => `self.model_embed_api_key or
  self.model_api_key`; `effective_embed_api_base` =>
  `self.model_embed_api_base or self.model_api_base`.
- Document in comments: OpenRouter conversion later = set
  `MODEL_EMBED_API_BASE=https://openrouter.ai/api/v1` +
  `MODEL_EMBED_API_KEY=<key>`; model slug stays as-is. Native genai-SDK
  embedding is OUT of scope for this plan.
- `config.inc.sh-example`: add a commented `MODEL_EMBED_*` block after the
  image block (all commented out = primary-model fallback in effect).
Tests: extend `tests/test_params.py`: defaults (name =
  "google/gemini-embedding-2", dim 0, key/base empty); CLI + env overrides;
  fallback properties (unset => primary key/base; set => own values); garbage
  dim (`"abc"`, `"-5"`) => 0.

Phase 1 verification gate: `pytest` green; bot starts and creates an empty
`pletykas-deepmemory.json` on first run; `/status` output unchanged; embed
params parse with OpenRouter-ready defaults.

---

## Phase 2: Archival + promotion + forget-across-stores (llm.py curation contract, handlers.py worker, memory.py moves)

Goal: entries older than the configured archive age can move hot<->deep;
`<FORGET>` clears both stores. No retrieval yet (deep entries are inert until
Phase 3).

### 2.1 Add move primitives to MemoryManager
File: `memory.py`.
Background: `discard_memory/discard_dynamic/discard_inside_joke` remove by
fuzzy target match; `add_memory/add_dynamic/add_inside_joke` append with
fresh timestamps. An archive MOVE must preserve the original `created_at`
(and `updated_at`), so plain add+discard is wrong — add would mint new
timestamps.
Work:
- Add `pop_memory(target) -> Optional[Dict]`: find the FIRST fact matching
  `_matches_memory_target(entry, target)`, remove it from the list, `save()`,
  return the removed dict (with original timestamps intact); return None if
  no match. Hold `self._lock` throughout.
- Add `pop_dynamic(target)` and `pop_inside_joke(target)` analogously, using
  `_matches_dynamic_target` / `_matches_inside_joke_target`.
- Add `append_entry(section: str, entry: Dict) -> None`: `section` is one of
  `"memories" | "dynamics" | "inside_jokes"`; append a shallow copy of
  `entry` verbatim (no timestamp rewrite), `save()`. Raise ValueError on
  unknown section.
- Locking note: callers will hold NO manager locks when calling these; each
  method takes its own lock. A move is two atomic saves (pop saves, append
  saves) with a crash window where the entry exists in NEITHER store after
  pop but before append. This is acceptable per plan (both files are
  rewritten atomically; a crash loses at most one low-value archived entry).
  To shrink the window, implement moves as append-first-then-pop in the
  caller (step 2.3): append to deep, THEN pop from hot — a crash then leaves
  a duplicate, which beats a loss. Document this ordering in a comment.
Tests: `tests/test_memory.py`: pop returns entry with original timestamps
and removes it; pop of missing target returns None; append_entry preserves
timestamps verbatim; invalid section raises.

### 2.2 Extend `curate_memory` contract with archive/promote lists
File: `llm.py`, function `curate_memory` (llm.py:1578).
Background: the curation prompt shows `[Current Memory Entries]` as JSON and
demands a strict JSON object with `facts_to_add/update/discard`,
`dynamics_to_add/discard`, `jokes_to_add/discard` (llm.py:1602-1611). The
`default_result` dict (llm.py:1648-1656) lists the same keys; parsing
(llm.py:1664-1670) coerces missing/non-list keys to `[]`.
Work:
- Change signature to `curate_memory(self, current_memories, recent_transcript,
  deep_memories=None, archive_candidates=None, archive_age_days=7)`.
  `deep_memories` is the deep store's `get_all_memories()` dict (or None);
  `archive_candidates` is a precomputed structure (from step 2.3) listing hot
  entries older than the configured age. `archive_age_days` is the configured
  threshold, used ONLY to render accurate prompt text — the caller (2.3) does
  the actual age filtering. All default for backward compat with existing
  tests/callers.
- In the prompt: add a `[Deep Memory Entries (archived, read-only reference)]`
  section with the deep summary JSON (same compact shape as current_summary),
  plus an `[Archive Candidates (hot entries older than N days)]` section where
  N is `archive_age_days`. If `archive_candidates` is None/empty, OMIT the
  candidates section entirely (do not show an empty section). Add instruction
  text: entries older than N days SHOULD move to the archive unless still
  actively referenced; NEVER re-add a fact already present in deep memory
  (return it in promote lists instead if it resurfaced); resurfacing archived
  topics go in promote lists.
- Extend the demanded JSON with SIX new keys:
  `facts_to_archive`, `dynamics_to_archive`, `jokes_to_archive` (each a list
  of topic/title strings matching hot entries) and `facts_to_promote`,
  `dynamics_to_promote`, `jokes_to_promote` (list of topic/title strings
  matching deep entries).
- Add all six keys to `default_result` as `[]` so old-model outputs and
  parse failures degrade safely.
- Keep the existing CRITICAL FORGETTING RULE untouched.
Tests: `tests/test_llm.py`: mock the model response to include archive/promote
keys and assert they parse through; assert a response WITHOUT the new keys
yields empty lists (backward compat). Follow the existing curation test
pattern in that file (mock `_call_genai` or `_call_openai_compatible` —
check what the existing tests patch).

### 2.3 Archive/promote execution in `_curate_worker`
File: `handlers.py`, function `_curate_worker` (handlers.py:873).
Background: the worker builds `current_memories` from hot memory
(handlers.py:885), calls `self.llm.curate_memory(current_memories,
transcript)` (handlers.py:890-892), backs up hot memory (handlers.py:901),
then applies add/update/discard lists and returns a summary dict.
`self.memory` is hot; `self.deep_memory` is deep (added in Phase 1).
Work:
- Before the LLM call: read `archive_days =
  self.state.get_deep_archive_days()` (step 1.3; default 7). If `archive_days
  <= 0`, set `archive_candidates = None` (age candidacy disabled) and pass
  `archive_age_days=0` through for prompt rendering. Otherwise compute
  `archive_candidates` from hot entries whose age exceeds `archive_days`.
  Age basis: facts use `updated_at` if present else `created_at`;
  dynamics/jokes use `created_at`. Parse ISO `...Z` timestamps with
  `datetime.fromisoformat(s.replace("Z", "+00:00"))`; unparseable or missing
  timestamps => NOT a candidate (never archive what you cannot date).
  Structure: `{"facts": [topic...], "dynamics": [members-joined...],
  "jokes": [title...]}` — reuse compact shapes, not full entries.
- Fetch `deep_current = self.deep_memory.get_all_memories()` and pass
  `deep_memories`, `archive_candidates`, and `archive_age_days=archive_days`
  into `curate_memory`.
- After the existing discard blocks: `self.deep_memory.create_backup()`, then
  apply the six new lists:
  - For each target in `facts_to_archive`: `entry =
    self.memory.pop_memory(target)`; if entry: `self.deep_memory.append_entry(
    "memories", entry)`. Count `archived_facts`. (Append-then-pop ordering
    from 2.1 is impossible across two managers without holding both locks;
    pop-then-append is used instead — the crash window loses one entry
    maximum. Document this tradeoff in a comment. Do NOT hold both locks.)
  - Same pattern for `dynamics_to_archive` (`pop_dynamic`/`"dynamics"`) and
    `jokes_to_archive` (`pop_inside_joke`/`"inside_jokes"`).
  - For each target in `facts_to_promote`: pop from deep, append to hot.
    Count `promoted_facts`. Same for dynamics/jokes.
- Add the six counters to the returned summary dict.
- Forced-eviction fallback: AFTER applying curator lists, if hot fact count
  still exceeds 300 (hard cap against pathological growth — count-based,
  intentionally much larger than any healthy hot set), pop oldest facts by
  age (same age basis) into deep until count is 300. Log a warning when this
  fires. Dynamics/jokes have no forced eviction (their volume is low).
- IMPORTANT: after any move in either direction, the embedding sidecar is
  stale. Do NOT handle embeddings here — Phase 3 adds a single
  `self._refresh_deep_embeddings()` hook call at the end of the move block;
  leave a `# Phase 3 hook` comment where it goes. (If Phase 3 lands in the
  same run, wire it then.)
Tests: `tests/test_handlers.py`: seed hot memory with entries dated 8 days ago,
3 days ago, and today; mock `curate_memory` to echo verdicts. Case A (default
7): 8-day entry is listed in the `archive_candidates` arg passed to the mock,
3-day/today are not. Case B (`set_deep_archive_days(2)`): both 8-day and 3-day
entries are candidates. Case C (`set_deep_archive_days(0)`): candidates arg is
None. Case D (moves): mock returns archive/promote verdicts; run
`_curate_worker(blocking=True)`; assert old entry moved to deep with timestamps
preserved, today's entry stayed, promoted deep entry returned to hot, summary
counters correct. Check how existing curation tests construct BotHandlers
(mocks needed for params/state/llm — state MUST be a real StateManager on a
temp file, not a Mock, so the new getter works) and mirror them; pass a
temp-file deep MemoryManager.

### 2.4 Forget runs against both stores
Files: `handlers.py` (forget dispatch), `llm.py` `curate_forget` (llm.py:1676).
Background: group forget requests go through `curate_forget` (which sees
`[Current Memory Entries]`) producing a forget spec consumed by
`MemoryManager.apply_forget` (memory.py:420). `apply_forget` with
`clear_all` wipes one store. Memhistory scrubbing of forgotten content is
separate and unchanged.
Work:
- Find the caller of `curate_forget` in handlers.py (grep `curate_forget`).
  Change it to pass COMBINED hot+deep entries as `current_memories`
  (concatenate the three lists), so the LLM can target archived items.
- Change the `apply_forget` call site to apply the returned spec to BOTH
  `self.memory` and `self.deep_memory`, summing the four counters.
  `clear_all` therefore wipes both stores. Add `create_backup()` on deep
  before applying (hot already backs up — verify at the call site).
- If the forget path sends the user a confirmation listing removed counts,
  keep the message shape but use summed counts.
- Leave `MemoryManager.apply_forget` itself UNCHANGED (single-store
  semantics; the caller loops over stores).
Tests: seed hot + deep with the same forgettable topic; mock `curate_forget`
to discard it; assert both stores no longer contain it and counts sum.

Phase 2 verification gate: `pytest` green; manual scenario — seed an entry older
than the configured age (default 7 days), run `/curate`, confirm it moved to
`pletykas-deepmemory.json` with timestamps intact; set `/archive_age 0`,
confirm age candidacy stops; issue a forget for an archived topic, confirm both
stores clean.

---

## Phase 3: Embedding retrieval (llm.py embed method, new deepmem.py retriever, evaluation wiring)

Goal: each conversational evaluation auto-injects top-3 relevant deep entries.
No model cooperation needed.

### 3.1 `LLMClient.embed_texts` via OpenAI-compatible `/embeddings`
File: `llm.py`.
Background: `_call_openai_compatible` (llm.py:342) shows the transport pattern:
`session = await self._get_session()` (llm.py:277, thread-local aiohttp
session), `url = api_base.rstrip("/") + "/chat/completions"`, Bearer auth
header, `session.post`, debug banners via `_log_debug_payload`, non-200 =>
raise RuntimeError. Embeddings reuse all of this with a different path and
payload. Model/key/base/dim come from step 1.5 params
(`model_embed_name`, `effective_embed_api_key`, `effective_embed_api_base`,
`model_embed_dim`).
Work:
- Add `async def _call_openai_embeddings(self, api_base: str, api_key: str,
  model_name: str, texts: List[str], dim: int = 0) -> List[List[float]]`:
  - `url = api_base.rstrip("/") + "/embeddings"`. If `api_base` is empty after
    fallback resolution => raise RuntimeError("no embed api_base configured")
    (caller converts to `[]`; fail fast here, not with a malformed URL).
  - Payload: `{"model": model_name, "input": texts}`; add `"dimensions": dim`
    ONLY when `dim > 0`. This is the OpenAI embeddings request shape, which
    OpenRouter's `POST /api/v1/embeddings` accepts verbatim
    (https://openrouter.ai/docs/api/api-reference/embeddings/submit-an-embedding-request).
  - Headers: `Authorization: Bearer {api_key}`, `Content-Type: application/json`.
  - Timeout: pass `timeout=aiohttp.ClientTimeout(total=5)` on the
    `session.post` call (per-request override of the 120s session default —
    do NOT change `_get_session`). aiohttp raises `asyncio.TimeoutError` on
    expiry; it propagates to `embed_texts`, which catches it into `[]`.
    Add module constant `DEEPMEM_EMBED_TIMEOUT_SEC = 5` and use it here.
  - Debug request banner mirrors `_call_openai_compatible` but TRUNCATES: log
    model + text count + first-80-chars preview per text (never full texts).
  - Response: parse JSON, extract `data[i]["embedding"]` sorted by `index`,
    coerce each element with `float(v)`. Validate: `len(vectors) ==
    len(texts)` and every vector non-empty, else raise RuntimeError
    ("embedding count/dimension mismatch").
  - Non-200 status => read text, debug-log it, raise RuntimeError with status
    + body (same style as `_call_openai_compatible`). NO retry logic, NO
    max_tokens-style recovery — embeddings has no equivalent knobs.
  - Debug response banner: vector count + dimension only
    (e.g. `"3 vectors x 768 dims"`). NEVER dump float arrays to stdout.
- Add `async def embed_texts(self, texts: List[str]) -> List[List[float]]`:
  - Return `[]` immediately for empty input.
  - Resolve `model = self.params.model_embed_name or
    "google/gemini-embedding-2"` (hardcoded safety default if config is
    somehow blank — the two literals MUST match; comment this), `key =
    self.params.effective_embed_api_key`, `base =
    self.params.effective_embed_api_base`, `dim = self.params.model_embed_dim`.
  - `try: return await self._call_openai_embeddings(base, key, model, texts,
    dim)` / `except Exception as e: logger.warning("Deep-memory embedding
    failed (model=%s): %s", model, e); return []`. Never raise.
- No batching: one call with all texts (evaluations send 1 query; curation
  syncs send the missing batch at once).
Tests: `tests/test_llm.py` (NO network — patch `LLMClient._get_session` to
return a fake session whose `post` yields canned `/embeddings` JSON):
- Happy path: 2 texts => 2 float vectors, sorted by `index` even if `data`
  arrives out of order; `dimensions` absent from payload when dim=0, present
  when dim=768.
- Count mismatch (1 vector for 2 texts) => embed_texts returns `[]`, not raise.
- HTTP 500 => `[]`, not raise. Empty input => `[]` without any HTTP call.
- Empty api_base (both embed + primary blank — construct Params manually, do
  NOT call parse which requires keys) => `[]`, not raise.
- Timeout: fake session whose `post` raises `asyncio.TimeoutError` =>
  `embed_texts` returns `[]`; assert the `post` call received
  `timeout.total == 5`.

### 3.2 New module `deepmem.py`: sidecar cache + cosine retrieval
New file: `deepmem.py`. Pure logic + file IO, no Telegram/LLM imports (takes
an embed callable for testability).
Background: sidecar decision — `pletykas-deepmemory-embeddings.json` holds
`{"model": str, "vectors": {content_hash: [floats]}}`. Content hash keys the
vector because entries are ID-free by invariant. Deep JSON schema stays
identical to hot JSON.
Work:
- `entry_text(entry: Dict, section: str) -> str`: canonical embeddable text.
  `memories` => `f"{topic}: {content}"`; `dynamics` =>
  `f"{' & '.join(members)}: {relation}"`; `inside_jokes` =>
  `f"{title}: {context}"`. Strip whitespace; empty => `""`.
- `entry_hash(entry: Dict, section: str) -> str`: sha256 hex of
  `section + "\0" + entry_text(entry, section) + "\0" +
  entry.get("created_at","")`. Including `created_at` disambiguates duplicate
  texts archived at different times.
- `class DeepMemoryIndex`:
  - `__init__(sidecar_path="pletykas-deepmemory-embeddings.json",
    embed_model="google/gemini-embedding-2")`. The default MUST match the
    params default (`model_embed_name`, step 1.5) — comment this. In
    production, handlers (3.3) passes `self.params.model_embed_name`
    explicitly so a config change invalidates stale vectors via the
    model-mismatch rule below; the literal default only serves tests and
    standalone use. Do NOT import params/llm into deepmem.py (keep it
    dependency-free).
  - Internal: `self._lock = threading.RLock()`, `self._vectors: Dict[str,
    List[float]]`, `self._model = ""`.
  - `load()`: read sidecar if exists; if `data.get("model") !=
    self._embed_model`, DISCARD all vectors (model mismatch => recompute)
    and log warning. Malformed file => discard + warning, never crash.
  - `save()`: atomic write (tempfile + os.replace, same pattern as
    MemoryManager.save). NO backup rotation (cache is regenerable —
    document why in a comment).
  - `sync(entries: List[Tuple[section, entry]], embed_fn) -> int`:
    `embed_fn` is a SYNC callable taking List[str] and returning
    List[List[float]] (the caller bridges async embed_texts via
    `asyncio.run` or a loop — see 3.4; deepmem stays sync for simplicity).
    Compute hashes for all current deep entries; drop vectors whose hash is
    gone; collect texts for missing hashes; call `embed_fn` ONCE with all
    missing texts; store results; `save()`; return count of newly embedded.
    If `embed_fn` returns wrong count or raises => log warning, keep old
    state, return 0.
  - `query(query_vector: List[float], entries: List[Tuple[section, entry]],
    top_k=3, min_score=0.55) -> List[Tuple[float, section, entry]]`:
    cosine similarity of query vs each stored vector (skip entries without
    vectors); return top_k with score >= min_score, descending. Pure
    function of inputs — no IO. Guard zero-norm vectors (skip them).
  - `format_hits(hits) -> str`: render as:
    ```
    [Recalled Deep Memory]
    - (Topic): content
    ...
    ```
    using the same `(<label>): <text>` line style as
    `MemoryManager.format_for_context`. Empty hits => `""` (caller omits the
    section entirely).
- Threshold note: 0.55 is a starting default for cosine similarity on short
  queries (model-agnostic starting point); log scores in debug so it can be
  tuned from real traffic. Make it a module constant `DEEPMEM_MIN_SCORE`.
  `top_k=3` module constant `DEEPMEM_TOP_K`.
Tests: new `tests/test_deepmem.py`: hash stability + changes on edit;
cosine ranking picks the right entry from canned vectors; min_score filters;
zero-norm safe; sync adds/drops vectors with a fake embed_fn; model-mismatch
discards; malformed file tolerated; save/load roundtrip. All offline.

### 3.3 Retrieval entry point used by handlers
File: `handlers.py` (new method on BotHandlers) + `deepmem.py` reuse.
Background: conversational evaluation happens in `_execute_evaluation`
(handlers.py:1172), which builds `memory_ctx` (handlers.py:1194) and calls
`self.llm.evaluate_and_reply(...)` (handlers.py:1253) for the non-photo path
and `self.llm.describe_and_reply_image(...)` (handlers.py:1236) for the photo
path. V1 recall covers ONLY the non-photo path.
Work:
- In `BotHandlers.__init__`, create `self.deep_index =
  DeepMemoryIndex(embed_model=self.params.model_embed_name or
  "google/gemini-embedding-2")` (sidecar path default; derive from deep memory
  file path? NO — keep the fixed default filename for v1, document it). Call
  `self.deep_index.load()`. Wrap in try/except that logs and continues with an
  empty index (never break startup). Passing the configured model name (not
  the deepmem.py literal) is what makes a future model swap invalidate stale
  vectors via the model-mismatch rule. NOTE: existing tests construct
  BotHandlers with Mock params — `self.params.model_embed_name` on a plain
  Mock returns a Mock, NOT a str. Guard with `getattr(self.params,
  "model_embed_name", "") or "google/gemini-embedding-2"` and coerce: `name =
  ...; model = name if isinstance(name, str) and name.strip() else
  "google/gemini-embedding-2"`. Comment why the coercion exists (Mock-safety).
- Add `async def _recall_deep_memory(self, query_text: str) -> str`:
  - If `not query_text.strip()` => return `""`.
  - Gather deep entries: `deep = self.deep_memory.get_all_memories()`;
    flatten to `[(section, entry), ...]` for all three sections. If empty =>
    return `""`.
  - `vectors = await self.llm.embed_texts([query_text])`; if empty => return
    `""` (embedding outage => silent no-injection + the warning already logged
    in embed_texts).
  - `hits = self.deep_index.query(vectors[0], flat_entries)`; if empty =>
    return `""`.
  - Debug: if `self.state.is_debug_mode()`, write a stdout banner via
    `self._log_debug_group_msg("DEEPMEM RECALL", f"Query:
    {query_text[:200]}\nHits: " + ", ".join(f"{score:.2f}:{label}"))`
    (labels = topic/title/members, NOT full contents — contents already flow
    through normal debug payloads).
  - Return `DeepMemoryIndex.format_hits(hits)` (staticmethod or module
    function — your choice, keep it importable for tests).
  - Whole body wrapped in try/except returning `""` on unexpected error with
    `logger.warning`. Retrieval NEVER raises into the evaluation path.
- Query text construction (in `_execute_evaluation`, non-photo branch only):
  `query_text` = trigger message text + previous 1 message text for context.
  Implementation: `history = self.state.get_chat_history()`; find
  trigger_entry by `trigger_msg_id` (already located at handlers.py:1199-1203);
  take up to 2 texts: trigger text + the immediately preceding history item
  text (if any); join with `" / "`, strip `"[Photo...]"` markers (recall is
  text-path only but history may contain photo descriptions — strip bracket
  tags via regex `r"\[Photo[^\]]*\]"`). Cap at 500 chars.
Tests: `tests/test_handlers.py`: with seeded deep entries + mocked
`embed_texts` returning a canned vector and a pre-populated index, assert
`_recall_deep_memory` returns formatted hits; assert embedding failure =>
`""`; assert empty deep store => `""` without calling embed_texts.

### 3.4 Wire recall into `_execute_evaluation` + curation hook + backfill
Files: `handlers.py`, `main.py`.
Work:
- In `_execute_evaluation` (handlers.py:1172), in the NON-PHOTO branch
  (handlers.py:1251-1260): before calling `evaluate_and_reply`, compute
  `recalled = await self._recall_deep_memory(query_text)` (query per 3.3);
  if non-empty, append it to `memory_ctx` as
  `memory_ctx = memory_ctx + "\n\n" + recalled`. Photo branch untouched.
- Curation hook: at the end of the move block in `_curate_worker` (the
  `# Phase 3 hook` comment from 2.3), call `self._refresh_deep_embeddings()`.
  Implement `_refresh_deep_embeddings(self) -> None` (sync — worker is
  already a background thread): flatten deep entries; define
  `embed_fn(texts)` bridging to async via `asyncio.run(
  self.llm.embed_texts(texts))` — WAIT: `_curate_worker` creates and closes
  its own event loop (handlers.py:887-898) BEFORE the move block, so by hook
  time no loop is running in that thread and `asyncio.run` is safe. Add an
  assertive comment explaining why `asyncio.run` is safe here (no running
  loop in worker thread at this point). Wrap the whole refresh in
  try/except logging warning only.
- First-boot backfill: in `main.py post_init` (after handlers exist), spawn
  a daemon thread running `handlers._refresh_deep_embeddings()` ONCE so
  pre-existing deep entries get vectors without blocking startup. Log info
  when done. Guard with try/except.
- Spontaneous + scheduled paths (`trigger_spontaneous_message` handlers.py:512,
  `_scheduled_reply_callback` handlers.py:660): NO recall in v1 (no trigger
  query text; keep scope tight). Document this exclusion in a comment at both
  `format_for_context` call sites? NO — do not touch those call sites at all.
  Only document here in the plan.
Tests: extend the 2.3 worker test: after moves, assert `_refresh_deep_embeddings`
was invoked (mock it) — OR integration-style with mocked embed_texts asserting
sidecar file gained vectors. Prefer the mock-invocation assertion (fast,
deterministic).

### 3.5 Conversational chat/vision timeout 10s (curation + images exempt)
File: `llm.py`.
Background: `_call_openai_compatible` (llm.py:342) is shared by SIX call
sites: `evaluate_and_reply` (llm.py:967), `_call_vision_model` (llm.py:1081),
`generate_spontaneous_message` (llm.py:1488), `generate_scheduled_reply`
(llm.py:1563), `curate_memory` (llm.py:1638), `curate_forget` (llm.py:1756).
Both `session.post` calls inside it (llm.py:380 initial, llm.py:408 retry)
inherit the 120s session default from `_get_session` (llm.py:282). Image
generation/editing (llm.py:1298/1335/1357) uses the same session and MUST keep
120s. Curation/forget calls emit up to 200k-token JSON and MUST keep 120s.
Work:
- Add module constant `CHAT_TIMEOUT_SEC = 10`.
- Add keyword param `timeout_sec: Optional[float] = None` to
  `_call_openai_compatible`. When not None, pass
  `timeout=aiohttp.ClientTimeout(total=timeout_sec)` on BOTH `session.post`
  calls (initial + retry). When None, pass nothing (120s session default
  applies). Do NOT change `_get_session` or the session default.
- Pass `timeout_sec=CHAT_TIMEOUT_SEC` at FOUR call sites: `evaluate_and_reply`
  (llm.py:967), `_call_vision_model` (llm.py:1081),
  `generate_spontaneous_message` (llm.py:1488), `generate_scheduled_reply`
  (llm.py:1563). Leave `curate_memory` and `curate_forget` WITHOUT the arg
  (exempt — comment each with `# No timeout: large JSON output needs headroom`).
- Timeout behavior on the interactive path is UNCHANGED code-wise:
  `asyncio.TimeoutError` propagates from `evaluate_and_reply` into
  `_execute_evaluation`'s existing try/except (handlers.py:1574), which logs
  and sends the "tangled up" fallback on direct triggers only. No handler
  changes in this step. Note for the implementer: a 10s timeout on a direct
  trigger now produces a user-visible fallback message — this is intended
  (fail fast + tell the user) per user decision.
- GenAI-SDK path (`_call_genai`): NO timeout change (SDK manages its own
  deadlines; out of scope).
Tests: `tests/test_llm.py` (patch `_get_session` with a fake session):
- conversational call passes `timeout.total == 10` on `post`; curation call
  passes NO timeout kwarg. Assert both by inspecting fake `post` kwargs.
- TimeoutError from `post` propagates out of `_call_openai_compatible`
  (do NOT swallow here — callers own the degradation policy).

Phase 3 verification gate: `pytest` green; live check with debug ON — send a
group message referencing an archived topic, observe `DEEPMEM RECALL` banner
with scores and confirm the reply uses the archived fact; break the embeddings
endpoint (bad base URL or key) and confirm replies still flow with no
injection; stall the chat endpoint (>10s delay) and confirm a fast failure +
fallback message on direct trigger, silence on passive entry; verify the
request payload matches OpenAI `/embeddings` shape so a
future OpenRouter switch needs only `MODEL_EMBED_API_BASE` +
`MODEL_EMBED_API_KEY`.

---

## Phase 4: Admin surface (handlers.py commands, HELP_MESSAGE)

Goal: `/deepmemories`, `/deepmemories load`, `/cancel` integration, `/status`
counts, help text.

### 4.1 `/deepmemories` download + `/deepmemories load` upload
File: `handlers.py`.
Background: `cmd_memories` (handlers.py:2500) + `_process_memory_upload`
(handlers.py:2443) implement hot-memory download/upload with
`_waiting_for_memory_upload` set (handlers.py:278) consumed by the document
handler (find where `_waiting_for_memory_upload` is checked — likely in
`on_message` document branch — and mirror it). `validate_memory_dict` from
memory.py validates; upload does backup + replace + reload.
Work:
- Add `self._waiting_for_deepmemory_upload: set[int]` in `__init__`.
- Refactor: extract the shared upload-core of `_process_memory_upload` into
  `_process_memory_upload_to_store(update, context, document, store:
  MemoryManager, waiting_set: set, label: str)` — parameters select hot vs
  deep. Keep `_process_memory_upload` as a thin wrapper calling it with
  `(self.memory, self._waiting_for_memory_upload, "memories")` so existing
  behavior/tests are untouched. (Clean cutover per repo contract: migrate
  the one caller, no shim left behind — the wrapper IS the hot path, not a
  shim.)
- Add `_process_deepmemory_upload` wrapper with `(self.deep_memory,
  self._waiting_for_deepmemory_upload, "deep memories")`, then trigger
  `threading.Thread(target=self._refresh_deep_embeddings, daemon=True).start()`
  after a successful deep load (vectors for the new content, background).
- Add `cmd_deepmemories` mirroring `cmd_memories` exactly (subcommands:
  none => download deep file; `load` => expect upload; else usage error),
  operating on `self.deep_memory` / `_waiting_for_deepmemory_upload`.
  Filenames/captions say "Deep Memories".
- Register `CommandHandler("deepmemories", self.cmd_deepmemories)` next to
  the memories registration (handlers.py:2608).
- Extend the document-receiving branch + `cmd_cancel` (handlers.py:2549) to
  handle the deep waiting set identically.
Tests: `tests/test_handlers.py`: mirror existing memories upload/download
tests for the deep variants (valid JSON loads into deep store; invalid JSON
rejected; cancel clears deep waiting set). Check existing test helper style
first.

### 4.2 `/archive_age` tunable command
File: `handlers.py`.
Background: numeric tunables follow the `cmd_cooldown` pattern
(handlers.py:1905-1922): admin-only + private-chat-only guards, no args =>
report current value, one int arg => set + confirm (re-reading via the getter
to show the clamped value), ValueError => usage error. Registration lives with
the other CommandHandlers (handlers.py:2594-2613); HELP_MESSAGE documents
commands (handlers.py:38-68).
Work:
- Add `cmd_archive_age`: no args => `📦 Deep archive age is currently: <b>N
  days</b>` (+ ` (disabled)` note when 0). One arg => parse int,
  `self.state.set_deep_archive_days(val)`, confirm `✅ Deep archive age
  updated to: <b>N days</b>` (or `disabled` when 0). ValueError => `❌ Please
  specify a non-negative integer in days (0 disables age-based archiving).`
- Register `CommandHandler(["archive_age", "archiveage"], self.cmd_archive_age)`
  next to the other tunable registrations.
- HELP_MESSAGE: under Memory Management, add `/archive_age [days] - View or
  adjust hot-memory archive age in days (0 disables)`.
- Find `cmd_status`; add `Archive age: N days` (or `disabled`) next to the
  existing memory line, matching its formatting style.
Tests: `tests/test_handlers.py`: no-args reports default 7; set to 3 persists
via the real StateManager getter; set to 0 reports disabled; non-integer =>
usage error; non-admin / group-chat invocations ignored (mirror existing
command-guard tests).

### 4.3 `/status` deep counts + help text
File: `handlers.py`.
Background: `cmd_status` (registered handlers.py:2594) reports status;
HELP_MESSAGE (handlers.py:38-68) documents commands under `<b>Memory
Management</b>`.
Work:
- Find `cmd_status`; add a line reporting deep memory counts (`Deep memory: X
  facts, Y dynamics, Z jokes`) next to the Archive age line from 4.2 (match its
  formatting style). Also report sidecar vector count
  (`Deep vectors: N`) from `len(self.deep_index._vectors)` — expose a
  `count_vectors()` method on DeepMemoryIndex instead of touching the
  private (add it in this step, trivial).
- HELP_MESSAGE: under Memory Management, add
  `/deepmemories` + `/deepmemories load` lines mirroring the `/memories` lines.
Tests: assert `/status` reply contains `Deep memory:`; assert HELP_MESSAGE
contains `/deepmemories`. (These are the rare allowed wording tests — admin
contract surface. Keep them to substring checks.)

Phase 4 verification gate: `pytest` green; in a private admin chat:
`/deepmemories` downloads the file, `/deepmemories load` + upload roundtrips,
`/status` shows both tiers + archive age, `/archive_age 3` then `/curate` uses
3-day candidacy, `/cancel` aborts a pending deep upload.

---

## Phase 5: Docs, AGENTS.md invariants, full verification

Goal: repo docs reflect the new tier; everything proven end to end.

### 5.1 Update AGENTS.md + README if it documents memory/commands
Files: `AGENTS.md`, `README.md` (check whether README lists commands/memory).
Work:
- AGENTS.md section 1: add invariant 11 (two-tier memory: adjustable archive
  age via `deep_archive_days` default 7 / `/archive_age`, OpenAI-compatible
  `/embeddings` auto-retrieval defaulting to `google/gemini-embedding-2`
  (OpenRouter-ready via `MODEL_EMBED_*`), sidecar cache, degrade-to-no-injection,
  vision excluded in v1).
- AGENTS.md section 2: add `deepmem.py` + `tests/test_deepmem.py` to the tree.
- AGENTS.md section 3: document `pletykas-deepmemory.json` (same schema as
  hot) + `pletykas-deepmemory-embeddings.json` (`{model, vectors}` cache,
  no rotation, regenerable) + `updated_at` on facts + `deep_archive_days`
  state key in the state JSON schema block.
- If README documents `/memories` or memory behavior, mirror the deep
  additions briefly.
Tests: none. Proofread the edited sections.

### 5.2 End-to-end verification
Work (do NOT write permanent tests for these — throwaway scripts + live runs):
- Full suite: `.venv/bin/pytest -v` green.
- Throwaway script: seed hot with facts of mixed ages, run `_curate_worker`
  with a mocked curator verdict, assert moves + sidecar vectors appear; then
  call `_recall_deep_memory` with a Hungarian query (mocked embedding close
  to the archived entry's vector) and assert injection text. Delete script
  after.
- Live smoke (requires bot token + group; use the live-input protocol above):
  start bot, ask the user to send `/curate` in the private admin chat with old
  entries present, confirm archive; ask the user to send a Hungarian message
  about an archived topic in the group with `/debug on`, confirm `DEEPMEM
  RECALL` banner + grounded reply; ask the user to run the `/deepmemories`
  roundtrip and `/status` in the private admin chat, confirm counts.
- Report: which of the above were actually exercised (for any live check not
  run even via user input, state explicitly that it was NOT run and what
  substituted).

Phase 5 verification gate: docs merged, suite green, verification report
written in the final message (not in the repo).
