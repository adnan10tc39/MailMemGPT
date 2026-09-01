# MailRecallAI Re-Run Experiment Harness — Design Specification

## Provider adaptation (2026-08-31)

The OpenAI account has no credits; the harness now runs on Groq + local
embeddings. Everything below this section is otherwise unchanged (tau/delta/
k/B/history-window and all interfaces are as pinned); read model/embedding
references in the original text through this mapping:

- **Chat**: Groq's OpenAI-compatible endpoint (`base_url
  https://api.groq.com/openai/v1`, key `GROQ_API_KEY` in the repo `.env`,
  openai python SDK). `chat_model = openai/gpt-oss-120b` (replaces gpt-4o),
  `summary_model = openai/gpt-oss-20b` (replaces gpt-3.5-turbo). These are
  reasoning models: the wrapper passes `reasoning_effort="low"`, discards the
  reasoning field (`content` = final message only), and floors `max_tokens`
  at 256 so a capped call can never return reasoning-only. The wrapper still
  accepts the legacy `functions`/`function_call` arguments and function-role
  messages; it translates them to the modern `tools`/`tool_calls` API at the
  boundary (callers unchanged).
- **Embeddings**: local sentence-transformers `BAAI/bge-base-en-v1.5`
  (768-d, `cfg.embed_dim`, device `cfg.embed_device` = cuda), lazy-loaded
  once per process, `normalize_embeddings=True` so cosine = dot. No network,
  tokens=0, zero cost. Store DDL sizes `vector(embed_dim)` from config and
  recreates `mr_fewshot`/`mr_memory` if an existing column dimension differs.
- **Rate limits (Groq free tier)**: 8000 tokens/min, 1000 requests/day. The
  wrapper throttles with a sliding 60 s window targeting 7000 tokens/min
  (estimate = prompt chars/3.5 + max_tokens, settled to actual usage), honors
  429 retry-after / "try again in Xs" hints within the 5-try ladder, and
  raises `RuntimeError("GROQ_DAILY_LIMIT")` on daily-quota 429s.
- **Pricing**: gpt-oss-120b $0.15/1M input, $0.60/1M output; gpt-oss-20b
  $0.075/$0.30; embeddings $0. The $25 cost-cap logic is unchanged.
- **Live smoke test**: `python3 -m experiments.run.livesmoke` (one JSON chat,
  one tool round-trip through the legacy interface, local embedding sanity).

Purpose: regenerate all triage/efficiency results for the EAAI major revision with full
per-email logging, so confusion matrices, per-class precision/recall/F1, macro-F1,
McNemar tests, Wilson/bootstrap CIs, latency breakdowns, ablations, and sensitivity
analyses can be computed honestly. The harness implements the configuration STATED in
the paper (authoritative per author decision): PostgreSQL + pgvector, cosine similarity,
triage threshold tau=0.37, retrieval top-k=5, retrieval threshold delta=0.70, prompt
budget B=8000 tokens, recent SQL history window = 10 pairs, gpt-4o (temperature 0.0),
summarization gpt-3.5-turbo, embeddings text-embedding-3-small (1536-d),
JSONL+gzip archival tier.

## Runtime environment
- Python 3.11 (anaconda, `python3`), packages available: openai, psycopg2-binary, pgvector,
  tiktoken, python-dotenv, numpy/scipy (install if missing).
- PostgreSQL: docker container `mailrecall-pg`, host=localhost port=5433 db=mailrecall
  user=postgres password=mailrecall. pgvector 0.8.6 enabled (`CREATE EXTENSION vector`).
- OpenAI key: loaded from `/media/adnan/DATA/Agentic-LongTerm-Memory/.env` via
  `load_dotenv(<explicit path>)` (find_dotenv is broken under heredocs — ALWAYS pass the path).
- All experiment code lives under `/media/adnan/DATA/Agentic-LongTerm-Memory/experiments/`.
- All artifacts (datasets, logs, results) under `experiments/data/` and `experiments/results/`.

## Package layout (implement exactly these files)
```
experiments/
  __init__.py
  common/
    __init__.py
    exp_config.py      # ExpConfig dataclass + phase presets (agent A)
    llm.py             # OpenAI wrapper (agent A)
    store.py           # Postgres+pgvector store + archival writer (agent A)
  dataset/
    __init__.py
    generate_synthetic.py   # (agent B)
    prepare_enron.py        # (agent B)
  agent/
    __init__.py
    triage.py          # (agent C)
    pipeline.py        # (agent C)
    respond.py         # (agent C)
  run/
    __init__.py
    run_phase.py       # (agent D)
    ablations.py       # (agent D)
  analysis/
    __init__.py
    metrics.py         # (agent D)
```
Run everything as modules from the repo root:
`cd /media/adnan/DATA/Agentic-LongTerm-Memory && python3 -m experiments.run.run_phase ...`

## Shared interfaces (PINNED — do not deviate)

### common/exp_config.py
```python
@dataclass
class ExpConfig:
    phase: str                  # "p1" | "p2" | "p3"
    run_id: str                 # e.g. "p3_synthetic_seed13"
    dataset_path: str           # jsonl file
    # paper-authoritative parameters (defaults shown):
    triage_tau: float = 0.37    # few-shot semantic triage threshold (max cosine sim)
    triage_topn: int = 3        # neighbours consulted per class collection
    retrieval_k: int = 5        # stage-2 top-k
    retrieval_delta: float = 0.70  # stage-2 min cosine similarity
    prompt_budget: int = 8000   # B, tokens
    history_window: int = 10    # recent SQL pairs loaded in stage 1
    chat_model: str = "gpt-4o"
    summary_model: str = "gpt-3.5-turbo"
    embed_model: str = "text-embedding-3-small"
    temperature: float = 0.0
    max_output_tokens: int = 2000
    seed: int = 13
    generate_responses: bool = True   # False => triage + prompt-assembly metrics only
    # ablation switches (all True for standard p3):
    enable_dedup: bool = True
    enable_budget: bool = True
    enable_sql_history: bool = True
    enable_vector_retrieval: bool = True   # p3 only
    naive_rag: bool = False     # RAG baseline: retrieval concat, no dedup/budget/structure
def preset(phase: str, **overrides) -> ExpConfig
```
Phase presets: p1 = no tools, no vector retrieval (SQL hot memory only);
p2 = p1 + reactive function calling (tools available to LLM, incl. search_similar_emails);
p3 = p2 operational tools + proactive 5-stage pipeline (automatic retrieval; memory-search
tool calls should become unnecessary).

### common/llm.py
```python
class LLM:
    def __init__(self, cfg: ExpConfig): ...
    def chat(self, messages, functions=None, function_call=None,
             model=None, temperature=None, max_tokens=None) -> ChatResult
    # ChatResult: .content, .function_call (name, arguments dict or None),
    #             .prompt_tokens, .completion_tokens, .latency_ms
    def embed(self, texts: list[str]) -> EmbedResult  # .vectors, .latency_ms, .tokens
    def count_tokens(self, text: str) -> int          # tiktoken o200k_base (gpt-4o)
    usage_totals: dict  # cumulative {prompt_tokens, completion_tokens, embed_tokens, cost_usd}
```
Retry with exponential backoff (5 tries) on rate-limit/transient errors. Cost accounting:
gpt-4o $2.50/1M in $10/1M out; gpt-3.5-turbo $0.50/$1.50; embeddings $0.02/1M.

### common/store.py
```python
class Store:
    def __init__(self, cfg: ExpConfig, namespace: str): ...
    # namespace isolates runs: every table gets a run-namespace column, every
    # vector collection is per-namespace; reset(namespace) drops that namespace's rows.
    def reset(self): ...
    # SQL hot memory:
    def save_interaction(self, email: dict, classification: str, confidence: float,
                         response_text: str | None): ...
    def recent_history(self, n_pairs: int) -> list[dict]        # chronological
    def get_user_profile(self) -> str; def get_rules(self) -> str
    def latest_summary(self) -> str | None
    def update_summary(self, text: str): ...
    # Warm memory (pgvector, cosine ops):
    def add_fewshot(self, label: str, doc: str, vec): ...
    def query_fewshot(self, label: str, vec, topn: int) -> list[(similarity, doc)]
    def add_memory(self, doc: str, vec, meta: dict): ...        # meta: email_id, thread_id, sender, subject, day
    def query_memory(self, vec, k: int) -> list[(similarity, doc, meta)]
    # Cold tier:
    def archive_email(self, email: dict, response_text: str | None): ...
    # appends to experiments/data/archive/<namespace>/day_<NNN>.jsonl.gz
```
Vector similarity MUST be cosine: `1 - (embedding <=> query)` with vector_cosine_ops index.
User profile + rules: seed one fixed profile ("John Doe, software engineering manager ...")
and the four triage rules from src/initialize + prepare_sqldb (adapt wording freely but fixed).

### Dataset JSONL schema (both synthetic and Enron)
```json
{"email_id": "syn-0001", "day": 3, "ts": "2026-01-05T09:14:00", "thread_id": "t-012",
 "sender": "Name <addr@dom>", "subject": "...", "body": "...",
 "label": "IGNORE|NOTIFY|RESPOND", "requires_context": false,
 "context_note": "optional free text on why context is needed"}
```
Sorted by (day, ts). Labels by construction for synthetic; for Enron: `"label_source"` field
("llm_prelabel" initially; the author will review — see prepare_enron.py notes).

### dataset/generate_synthetic.py (agent B)
Spec (from the paper's methodology): 500 emails over 12 weeks (days 1..84) of simulated
history for ONE assistant owner interacting with 10 fictional correspondents/personas;
200 IGNORE (spam/promotional/newsletters/auto-replies), 150 NOTIFY (system notifications,
FYI announcements, calendar/holiday notices, build/CI status), 150 RESPOND (questions,
meeting requests, follow-ups referencing earlier threads, approvals, urgent issues).
Structure requirements:
- ~40 multi-email threads; >=60 RESPOND emails must be context-dependent follow-ups
  (their correct handling references content from an earlier email in the corpus,
  mark requires_context=true, fill context_note naming the earlier email_id).
- Recurring patterns: weekly status meeting requests, an ongoing "Q1 platform migration"
  project, a partnership negotiation, invoice/payment thread, recruiting thread.
- Deterministic scenario skeleton from `random.Random(seed)`; bodies written by gpt-4o
  (temperature 0.7 for variety) from structured scenario prompts; NO placeholder text.
- Two-pass: pass 1 build scenario objects; pass 2 batch-generate bodies (5 emails/call,
  JSON mode). Save to experiments/data/synthetic_500.jsonl + a stats.json
  (class counts, thread stats, context-dependency counts).
- Also emit experiments/data/synthetic_pilot_20.jsonl = deterministic 20-email subsample
  (balanced, incl. >=4 context-dependent) for pilot runs.

### dataset/prepare_enron.py (agent B)
- Source tarball: experiments/data/enron/enron_mail_20150507.tar.gz (downloading in background;
  script must work once present; also support --maildir if already extracted).
- Sample 200 emails: from 4-6 custodians' inbox/ folders, preserve thread groups (same
  normalized subject within custodian), parse headers (From, Subject, Date, body text),
  strip quoted reply tails and headers/footers, drop <30-token and >800-token bodies,
  drop duplicates. Deterministic with seed.
- Pre-label with gpt-4o (JSON, temperature 0.0) using written labeling guidelines
  (include the guidelines verbatim as a module constant, mirroring the paper's three
  classes; treat mass announcements as NOTIFY, spam/irrelevant as IGNORE, direct
  questions/requests to the recipient as RESPOND).
- Output: experiments/data/enron_200.jsonl (label_source="llm_prelabel") AND
  experiments/data/enron_labels_for_review.csv (email_id, sender, subject, body 300-char
  preview, prelabel, blank column "author_label") for the author's manual verification.
  After review, --apply-review merges corrected labels and sets label_source="author_reviewed".

### agent/triage.py (agent C)
Implements the paper's Eq. (1): for each class c, s_c = MAX cosine similarity between the
incoming email embedding and the class's few-shot collection (query_fewshot topn=3, take max).
If max_c(s_c) >= tau: semantic classification -> argmax class, confidence = max similarity.
Else: LLM fallback conditioned on SQL rules + phase-dependent context:
  p1: rules + recent history (window pairs);
  p2: rules + recent history + tools available (search_similar_emails, get_user_info) —
      the LLM may call at most 2 tools during classification (reactive retrieval);
  p3: rules + recent history + proactively retrieved top-k>=delta semantic context
      (no tool decision needed).
Fallback output STRICT JSON {"classification": "...", "confidence": 0-1}; invalid -> RESPOND 0.5
(fail-safe, as in paper). Returns TriageResult: label, confidence, method
("semantic"|"llm_p1"|"llm_p2"|"llm_p3"), per-stage latencies, tokens, tool_calls list.
Few-shot examples: the 17 from src/initialize_few_shot_examples.py (5 IGNORE, 5 NOTIFY,
7 RESPOND) — copy them verbatim into a module constant.

### agent/pipeline.py (agent C)
Five stages; each returns its text block AND a StageStats(latency_ms, tokens_in_block,
items, removed_items, tokens_saved). Stage list:
1. sql_load: profile, rules, latest summary, recent history (history_window pairs,
   formatted "From/Subject/Body/Response", chronological).
2. vector_retrieve (p3 or naive_rag only): embed email (subject + body first 200 chars —
   per paper), query_memory top-k, keep similarity >= delta, format with metadata lines.
3. dedup (if enable_dedup): drop retrieved items whose email_id appears in the recent
   history window (primary key match); fallback when meta missing: normalized
   (sender-address, subject-with-Re/Fwd-stripped) pair match. Count tokens_saved.
4. budget (if enable_budget): count tokens of assembled sections with tiktoken; if over B,
   first summarize the retrieved-memories block via summary_model (target: fit budget);
   if still over, truncate lowest-priority sections in order (retrieved -> summary ->
   older history pairs) preserving profile/rules/current email. Record actions taken.
5. assemble: fixed section order (per paper Fig. 6): [role+profile] [agent instructions
   +rules] [tool descriptions (p2/p3)] [prior tool results] [summary of older interactions]
   [recent history] [retrieved semantic memories] [response instructions]. Return final
   system prompt string + PipelineStats.
naive_rag mode: skip dedup+budget+structure — system prompt = profile + rules + raw
concatenation of top-k retrieved docs + instructions.

### agent/respond.py (agent C)
Generates the reply for RESPOND-classified emails. Tools (p2/p3): search_similar_emails
(vector query, returns top cfg 3 docs), get_user_info, check_calendar_availability (stub
fixed slots), schedule_meeting (stub confirm), write_email_tool (stub formatter).
p1: no functions. p2/p3: OpenAI function-calling loop, loop-detection by
(name, sorted-args) signature repetition -> force direct answer; max 5 iterations.
Tool results appended into the "prior tool results" section, prompt rebuilt, reinvoke.
Returns RespondResult: text, tool_calls list, latencies, tokens.

### run/run_phase.py (agent D)
CLI: `python3 -m experiments.run.run_phase --phase p3 --dataset experiments/data/synthetic_500.jsonl
--namespace p3_syn --seed 13 [--no-responses] [--limit N] [--resume]`
Chronological loop over emails; for each: triage -> if RESPOND and generate_responses:
pipeline + respond; ALWAYS: save_interaction, archive_email, add_memory (embed
"email+response" doc with metadata) for RESPOND; summary update every 10 processed
RESPOND pairs (rolling, summary_model). Write one JSONL log line per email to
experiments/results/<namespace>/log.jsonl:
{email_id, day, label_true, label_pred, confidence, method, tool_calls, latencies:{triage_embed,
triage_query, triage_llm, sql_load, retrieve, dedup, budget, assemble, respond_llm, total},
tokens:{prompt, completion, final_prompt_tokens, budget_used_pct}, retrieval:{n_raw, n_kept,
n_deduped, sims:[...], tokens_saved}, response_len_chars, error:null|str}
Plus run_meta.json (config dump, git hash, started/finished, usage_totals).
--resume skips email_ids already in log. Errors: retry email once, then log error line and
continue (never crash the run). IMPORTANT: process emails strictly in dataset order so
memory accumulates chronologically; the store namespace must be reset at run start
(unless --resume) and re-seeded with few-shot examples + profile + rules.

### run/ablations.py (agent D)
Presets over the synthetic dataset, ALL with generate_responses=False for cost control
(prompt assembly still runs fully so token/dedup metrics are real; respond_llm skipped):
  ablation_no_dedup (p3, enable_dedup=False)
  ablation_no_budget (p3, enable_budget=False)
  ablation_vector_only (p3, enable_sql_history=False)
  ablation_sql_only == p1 (reuse main run)
  ablation_naive_rag (naive_rag=True)
Sensitivity sweeps (triage-only, cheap): tau in {0.27,0.32,0.37,0.42,0.47};
k in {1,3,5,7,10}; delta in {0.60,0.65,0.70,0.75,0.80}; B in {4000,6000,8000,12000}.
For tau sweep no re-runs of the LLM fallback are allowed to explode cost: run triage once
recording per-class max similarities per email, then re-threshold offline where possible;
LLM fallback calls only where the decision path actually changes. Each sweep writes
results/<sweep_name>/summary.json.

### analysis/metrics.py (agent D)
CLI: `python3 -m experiments.analysis.metrics --runs p1_syn p2_syn p3_syn --out experiments/results/analysis_syn`
Reads log.jsonl files; computes and writes (JSON + LaTeX table fragments + printed report):
- 3x3 confusion matrix per run (rows=true, cols=pred, class order IGNORE, NOTIFY, RESPOND)
- per-class precision/recall/F1 + macro-F1 + micro accuracy, per run
- accuracy with 95% Wilson CI
- pairwise McNemar between runs on common email_ids: exact binomial when discordant<25
  else chi-square with continuity correction; report b, c, statistic, p
- bootstrap 95% CI (10k resamples, seeded) for accuracy deltas between runs
- latency table: mean/median/p95 per stage per run
- token table: mean/median/p95 prompt tokens, budget utilization, % exceeding B,
  % below 50% B; dedup savings distribution
- method mix: % semantic vs LLM-fallback triage; tool-call counts (p2 vs p3)
Implement Wilson, McNemar, bootstrap with numpy/scipy directly (no sklearn dependency
assumptions — implement P/R/F1 by hand).

## Testing requirements (every agent)
- Each module gets a `python3 -m experiments...<module> --selftest` path OR a tiny
  `tests/test_<module>.py` runnable with plain `python3` (no pytest dependency) that
  exercises the module WITHOUT calling OpenAI (mock/inject fake LLM + fake vectors).
- NO OpenAI calls in selftests. Store selftests may use the live Postgres (namespace
  "selftest_*", reset before/after).
- Style: type hints, docstrings, no prints except CLI progress lines; deterministic
  seeding everywhere; NEVER catch-and-silence exceptions except where the spec says.

## Cost guardrails
- generate_synthetic pilot mode first (20 emails) before the full 500.
- run_phase --limit for pilots.
- LLM wrapper aborts the process if cumulative cost_usd exceeds $25 in a single run
  unless env MAILRECALL_COST_CAP overrides.
