"""Postgres + pgvector storage tiers for the experiment harness.

Single shared tables namespaced by a ``namespace`` column:
hot SQL memory (interactions, profile, rules, summary), warm vector memory
(few-shot triage examples, semantic email memories; cosine ops), and a cold
gzip-JSONL archive tier under ``experiments/data/archive/<namespace>/``.

Vector columns are sized from ``cfg.embed_dim`` (768 for the local
BAAI/bge-base-en-v1.5 embedder); on init the store checks the existing
``embedding`` column dimension via ``pg_attribute.atttypmod`` and recreates
``mr_fewshot``/``mr_memory`` when it differs (vector rows are dropped — they
are re-seeded per run).

Selftest (``--selftest``) uses the live Postgres with namespace ``selftest_a``
and fake deterministic ``cfg.embed_dim``-d vectors — no network calls.
"""
from __future__ import annotations

import gzip
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import psycopg2
from pgvector.psycopg2 import register_vector

from experiments.common.exp_config import ExpConfig

PG_DSN = "host=localhost port=5433 dbname=mailrecall user=postgres password=mailrecall"
ARCHIVE_ROOT = Path(__file__).resolve().parents[1] / "data" / "archive"

DEFAULT_PROFILE = (
    "John Doe, software engineering manager at TechCorp. Leads the platform "
    "engineering team, oversees the Q1 platform migration project, and handles "
    "hiring, vendor, and partner communications. Prefers concise, professional "
    "email replies signed 'John'."
)

# Adapted from src/prepare_sqldb.py default email rules (fixed wording).
DEFAULT_RULES = (
    "IGNORE rules: Spam, promotional emails, mass marketing, newsletters, and "
    "automated messages that require no action.\n"
    "NOTIFY rules: Important information the user should know but that does not "
    "need a response (e.g., system notifications, status updates, FYI "
    "announcements, calendar notices).\n"
    "RESPOND rules: Emails that need a direct response from the user (e.g., "
    "questions, meeting requests, approvals, action items, follow-ups).\n"
    "Agent instructions: Use the available tools when appropriate to help manage "
    "tasks efficiently. Generate professional, courteous email responses."
)

_DDL_TEMPLATE = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS mr_interactions (
    id BIGSERIAL PRIMARY KEY,
    namespace TEXT NOT NULL,
    email_id TEXT NOT NULL,
    day INTEGER,
    ts TEXT,
    sender TEXT,
    subject TEXT,
    body TEXT,
    classification TEXT,
    confidence DOUBLE PRECISION,
    response_text TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS mr_interactions_ns_idx ON mr_interactions (namespace, id);
CREATE TABLE IF NOT EXISTS mr_profile (
    namespace TEXT PRIMARY KEY,
    content TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS mr_rules (
    namespace TEXT PRIMARY KEY,
    content TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS mr_summary (
    namespace TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS mr_fewshot (
    id BIGSERIAL PRIMARY KEY,
    namespace TEXT NOT NULL,
    label TEXT NOT NULL,
    doc TEXT NOT NULL,
    embedding vector({embed_dim})
);
CREATE INDEX IF NOT EXISTS mr_fewshot_ns_idx ON mr_fewshot (namespace, label);
CREATE INDEX IF NOT EXISTS mr_fewshot_emb_idx
    ON mr_fewshot USING hnsw (embedding vector_cosine_ops);
CREATE TABLE IF NOT EXISTS mr_memory (
    id BIGSERIAL PRIMARY KEY,
    namespace TEXT NOT NULL,
    email_id TEXT,
    thread_id TEXT,
    sender TEXT,
    subject TEXT,
    day INTEGER,
    doc TEXT NOT NULL,
    embedding vector({embed_dim})
);
CREATE INDEX IF NOT EXISTS mr_memory_ns_idx ON mr_memory (namespace);
CREATE INDEX IF NOT EXISTS mr_memory_emb_idx
    ON mr_memory USING hnsw (embedding vector_cosine_ops);
"""

_NAMESPACED_TABLES = ("mr_interactions", "mr_profile", "mr_rules", "mr_summary",
                      "mr_fewshot", "mr_memory")

_VECTOR_TABLES = ("mr_fewshot", "mr_memory")

_HISTORY_COLS = ("email_id", "day", "ts", "sender", "subject", "body",
                 "classification", "confidence", "response_text")


def _as_vec(vec: Sequence[float]) -> np.ndarray:
    """Normalize a vector argument to a float32 numpy array for pgvector."""
    return np.asarray(vec, dtype=np.float32)


class Store:
    """Namespaced Postgres+pgvector store with a gzip-JSONL cold archive."""

    def __init__(self, cfg: ExpConfig, namespace: str) -> None:
        self.cfg = cfg
        self.namespace = namespace
        self._conn = psycopg2.connect(PG_DSN)
        self._conn.autocommit = True
        self._ensure_schema(int(cfg.embed_dim))
        register_vector(self._conn)

    def _ensure_schema(self, embed_dim: int) -> None:
        """Create all tables with vector(embed_dim) columns.

        If a vector table already exists with a different embedding dimension
        (pgvector stores it as ``pg_attribute.atttypmod``), the table is
        dropped and recreated: vector rows are per-run seed/derived data.
        """
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            for table in _VECTOR_TABLES:
                cur.execute(
                    """SELECT atttypmod FROM pg_attribute
                       WHERE attrelid = to_regclass(%s) AND attname = 'embedding'""",
                    (table,),
                )
                row = cur.fetchone()
                if row is not None and row[0] != embed_dim:
                    print(f"[store] recreating {table}: embedding dim "
                          f"{row[0]} -> {embed_dim} (vector rows dropped)",
                          flush=True)
                    cur.execute(f"DROP TABLE {table}")
            cur.execute(_DDL_TEMPLATE.format(embed_dim=embed_dim))

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    # ------------------------------------------------------------------- reset
    def reset(self) -> None:
        """Delete this namespace's rows in every table and its archive dir."""
        with self._conn.cursor() as cur:
            for table in _NAMESPACED_TABLES:
                cur.execute(f"DELETE FROM {table} WHERE namespace = %s", (self.namespace,))
        archive_dir = ARCHIVE_ROOT / self.namespace
        if archive_dir.exists():
            shutil.rmtree(archive_dir)

    # --------------------------------------------------------- SQL hot memory
    def save_interaction(self, email: dict, classification: str, confidence: float,
                         response_text: Optional[str]) -> None:
        """Persist one processed email (with classification and any reply)."""
        with self._conn.cursor() as cur:
            cur.execute(
                """INSERT INTO mr_interactions
                   (namespace, email_id, day, ts, sender, subject, body,
                    classification, confidence, response_text)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (self.namespace, email["email_id"], email.get("day"), email.get("ts"),
                 email.get("sender"), email.get("subject"), email.get("body"),
                 classification, confidence, response_text),
            )

    def recent_history(self, n_pairs: int) -> list[dict]:
        """Return the last ``n_pairs`` interactions in chronological order."""
        with self._conn.cursor() as cur:
            cur.execute(
                f"""SELECT {', '.join(_HISTORY_COLS)} FROM mr_interactions
                    WHERE namespace = %s ORDER BY id DESC LIMIT %s""",
                (self.namespace, n_pairs),
            )
            rows = cur.fetchall()
        return [dict(zip(_HISTORY_COLS, row)) for row in reversed(rows)]

    def get_user_profile(self) -> str:
        """Return the seeded user profile ('' if not seeded)."""
        return self._get_content("mr_profile") or ""

    def get_rules(self) -> str:
        """Return the seeded triage rules ('' if not seeded)."""
        return self._get_content("mr_rules") or ""

    def latest_summary(self) -> Optional[str]:
        """Return the rolling summary text, or None if none stored yet."""
        return self._get_content("mr_summary")

    def _get_content(self, table: str) -> Optional[str]:
        """Fetch the single ``content`` row for this namespace from ``table``."""
        with self._conn.cursor() as cur:
            cur.execute(f"SELECT content FROM {table} WHERE namespace = %s", (self.namespace,))
            row = cur.fetchone()
        return row[0] if row else None

    def update_summary(self, text: str) -> None:
        """Upsert the rolling summary for this namespace."""
        with self._conn.cursor() as cur:
            cur.execute(
                """INSERT INTO mr_summary (namespace, content, updated_at)
                   VALUES (%s, %s, now())
                   ON CONFLICT (namespace)
                   DO UPDATE SET content = EXCLUDED.content, updated_at = now()""",
                (self.namespace, text),
            )

    def _upsert_content(self, table: str, content: str) -> None:
        """Upsert the single ``content`` row for this namespace into ``table``."""
        with self._conn.cursor() as cur:
            cur.execute(
                f"""INSERT INTO {table} (namespace, content) VALUES (%s, %s)
                    ON CONFLICT (namespace) DO UPDATE SET content = EXCLUDED.content""",
                (self.namespace, content),
            )

    # -------------------------------------------------- warm memory (pgvector)
    def add_fewshot(self, label: str, doc: str, vec: Sequence[float]) -> None:
        """Add one few-shot triage example to the class collection ``label``."""
        with self._conn.cursor() as cur:
            cur.execute(
                """INSERT INTO mr_fewshot (namespace, label, doc, embedding)
                   VALUES (%s, %s, %s, %s)""",
                (self.namespace, label, doc, _as_vec(vec)),
            )

    def query_fewshot(self, label: str, vec: Sequence[float], topn: int) -> list[tuple[float, str]]:
        """Top-``topn`` few-shot examples of ``label`` by cosine similarity."""
        v = _as_vec(vec)
        with self._conn.cursor() as cur:
            cur.execute(
                """SELECT 1 - (embedding <=> %s) AS sim, doc FROM mr_fewshot
                   WHERE namespace = %s AND label = %s
                   ORDER BY embedding <=> %s LIMIT %s""",
                (v, self.namespace, label, v, topn),
            )
            return [(float(sim), doc) for sim, doc in cur.fetchall()]

    def add_memory(self, doc: str, vec: Sequence[float], meta: dict) -> None:
        """Add one semantic email memory; meta: email_id, thread_id, sender, subject, day."""
        with self._conn.cursor() as cur:
            cur.execute(
                """INSERT INTO mr_memory
                   (namespace, email_id, thread_id, sender, subject, day, doc, embedding)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                (self.namespace, meta.get("email_id"), meta.get("thread_id"),
                 meta.get("sender"), meta.get("subject"), meta.get("day"),
                 doc, _as_vec(vec)),
            )

    def query_memory(self, vec: Sequence[float], k: int) -> list[tuple[float, str, dict]]:
        """Top-``k`` semantic memories by cosine similarity, with metadata."""
        v = _as_vec(vec)
        with self._conn.cursor() as cur:
            cur.execute(
                """SELECT 1 - (embedding <=> %s) AS sim, doc,
                          email_id, thread_id, sender, subject, day
                   FROM mr_memory WHERE namespace = %s
                   ORDER BY embedding <=> %s LIMIT %s""",
                (v, self.namespace, v, k),
            )
            return [
                (float(sim), doc, {"email_id": eid, "thread_id": tid,
                                   "sender": sender, "subject": subject, "day": day})
                for sim, doc, eid, tid, sender, subject, day in cur.fetchall()
            ]

    # --------------------------------------------------------------- cold tier
    def archive_email(self, email: dict, response_text: Optional[str]) -> None:
        """Append one record to ``archive/<namespace>/day_<NNN>.jsonl.gz``.

        Opened in ``'ab'`` so each append is a fresh gzip member; concatenated
        members decompress as one valid stream.
        """
        archive_dir = ARCHIVE_ROOT / self.namespace
        archive_dir.mkdir(parents=True, exist_ok=True)
        record = dict(email)
        record["response_text"] = response_text
        record["archived_at"] = datetime.now(timezone.utc).isoformat()
        path = archive_dir / f"day_{int(email['day']):03d}.jsonl.gz"
        with gzip.open(path, "ab") as f:
            f.write((json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8"))


def seed_profile_and_rules(store: Store) -> None:
    """Seed the fixed John Doe profile and the four triage/agent rules."""
    store._upsert_content("mr_profile", DEFAULT_PROFILE)
    store._upsert_content("mr_rules", DEFAULT_RULES)


# --------------------------------------------------------------------- selftest
def _selftest() -> None:
    """Round-trip every tier against live Postgres (namespace selftest_a)."""
    import random

    from experiments.common.exp_config import preset

    rng = random.Random(0)
    cfg = preset("p3", run_id="selftest_a", dataset_path="")

    def rand_vec() -> list[float]:
        return [rng.gauss(0.0, 1.0) for _ in range(cfg.embed_dim)]

    store = Store(cfg, "selftest_a")
    other = Store(cfg, "selftest_a_other")
    store.reset()
    other.reset()
    try:
        # Vector columns carry the configured dimension.
        with store._conn.cursor() as cur:
            for table in _VECTOR_TABLES:
                cur.execute(
                    """SELECT atttypmod FROM pg_attribute
                       WHERE attrelid = to_regclass(%s) AND attname = 'embedding'""",
                    (table,),
                )
                assert cur.fetchone()[0] == cfg.embed_dim, table
        # Profile / rules / summary.
        assert store.get_user_profile() == "" and store.get_rules() == ""
        seed_profile_and_rules(store)
        seed_profile_and_rules(store)  # idempotent upsert
        assert store.get_user_profile() == DEFAULT_PROFILE
        assert store.get_rules() == DEFAULT_RULES
        assert store.latest_summary() is None
        store.update_summary("summary v1")
        store.update_summary("summary v2")
        assert store.latest_summary() == "summary v2"

        # Interactions round-trip, chronological order, window limit.
        for i in range(4):
            store.save_interaction(
                {"email_id": f"e{i}", "day": i + 1, "ts": f"2026-01-0{i + 1}T09:00:00",
                 "sender": f"S{i} <s{i}@x.com>", "subject": f"subj {i}", "body": f"body {i}"},
                classification="RESPOND", confidence=0.9,
                response_text=f"reply {i}" if i % 2 == 0 else None)
        hist = store.recent_history(3)
        assert [h["email_id"] for h in hist] == ["e1", "e2", "e3"]
        assert hist[0]["response_text"] is None and hist[1]["subject"] == "subj 2"
        assert hist[2]["confidence"] == 0.9 and hist[2]["day"] == 4

        # Few-shot cosine ordering: identical > slightly perturbed > random.
        base = rand_vec()
        near = [x + rng.gauss(0.0, 0.05) for x in base]
        store.add_fewshot("RESPOND", "doc_base", base)
        store.add_fewshot("RESPOND", "doc_near", near)
        store.add_fewshot("RESPOND", "doc_far", rand_vec())
        store.add_fewshot("IGNORE", "doc_other_label", base)
        res = store.query_fewshot("RESPOND", base, 3)
        assert [d for _, d in res] == ["doc_base", "doc_near", "doc_far"]
        assert res[0][0] > 0.999 and res[0][0] >= res[1][0] >= res[2][0]
        assert res[1][0] > 0.9 and res[2][0] < 0.2
        assert len(store.query_fewshot("RESPOND", base, 2)) == 2

        # Memory round-trip with metadata; k limit; namespace isolation.
        meta = {"email_id": "e9", "thread_id": "t-01", "sender": "A <a@x.com>",
                "subject": "Re: migration", "day": 7}
        store.add_memory("mem_base", base, meta)
        store.add_memory("mem_far", rand_vec(), {"email_id": "e10", "thread_id": "t-02",
                                                 "sender": "B <b@x.com>", "subject": "other",
                                                 "day": 8})
        sim, doc, got_meta = store.query_memory(base, 5)[0]
        assert sim > 0.999 and doc == "mem_base" and got_meta == meta
        assert len(store.query_memory(base, 1)) == 1
        assert store.query_memory(base, 5)[1][0] < 0.2
        assert other.query_memory(base, 5) == [] and other.recent_history(5) == []

        # Archive: two appends (two gzip members) read back as two JSONL lines.
        email = {"email_id": "e1", "day": 3, "ts": "2026-01-05T09:14:00",
                 "sender": "A <a@x.com>", "subject": "hello", "body": "hi"}
        store.archive_email(email, "reply text")
        store.archive_email(email, None)
        path = ARCHIVE_ROOT / "selftest_a" / "day_003.jsonl.gz"
        assert path.exists()
        with gzip.open(path, "rt", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f]
        assert len(lines) == 2
        assert lines[0]["email_id"] == "e1" and lines[0]["response_text"] == "reply text"
        assert lines[1]["response_text"] is None and "archived_at" in lines[1]

        # Reset clears rows and archive dir.
        store.reset()
        assert store.recent_history(10) == [] and store.get_user_profile() == ""
        assert store.latest_summary() is None
        assert store.query_fewshot("RESPOND", base, 3) == []
        assert store.query_memory(base, 5) == []
        assert not (ARCHIVE_ROOT / "selftest_a").exists()
    finally:
        store.reset()
        other.reset()
        store.close()
        other.close()
    print("store selftest: OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        print("usage: python3 -m experiments.common.store --selftest")
