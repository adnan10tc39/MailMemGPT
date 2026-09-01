"""Enron corpus sampler + gpt-4o pre-labeler (Agent B).

Streams the Enron tarball (``experiments/data/enron/enron_mail_20150507.tar.gz``,
which may still be downloading — nothing here requires it except a real run)
with :mod:`tarfile` + :mod:`email`, WITHOUT extracting all 500k files: members
are iterated lazily, filtered to the ``inbox/`` folders of six custodians, and
scanning caps at the first 3000 matching members. Also supports an already
extracted maildir via ``--maildir``.

Processing: clean bodies (strip '-----Original Message-----' / forwarded-header
tails and '>' quoted lines), keep 30..800-word bodies (whitespace tokens),
dedupe by (normalized subject, body hash), sample 200 preserving thread groups
(same normalized subject within custodian), deterministic with ``--seed``.

Labels: batched gpt-4o JSON calls (10 emails/call, temperature 0.0) using
:data:`LABELING_GUIDELINES`. Outputs ``experiments/data/enron_200.jsonl``
(``label_source="llm_prelabel"``) and ``enron_labels_for_review.csv`` with a
blank ``author_label`` column; ``--apply-review`` merges the author's corrected
labels back and sets ``label_source="author_reviewed"``.

CLI:
    python3 -m experiments.dataset.prepare_enron
        [--tarball P] [--maildir P] [--out P] [--csv P] [--seed N]
        [--apply-review] [--selftest]

The selftest builds a tiny in-memory tar fixture and injects a fake labeler —
no OpenAI call, no Enron tarball needed.
"""
from __future__ import annotations

import argparse
import csv
import email
import email.utils
import hashlib
import io
import json
import random
import re
import sys
import tarfile
import tempfile
from datetime import timezone
from pathlib import Path
from typing import Iterator

from experiments.dataset.generate_synthetic import DATA_DIR, JSONChat, OpenAIJSON

CUSTODIANS = ["kaminski-v", "farmer-d", "beck-s", "lokay-m", "sanders-r", "dasovich-j"]
MAX_SCAN = 3000          #: cap on inbox members scanned (streaming guard)
SAMPLE_SIZE = 200
MIN_WORDS, MAX_WORDS = 30, 800
LABELS = ("IGNORE", "NOTIFY", "RESPOND")

_INBOX_RE = re.compile(r"^(?:\./)?maildir/([^/]+)/inbox/[^/]+$")
_QUOTE_MARKERS = ("-----Original Message-----", "----- Original Message -----",
                  "---------------------- Forwarded by", "----- Forwarded by",
                  "-----Forwarded by", "________________________________")

LABELING_GUIDELINES = """Label each email with exactly one of IGNORE, NOTIFY, RESPOND from the
recipient's (the mailbox owner's) point of view.

IGNORE — spam, promotions, newsletters, mailing-list chatter, and mass mail with
no relevance to the recipient's own work: nothing to read closely, nothing to do.

NOTIFY — informational content worth seeing but needing no reply: FYI
announcements, automated notices and system/status messages, mass announcements
to staff, calendar or holiday notices, reports distributed for awareness.

RESPOND — the sender addresses the recipient directly with a question, request,
task, approval, meeting arrangement, or commitment that expects a reply or
action from the recipient personally.

Decision rules: mass announcements are NOTIFY, not RESPOND; spam or content
irrelevant to the recipient is IGNORE even if it contains questions; when a
direct question or request to the recipient appears anywhere in the email,
choose RESPOND; if torn between NOTIFY and IGNORE, choose NOTIFY only when a
reasonable employee would want to see it."""

_PRELABEL_SYSTEM = (LABELING_GUIDELINES +
                    '\n\nReturn STRICT JSON {"labels": [{"email_id": "...", "label": '
                    '"IGNORE|NOTIFY|RESPOND"}]} covering every email exactly once.')


# --------------------------------------------------------------------------- #
# Scanning + parsing
# --------------------------------------------------------------------------- #

def iter_inbox_messages(tar_path: Path | None = None,
                        maildir: Path | None = None) -> Iterator[tuple[str, bytes]]:
    """Lazily yield ``(custodian, raw_rfc822_bytes)`` for custodian inbox files.

    Streams the tarball (``r|gz``) without extraction, or walks an extracted
    maildir; stops after :data:`MAX_SCAN` matching members either way.
    """
    n = 0
    if maildir is not None:
        for cust in CUSTODIANS:
            inbox = Path(maildir) / cust / "inbox"
            if not inbox.is_dir():
                continue
            for f in sorted(p for p in inbox.iterdir() if p.is_file()):
                yield cust, f.read_bytes()
                n += 1
                if n >= MAX_SCAN:
                    return
        return
    custodians = set(CUSTODIANS)
    with tarfile.open(tar_path, "r|gz") as tf:
        for member in tf:
            m = _INBOX_RE.match(member.name)
            if not m or m.group(1) not in custodians or not member.isfile():
                continue
            fo = tf.extractfile(member)
            if fo is None:
                continue
            yield m.group(1), fo.read()
            n += 1
            if n >= MAX_SCAN:
                return


def parse_message(raw: bytes) -> dict | None:
    """Parse one RFC-822 message; return sender/subject/date/body or None."""
    msg = email.message_from_bytes(raw)
    if msg.is_multipart():
        return None
    from_addr = (msg.get("From") or "").strip()
    date_hdr = msg.get("Date")
    if not from_addr or not date_hdr:
        return None
    try:
        dt = email.utils.parsedate_to_datetime(date_hdr)
    except (TypeError, ValueError):
        return None
    if dt is None:
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    payload = msg.get_payload(decode=True)
    body = payload.decode("latin-1", "replace") if isinstance(payload, bytes) \
        else str(msg.get_payload() or "")
    name = (msg.get("X-From") or "").strip().strip('"') or from_addr.split("@")[0]
    return {"sender": f"{name} <{from_addr}>",
            "subject": (msg.get("Subject") or "").strip(), "date": dt, "body": body}


def clean_body(text: str) -> str:
    """Strip quoted reply tails, forwarded-message headers and '>' blocks."""
    out: list[str] = []
    for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        stripped = line.strip()
        if any(stripped.startswith(marker) for marker in _QUOTE_MARKERS):
            break
        if stripped.startswith(">"):
            continue
        out.append(line.rstrip())
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


def norm_subject(subject: str) -> str:
    """Lowercase subject with Re:/Fw:/Fwd: prefixes stripped (thread key)."""
    t = subject.strip().lower()
    t = re.sub(r"^(?:(?:re|fw|fwd)\s*:\s*)+", "", t)
    return re.sub(r"\s+", " ", t)


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #

def build_records(raw_msgs: Iterator[tuple[str, bytes]], seed: int,
                  sample_size: int = SAMPLE_SIZE) -> list[dict]:
    """Parse, clean, filter, dedupe, thread-group and sample the scanned mail.

    Deterministic for a given seed and scan order. Returns chronologically
    sorted dataset records (labels blank until :func:`prelabel`); ``day`` is
    the dense rank of the email's calendar date within the sample.
    """
    cleaned: list[dict] = []
    seen: set[tuple[str, str]] = set()
    n_scanned = 0
    for cust, raw in raw_msgs:
        n_scanned += 1
        parsed = parse_message(raw)
        if parsed is None:
            continue
        body = clean_body(parsed["body"])
        if not MIN_WORDS <= len(body.split()) <= MAX_WORDS:
            continue
        key = (norm_subject(parsed["subject"]),
               hashlib.md5(re.sub(r"\s+", " ", body.lower()).encode()).hexdigest())
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(parsed | {"body": body, "custodian": cust})
    print(f"[enron] scanned {n_scanned} messages, kept {len(cleaned)} after clean/filter/dedupe")

    groups: dict[tuple[str, str], list[dict]] = {}
    order: list[tuple[str, str]] = []
    for i, m in enumerate(cleaned):
        ns = norm_subject(m["subject"])
        gkey = (m["custodian"], ns if ns else f"<no-subject-{i}>")
        m["_gkey"] = gkey
        if gkey not in groups:
            order.append(gkey)
        groups.setdefault(gkey, []).append(m)

    glist = [groups[k] for k in order]
    random.Random(seed).shuffle(glist)
    sampled: list[dict] = []
    for g in glist:
        if len(sampled) + len(g) <= sample_size:
            sampled.extend(g)
        if len(sampled) == sample_size:
            break
    sampled.sort(key=lambda m: m["date"])

    day_rank = {d: i + 1 for i, d in enumerate(sorted({m["date"].date() for m in sampled}))}
    tmap: dict[tuple[str, str], str] = {}
    records = []
    for i, m in enumerate(sampled):
        tid = tmap.setdefault(m["_gkey"], f"t-{len(tmap) + 1:03d}")
        records.append({"email_id": f"enr-{i + 1:04d}", "day": day_rank[m["date"].date()],
                        "ts": m["date"].isoformat(timespec="seconds"), "thread_id": tid,
                        "sender": m["sender"], "subject": m["subject"], "body": m["body"],
                        "label": "", "requires_context": False, "context_note": "",
                        "label_source": ""})
    return records


# --------------------------------------------------------------------------- #
# Pre-labeling
# --------------------------------------------------------------------------- #

def prelabel(records: list[dict], llm: JSONChat, batch_size: int = 10) -> None:
    """Label records in place via batched JSON-mode calls (10 emails/call)."""
    batches = [records[i:i + batch_size] for i in range(0, len(records), batch_size)]
    for bi, batch in enumerate(batches):
        payload = {"emails": [{"email_id": r["email_id"], "sender": r["sender"],
                               "subject": r["subject"], "body": r["body"][:1500]}
                              for r in batch]}
        user = "EMAILS JSON:\n" + json.dumps(payload)
        want = {r["email_id"] for r in batch}
        last_err: Exception | None = None
        for _ in range(3):
            try:
                out = llm.complete_json(_PRELABEL_SYSTEM, user)
                got = {e["email_id"]: str(e["label"]).strip().upper()
                       for e in out["labels"]}
                if set(got) != want or any(v not in LABELS for v in got.values()):
                    raise ValueError(f"bad label set: {got}")
                break
            except (ValueError, KeyError, TypeError) as err:
                last_err = err
        else:
            raise RuntimeError(f"prelabel batch {bi}: invalid LLM output after 3 tries: {last_err}")
        for r in batch:
            r["label"] = got[r["email_id"]]
            r["label_source"] = "llm_prelabel"
        print(f"[enron] prelabel batch {bi + 1}/{len(batches)} done")


# --------------------------------------------------------------------------- #
# Outputs + review merge
# --------------------------------------------------------------------------- #

def write_outputs(records: list[dict], out_jsonl: Path, csv_path: Path) -> None:
    """Write the dataset jsonl and the author-review CSV (blank author_label)."""
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["email_id", "sender", "subject", "body_preview", "prelabel", "author_label"])
        for r in records:
            preview = re.sub(r"\s+", " ", r["body"])[:300]
            w.writerow([r["email_id"], r["sender"], r["subject"], preview, r["label"], ""])


def apply_review(jsonl_path: Path, csv_path: Path) -> int:
    """Merge the author's CSV corrections into the jsonl; return #corrections.

    Non-empty ``author_label`` cells override the prelabel; every record's
    ``label_source`` becomes ``author_reviewed`` (a blank cell means the author
    accepted the prelabel).
    """
    with open(csv_path, newline="", encoding="utf-8") as fh:
        corrections = {}
        for row in csv.DictReader(fh):
            lab = (row.get("author_label") or "").strip().upper()
            if lab:
                if lab not in LABELS:
                    raise ValueError(f"invalid author_label {lab!r} for {row['email_id']}")
                corrections[row["email_id"]] = lab
    with open(jsonl_path, encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    n_changed = 0
    for r in records:
        new = corrections.get(r["email_id"])
        if new is not None and new != r["label"]:
            r["label"] = new
            n_changed += 1
        r["label_source"] = "author_reviewed"
    with open(jsonl_path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return n_changed


# --------------------------------------------------------------------------- #
# Selftest (no OpenAI, no tarball)
# --------------------------------------------------------------------------- #

class _FakePrelabeler:
    """Deterministic labeler stand-in: cycles RESPOND/NOTIFY/IGNORE per batch."""

    def complete_json(self, system: str, user: str) -> dict:
        payload = json.loads(user.split("EMAILS JSON:\n", 1)[1])
        cyc = ["RESPOND", "NOTIFY", "IGNORE"]
        return {"labels": [{"email_id": e["email_id"], "label": cyc[i % 3]}
                           for i, e in enumerate(payload["emails"])]}


def _fixture_msg(mid: str, date: str, subject: str, body: str) -> str:
    return (f"Message-ID: <{mid}.selftest>\nDate: {date}\n"
            "From: vince.kaminski@enron.com\nTo: john.arnold@enron.com\n"
            f"Subject: {subject}\nX-From: Vince J Kaminski\nX-To: John Arnold\n\n{body}\n")


def _build_fixture_tar(dirpath: Path) -> Path:
    """Write a tiny tar.gz with 5 inbox messages + 2 members that must be skipped."""
    long_body = ("The revised VaR model recalibration is ready for review and I would like "
                 "your comments on the volatility surface assumptions before Friday. "
                 "The gas desk numbers moved noticeably and the credit group asked for "
                 "an updated stress scenario as well. Please advise on timing.")
    reply_body = ("Thanks for the quick turnaround on the recalibration question earlier. "
                  "I checked the volatility surface assumptions against the January curves "
                  "and the differences are small, so I suggest we proceed with the Friday "
                  "review as planned and circulate the stress scenario afterwards to the desk.")
    quoted_body = ("Here is the summary of the storage optimization meeting for the team. "
                   "We agreed to rerun the injection schedule with the new basis curves and "
                   "to compare hub spreads across the March and April contracts before "
                   "the next call with the traders on Thursday morning at ten.\n"
                   "-----Original Message-----\nFrom: someone@enron.com\n"
                   "This entire quoted tail must be stripped by the cleaner.\n"
                   "> and this quoted line too\n> and this one")
    msgs = {
        "maildir/kaminski-v/inbox/1.": _fixture_msg("1", "Mon, 14 May 2001 08:22:00 -0700 (PDT)",
                                                    "Var model recalibration", long_body),
        "maildir/kaminski-v/inbox/2.": _fixture_msg("2", "Tue, 15 May 2001 10:05:00 -0700 (PDT)",
                                                    "Storage optimization notes", quoted_body),
        "maildir/kaminski-v/inbox/3.": _fixture_msg("3", "Sun, 20 May 2001 09:00:00 -0700 (PDT)",
                                                    "Var model recalibration", long_body),
        "maildir/kaminski-v/inbox/4.": _fixture_msg("4", "Thu, 17 May 2001 11:30:00 -0700 (PDT)",
                                                    "Short note", "Too short to keep around."),
        "maildir/kaminski-v/inbox/5.": _fixture_msg("5", "Wed, 16 May 2001 14:45:00 -0700 (PDT)",
                                                    "RE: Var model recalibration", reply_body),
        "maildir/kaminski-v/_sent_mail/1.": _fixture_msg("6", "Mon, 14 May 2001 09:00:00 -0700 (PDT)",
                                                         "Sent mail must be skipped", long_body),
        "maildir/skilling-j/inbox/1.": _fixture_msg("7", "Mon, 14 May 2001 09:30:00 -0700 (PDT)",
                                                    "Non-custodian must be skipped", long_body),
    }
    tar_path = dirpath / "fixture.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tf:
        for name, content in msgs.items():
            data = content.encode("latin-1")
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return tar_path


def _selftest() -> None:
    """End-to-end run on the in-memory fixture with a fake labeler."""
    with tempfile.TemporaryDirectory(prefix="enron_selftest_") as tmp:
        tmpdir = Path(tmp)
        tar_path = _build_fixture_tar(tmpdir)
        raw = list(iter_inbox_messages(tar_path=tar_path))
        assert len(raw) == 5 and all(c == "kaminski-v" for c, _ in raw), \
            "inbox filter failed"

        records = build_records(iter(raw), seed=13)
        # msg4 (too short) filtered, msg3 (duplicate subject+body) deduped -> 3 left
        assert len(records) == 3, [r["subject"] for r in records]
        assert [r["email_id"] for r in records] == ["enr-0001", "enr-0002", "enr-0003"]
        keys = [(r["day"], r["ts"]) for r in records]
        assert keys == sorted(keys) and [r["day"] for r in records] == [1, 2, 3]
        by_subj = {r["subject"]: r for r in records}
        assert (by_subj["Var model recalibration"]["thread_id"]
                == by_subj["RE: Var model recalibration"]["thread_id"]), "thread group broken"
        assert by_subj["Storage optimization notes"]["thread_id"] \
            != by_subj["Var model recalibration"]["thread_id"]
        assert "-----Original Message-----" not in by_subj["Storage optimization notes"]["body"]
        assert ">" not in by_subj["Storage optimization notes"]["body"]

        # determinism
        records_b = build_records(iter(raw), seed=13)
        assert records == records_b, "build_records not deterministic"

        prelabel(records, _FakePrelabeler())
        assert all(r["label"] in LABELS and r["label_source"] == "llm_prelabel"
                   for r in records)

        out_jsonl = tmpdir / "enron_200.jsonl"
        csv_path = tmpdir / "enron_labels_for_review.csv"
        write_outputs(records, out_jsonl, csv_path)
        with open(out_jsonl, encoding="utf-8") as fh:
            reloaded = [json.loads(line) for line in fh]
        want_keys = {"email_id", "day", "ts", "thread_id", "sender", "subject", "body",
                     "label", "requires_context", "context_note", "label_source"}
        assert len(reloaded) == 3 and all(set(r) == want_keys for r in reloaded)
        with open(csv_path, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 3 and all(row["author_label"] == "" for row in rows)
        assert all(len(row["body_preview"]) <= 300 for row in rows)

        # simulate the author correcting the first row, then merge
        rows[0]["author_label"] = "IGNORE"
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        n = apply_review(out_jsonl, csv_path)
        with open(out_jsonl, encoding="utf-8") as fh:
            merged = [json.loads(line) for line in fh]
        assert n == 1 and merged[0]["label"] == "IGNORE"
        assert all(r["label_source"] == "author_reviewed" for r in merged)
    print("SELFTEST OK prepare_enron: fixture parse/clean/dedupe/sample/prelabel/review-merge")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> None:
    """CLI entry point (see module docstring)."""
    ap = argparse.ArgumentParser(description="Enron sampler + gpt-4o pre-labeler")
    ap.add_argument("--tarball", type=Path,
                    default=DATA_DIR / "enron" / "enron_mail_20150507.tar.gz")
    ap.add_argument("--maildir", type=Path, default=None,
                    help="already-extracted maildir root (used instead of --tarball)")
    ap.add_argument("--out", type=Path, default=DATA_DIR / "enron_200.jsonl")
    ap.add_argument("--csv", type=Path, default=DATA_DIR / "enron_labels_for_review.csv")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--apply-review", action="store_true",
                    help="merge author labels from --csv into --out")
    ap.add_argument("--selftest", action="store_true", help="run offline selftest (no OpenAI)")
    args = ap.parse_args(argv)

    if args.selftest:
        _selftest()
        return
    if args.apply_review:
        n = apply_review(args.out, args.csv)
        print(f"[enron] applied review: {n} corrections; label_source=author_reviewed -> {args.out}")
        return
    if args.maildir is None and not args.tarball.exists():
        sys.exit(f"[enron] tarball not present yet: {args.tarball} (still downloading?)")

    raw = iter_inbox_messages(tar_path=args.tarball, maildir=args.maildir)
    records = build_records(raw, seed=args.seed)
    print(f"[enron] sampled {len(records)} emails")
    prelabel(records, OpenAIJSON(model="gpt-4o", temperature=0.0, provider="openai"))
    write_outputs(records, args.out, args.csv)
    print(f"[enron] wrote {args.out} and {args.csv}")


if __name__ == "__main__":
    main()
