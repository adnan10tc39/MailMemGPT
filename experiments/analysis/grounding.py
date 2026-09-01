"""Context-grounding metric for context-dependent RESPOND emails.

For every context-dependent email (``requires_context`` true, with a parent
email named in ``context_note``), we extract *anchor tokens*: distinctive
strings that occur in the parent email but NOT in the child email itself
(identifiers with digits such as invoice numbers or build IDs, and
capitalized name-like words). A generated draft can therefore only mention
an anchor if the system actually surfaced the parent context (recent
history window, rolling summary, or semantic retrieval) - the child email
alone cannot supply it.

Grounding rate = fraction of context-dependent emails whose draft response
contains at least one parent anchor. Reported overall and stratified by
whether the parent lies within the recent SQL history window (in-window)
or beyond it (beyond-window, where Phase 1/2 have no direct access and
Phase 3's semantic retrieval is the only systematic path).

Usage:
    python3 -m experiments.analysis.grounding \
        --dataset experiments/data/synthetic_500.jsonl \
        --runs p1_syn p2_syn p3_syn --window 10 \
        [--out experiments/results/analysis_syn/grounding.json]
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path("/media/adnan/DATA/Agentic-LongTerm-Memory")
ARCHIVE = ROOT / "experiments" / "data" / "archive"

# Words too generic to serve as anchors even when capitalized.
_STOP = {
    "The", "This", "That", "These", "Those", "Hi", "Hello", "Dear", "Thanks",
    "Thank", "Best", "Regards", "Please", "From", "Subject", "Body", "Re",
    "Fwd", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday",
    "Sunday", "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December", "John", "Doe",
    "NovaCore", "Team", "All", "Just", "Also", "However", "Since", "After",
    "Before", "Once", "Our", "Your", "Their", "His", "Her", "Its", "You",
    "What", "When", "Where", "Which", "While", "With", "Would", "Could",
}


def _tokens(text: str) -> set[str]:
    """Anchor-candidate tokens: digit-bearing identifiers and name-like words."""
    out: set[str] = set()
    # Identifiers containing digits: #BP-2251, 3421, v1.8, 14:00, $12,500 ...
    for m in re.findall(r"\b[A-Za-z]*-?\d[\w.:%,-]*", text):
        tok = m.strip(".,:;()[]-").lstrip("#$")
        if len(tok) >= 2 and not re.fullmatch(r"\d{1,2}", tok):
            out.add(tok.lower())
    # Capitalized name-like words (not sentence-initial filtering; STOP covers).
    for m in re.findall(r"\b[A-Z][a-z]{2,}\b", text):
        if m not in _STOP:
            out.add(m.lower())
    return out


def _contains(draft: str, anchors: set[str]) -> set[str]:
    low = draft.lower()
    return {a for a in anchors if a in low}


def load_drafts(namespace: str) -> dict[str, str]:
    """email_id -> response_text from the run's archive."""
    drafts: dict[str, str] = {}
    for fp in sorted((ARCHIVE / namespace).glob("day_*.jsonl.gz")):
        with gzip.open(fp, "rt", encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                if rec.get("response_text"):
                    drafts[rec["email_id"]] = rec["response_text"]
    return drafts


def analyze(dataset: str, runs: list[str], window: int) -> dict:
    emails = [json.loads(l) for l in open(dataset, encoding="utf-8")]
    by_id = {e["email_id"]: e for e in emails}
    pos = {e["email_id"]: i for i, e in enumerate(emails)}

    # Build the evaluation set: context-dependent emails with resolvable parents
    # and at least one usable anchor.
    cases = []
    for e in emails:
        if not e.get("requires_context"):
            continue
        m = re.search(r"syn-\d+|enr-\d+", e.get("context_note") or "")
        if not m or m.group(0) not in by_id:
            continue
        parent = by_id[m.group(0)]
        child_toks = _tokens(e["subject"] + " " + e["body"])
        anchors = _tokens(parent["subject"] + " " + parent["body"]) - child_toks
        if not anchors:
            continue
        # Parent distance in RESPOND-pair steps approximated by email index gap.
        dist = pos[e["email_id"]] - pos[parent["email_id"]]
        cases.append({"email_id": e["email_id"], "parent": parent["email_id"],
                      "distance": dist, "in_window": dist <= window,
                      "anchors": anchors})

    result: dict = {"dataset": dataset, "window": window,
                    "n_cases": len(cases),
                    "n_beyond_window": sum(1 for c in cases if not c["in_window"]),
                    "runs": {}}
    for run in runs:
        drafts = load_drafts(run)
        hit = miss_no_draft = 0
        strata = {"in_window": [0, 0], "beyond_window": [0, 0]}  # [hits, total]
        matched_examples = []
        for c in cases:
            draft = drafts.get(c["email_id"])
            if draft is None:
                miss_no_draft += 1
                continue
            found = _contains(draft, c["anchors"])
            key = "in_window" if c["in_window"] else "beyond_window"
            strata[key][1] += 1
            if found:
                hit += 1
                strata[key][0] += 1
                if len(matched_examples) < 5:
                    matched_examples.append(
                        {"email_id": c["email_id"], "anchors_found": sorted(found)[:5]})
        n_scored = strata["in_window"][1] + strata["beyond_window"][1]
        result["runs"][run] = {
            "n_scored": n_scored,
            "n_without_draft": miss_no_draft,
            "grounding_rate": round(hit / n_scored, 4) if n_scored else None,
            "in_window_rate": (round(strata["in_window"][0] / strata["in_window"][1], 4)
                               if strata["in_window"][1] else None),
            "beyond_window_rate": (round(strata["beyond_window"][0] / strata["beyond_window"][1], 4)
                                   if strata["beyond_window"][1] else None),
            "examples": matched_examples,
        }
    return result


def _selftest() -> None:
    toks = _tokens("Invoice #BP-2251 for January consulting, PO 44821, Maya Torres")
    assert "bp-2251" in toks and "44821" in toks and "maya" in toks and "torres" in toks
    assert "january" not in toks  # month is stopworded
    assert _contains("please confirm po 44821 today", {"bp-2251", "44821"}) == {"44821"}
    print("grounding selftest: OK")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", default="experiments/data/synthetic_500.jsonl")
    ap.add_argument("--runs", nargs="+", default=["p1_syn", "p2_syn", "p3_syn"])
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--out", default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return
    res = analyze(args.dataset, args.runs, args.window)
    text = json.dumps(res, indent=2, default=list)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
