"""Class-critical subset analysis.

A *class-critical* email is one whose correct triage label is decidable only
from a specific parent message that occurred earlier in the same inbox, beyond
the recent-history window a memory-less configuration can see. The generator
marks these with ``class_critical: true`` and names the parent in
``context_note`` ("<parent-id>: class depends on parent ...").

Reports, per run: accuracy on the class-critical subset with a Wilson 95%
interval, accuracy on the remainder, and exact-McNemar plus seeded-bootstrap
comparisons between runs on each stratum. Emits a LaTeX table.

    python3 -m experiments.analysis.class_critical \
        --runs p1_syn p2_syn p3_syn \
        --dataset experiments/data/synthetic_500.jsonl \
        --out experiments/results/analysis_syn

Selftest (no database, no network):
    python3 -m experiments.analysis.class_critical --selftest
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "experiments" / "results"
BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 13
_PARENT_RE = re.compile(r"^([A-Za-z0-9_-]+):")


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def exact_mcnemar(a_correct: Sequence[bool], b_correct: Sequence[bool]) -> Tuple[int, int, float]:
    """Two-sided exact McNemar test; returns (b, c, p) with b = a-only-correct."""
    b = sum(1 for x, y in zip(a_correct, b_correct) if x and not y)
    c = sum(1 for x, y in zip(a_correct, b_correct) if y and not x)
    n = b + c
    if n == 0:
        return b, c, 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1))
    return b, c, min(1.0, 2.0 * tail / (2 ** n))


def bootstrap_delta_ci(a_correct: Sequence[bool], b_correct: Sequence[bool],
                       n_resamples: int = BOOTSTRAP_N,
                       seed: int = BOOTSTRAP_SEED) -> Tuple[float, float]:
    """Percentile 95% CI for accuracy(b) - accuracy(a) by paired resampling."""
    m = len(a_correct)
    if m == 0:
        return (0.0, 0.0)
    rng = random.Random(seed)
    a = [1 if x else 0 for x in a_correct]
    b = [1 if x else 0 for x in b_correct]
    deltas: List[float] = []
    for _ in range(n_resamples):
        idx = [rng.randrange(m) for _ in range(m)]
        deltas.append(sum(b[i] for i in idx) / m - sum(a[i] for i in idx) / m)
    deltas.sort()
    return (deltas[int(0.025 * n_resamples)], deltas[int(0.975 * n_resamples)])


def parent_of(email: dict) -> str | None:
    """Parent email id recorded in a class-critical email's context note."""
    m = _PARENT_RE.match(str(email.get("context_note") or ""))
    return m.group(1) if m else None


def load_dataset(path: str | Path) -> Dict[str, dict]:
    return {json.loads(l)["email_id"]: json.loads(l) for l in Path(path).open(encoding="utf-8")}


def load_run(namespace: str, results_root: Path = RESULTS_ROOT) -> Dict[str, dict]:
    path = Path(results_root) / namespace / "log.jsonl"
    return {json.loads(l)["email_id"]: json.loads(l) for l in path.open(encoding="utf-8")}


def _correct(run: Dict[str, dict], ids: Sequence[str]) -> List[bool]:
    return [run[e].get("label_pred") == run[e].get("label_true") for e in ids]


def analyse(dataset: Dict[str, dict], runs: Dict[str, Dict[str, dict]]) -> dict:
    """Stratified accuracy and pairwise tests over the class-critical split."""
    shared = sorted(set(dataset) & set.intersection(*(set(r) for r in runs.values())))
    critical = [e for e in shared if dataset[e].get("class_critical")]
    rest = [e for e in shared if e not in set(critical)]

    gap = []
    order = {e: i for i, e in enumerate(sorted(dataset))}
    for e in critical:
        p = parent_of(dataset[e])
        if p in order:
            gap.append(order[e] - order[p])
    gap.sort()

    out: dict = {
        "n_total": len(shared),
        "n_class_critical": len(critical),
        "n_rest": len(rest),
        "parent_gap": {
            "min": gap[0] if gap else None,
            "median": (gap[len(gap) // 2] if len(gap) % 2 else
                       (gap[len(gap) // 2 - 1] + gap[len(gap) // 2]) / 2) if gap else None,
            "max": gap[-1] if gap else None,
        },
        "runs": {},
        "pairwise": {},
    }
    for name, run in runs.items():
        entry = {}
        for stratum, ids in (("class_critical", critical), ("rest", rest), ("all", shared)):
            corr = _correct(run, ids)
            k, n = sum(corr), len(ids)
            lo, hi = wilson_ci(k, n)
            entry[stratum] = {"correct": k, "n": n,
                              "accuracy": (k / n) if n else 0.0,
                              "wilson_ci95": [lo, hi]}
        out["runs"][name] = entry

    names = list(runs)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            for stratum, ids in (("class_critical", critical), ("rest", rest), ("all", shared)):
                ca, cb = _correct(runs[a], ids), _correct(runs[b], ids)
                bb, cc, p = exact_mcnemar(ca, cb)
                lo, hi = bootstrap_delta_ci(ca, cb)
                out["pairwise"][f"{a}|{b}|{stratum}"] = {
                    "b_first_only": bb, "c_second_only": cc, "p_value": p,
                    "delta_accuracy": (sum(cb) - sum(ca)) / len(ids) if ids else 0.0,
                    "bootstrap_ci95": [lo, hi],
                }
    return out


def latex_table(res: dict, label: str = "tab:class_critical") -> str:
    """LaTeX table of stratified accuracy, in the revision's colour idiom."""
    gap = res["parent_gap"]
    rows = []
    for name, e in res["runs"].items():
        cc, rest, all_ = e["class_critical"], e["rest"], e["all"]
        rows.append(
            f"{name.replace('_', chr(92) + '_')} & "
            f"{all_['correct']}/{all_['n']} = {all_['accuracy']*100:.1f}\\% & "
            f"{cc['correct']}/{cc['n']} = {cc['accuracy']*100:.1f}\\% "
            f"[{cc['wilson_ci95'][0]*100:.1f}, {cc['wilson_ci95'][1]*100:.1f}] & "
            f"{rest['correct']}/{rest['n']} = {rest['accuracy']*100:.1f}\\% \\\\")
    return "\n".join([
        r"\begin{center}",
        r"{\color{red}",
        r"\captionof{table}{Triage accuracy on the class-critical subset, whose labels are "
        r"decidable only from a parent message " f"{gap['min']}--{gap['max']} positions earlier "
        f"(median {gap['median']}), against the remainder of the corpus. "
        r"Brackets give Wilson 95\% intervals.}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"\textbf{Configuration} & \textbf{Overall} & "
        r"\textbf{Class-critical} & \textbf{Remainder} \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{center}",
    ])


def _selftest() -> None:
    assert wilson_ci(0, 0) == (0.0, 0.0)
    lo, hi = wilson_ci(19, 44)
    assert 0.29 < lo < 0.31 and 0.57 < hi < 0.59, (lo, hi)
    # exact McNemar: b=6, c=2 -> two-sided p = 2 * sum_{i<=2} C(8,i) / 2^8
    b, c, p = exact_mcnemar([True] * 6 + [False] * 2, [False] * 6 + [True] * 2)
    assert (b, c) == (6, 2) and abs(p - 2 * (1 + 8 + 28) / 256) < 1e-12, (b, c, p)
    b, c, p = exact_mcnemar([True, False], [True, False])
    assert (b, c, p) == (0, 0, 1.0)
    lo, hi = bootstrap_delta_ci([True] * 10, [True] * 10)
    assert lo == hi == 0.0
    assert parent_of({"context_note": "syn-0004: class depends on parent"}) == "syn-0004"
    assert parent_of({"context_note": "no parent here"}) is None
    assert parent_of({}) is None

    ds = {f"e{i}": {"email_id": f"e{i}", "class_critical": i < 2,
                    "context_note": "e9: class depends on parent" if i < 2 else ""}
          for i in range(4)}
    ds["e9"] = {"email_id": "e9", "class_critical": False, "context_note": ""}
    mk = lambda preds: {e: {"email_id": e, "label_pred": p, "label_true": "A"}
                        for e, p in preds.items()}
    runs = {"good": mk({e: "A" for e in ds}),
            "bad": mk({**{e: "A" for e in ds}, "e0": "B", "e1": "B"})}
    res = analyse(ds, runs)
    assert res["n_class_critical"] == 2 and res["n_rest"] == 3
    assert res["runs"]["good"]["class_critical"]["accuracy"] == 1.0
    assert res["runs"]["bad"]["class_critical"]["accuracy"] == 0.0
    assert res["runs"]["bad"]["rest"]["accuracy"] == 1.0
    key = "good|bad|class_critical"
    assert res["pairwise"][key]["b_first_only"] == 2
    assert res["pairwise"][key]["c_second_only"] == 0
    tex = latex_table(res)
    assert tex.startswith(r"\begin{center}") and r"\captionof{table}" in tex
    assert r"\begin{table}" not in tex, "must not emit a float; sn-jnl breaks on coloured floats"
    print("class_critical selftest: OK")


def main(argv: List[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Class-critical subset analysis")
    ap.add_argument("--runs", nargs="+", help="result namespaces to compare")
    ap.add_argument("--dataset", default="experiments/data/synthetic_500.jsonl")
    ap.add_argument("--out", help="directory for class_critical.json and .tex")
    ap.add_argument("--results-root", default=str(RESULTS_ROOT))
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)

    if args.selftest:
        _selftest()
        return
    if not args.runs:
        ap.error("--runs is required")

    ds = load_dataset(args.dataset)
    runs = {r: load_run(r, Path(args.results_root)) for r in args.runs}
    res = analyse(ds, runs)
    print(json.dumps(res, indent=2))
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "class_critical.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
        (out / "class_critical_table.tex").write_text(latex_table(res), encoding="utf-8")
        print(f"\nwrote {out/'class_critical.json'} and {out/'class_critical_table.tex'}")


if __name__ == "__main__":
    main()
