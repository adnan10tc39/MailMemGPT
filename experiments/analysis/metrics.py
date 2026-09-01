"""Metrics computation over per-email run logs (log.jsonl files).

Computes per-run confusion matrices, per-class P/R/F1, macro-F1, accuracy with
Wilson 95% CIs, pairwise McNemar tests, seeded bootstrap CIs for accuracy deltas,
per-stage latency stats, token/budget stats, dedup savings, triage method mix and
tool-call counts. Everything is implemented by hand with numpy/scipy (no sklearn).

CLI:
    python3 -m experiments.analysis.metrics --runs p1_syn p2_syn p3_syn \
        --out experiments/results/analysis_syn
Selftest (no OpenAI, no Postgres):
    python3 -m experiments.analysis.metrics --selftest
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import chi2

CLASSES: List[str] = ["IGNORE", "NOTIFY", "RESPOND"]
STAGES: List[str] = [
    "triage_embed", "triage_query", "triage_llm", "sql_load", "retrieve",
    "dedup", "budget", "assemble", "respond_llm", "total",
]
Z_95: float = 1.959963984540054

REPO_ROOT = Path("/media/adnan/DATA/Agentic-LongTerm-Memory")
DEFAULT_RESULTS_ROOT = REPO_ROOT / "experiments" / "results"


# --------------------------------------------------------------------------- IO

def load_log(path: Path) -> List[dict]:
    """Read a log.jsonl file into a list of per-email records."""
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _valid(rows: Sequence[dict]) -> List[dict]:
    """Rows usable for classification metrics (no error, prediction present)."""
    return [r for r in rows if not r.get("error") and r.get("label_pred") in CLASSES
            and r.get("label_true") in CLASSES]


# ------------------------------------------------------------- classification

def confusion(y_true: Sequence[str], y_pred: Sequence[str]) -> np.ndarray:
    """3x3 confusion matrix, rows=true cols=pred, class order CLASSES."""
    idx = {c: i for i, c in enumerate(CLASSES)}
    m = np.zeros((3, 3), dtype=int)
    for t, p in zip(y_true, y_pred):
        m[idx[t], idx[p]] += 1
    return m


def prf(conf: np.ndarray) -> Dict[str, Any]:
    """Per-class precision/recall/F1, macro-F1 and micro accuracy from a confusion matrix."""
    per_class: Dict[str, Dict[str, float]] = {}
    f1s: List[float] = []
    for i, c in enumerate(CLASSES):
        tp = float(conf[i, i])
        fp = float(conf[:, i].sum() - tp)
        fn = float(conf[i, :].sum() - tp)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        per_class[c] = {"precision": p, "recall": r, "f1": f1,
                        "support": int(conf[i, :].sum())}
        f1s.append(f1)
    n = int(conf.sum())
    acc = float(np.trace(conf)) / n if n else 0.0
    return {"per_class": per_class, "macro_f1": float(np.mean(f1s)), "accuracy": acc, "n": n}


def wilson_ci(k: int, n: int, z: float = Z_95) -> Tuple[float, float]:
    """Wilson score 95% confidence interval for a binomial proportion k/n."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    rad = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, center - rad), min(1.0, center + rad))


def mcnemar(b: int, c: int) -> Dict[str, Any]:
    """McNemar test on discordant-pair counts b, c.

    Exact two-sided binomial when b+c < 25, else chi-square with continuity
    correction. Returns b, c, statistic, p, method.
    """
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "statistic": None, "p": 1.0, "method": "exact_binomial"}
    if n < 25:
        k = min(b, c)
        p = min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) * 0.5 ** n)
        return {"b": b, "c": c, "statistic": float(k), "p": float(p),
                "method": "exact_binomial"}
    stat = (abs(b - c) - 1) ** 2 / n
    p = float(chi2.sf(stat, 1))
    return {"b": b, "c": c, "statistic": float(stat), "p": p, "method": "chi2_cc"}


def bootstrap_delta(correct_a: np.ndarray, correct_b: np.ndarray,
                    n_boot: int = 10000, seed: int = 13) -> Dict[str, float]:
    """Seeded bootstrap (n_boot resamples) 95% CI for accuracy(A) - accuracy(B).

    correct_a/correct_b: boolean arrays over the SAME (common) email ids.
    """
    a = np.asarray(correct_a, dtype=float)
    b = np.asarray(correct_b, dtype=float)
    n = len(a)
    delta = float(a.mean() - b.mean()) if n else 0.0
    if n == 0:
        return {"delta": 0.0, "lo": 0.0, "hi": 0.0, "n_boot": n_boot}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    deltas = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {"delta": delta, "lo": float(lo), "hi": float(hi), "n_boot": n_boot}


# ----------------------------------------------------------- latency / tokens

def _dist(values: Sequence[float]) -> Optional[Dict[str, float]]:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    arr = np.asarray(vals)
    return {"mean": float(arr.mean()), "median": float(np.median(arr)),
            "p95": float(np.percentile(arr, 95)), "n": len(vals)}


def latency_stats(rows: Sequence[dict]) -> Dict[str, Optional[Dict[str, float]]]:
    """mean/median/p95 latency (ms) per pipeline stage."""
    out: Dict[str, Optional[Dict[str, float]]] = {}
    for stage in STAGES:
        out[stage] = _dist([(r.get("latencies") or {}).get(stage) for r in rows])
    return out


def token_stats(rows: Sequence[dict]) -> Dict[str, Any]:
    """Prompt-token distribution, budget utilization and dedup savings."""
    toks = [r.get("tokens") or {} for r in rows]
    prompt = _dist([t.get("prompt") for t in toks])
    final_prompt = _dist([t.get("final_prompt_tokens") for t in toks])
    utils = [t.get("budget_used_pct") for t in toks if t.get("budget_used_pct") is not None]
    util = _dist(utils)
    pct_over = 100.0 * sum(1 for u in utils if u > 100.0) / len(utils) if utils else None
    pct_under_half = 100.0 * sum(1 for u in utils if u < 50.0) / len(utils) if utils else None
    savings = [(r.get("retrieval") or {}).get("tokens_saved") for r in rows]
    savings = [s for s in savings if s is not None]
    dedup = _dist(savings)
    if dedup is not None:
        dedup["total"] = float(sum(savings))
    return {"prompt_tokens": prompt, "final_prompt_tokens": final_prompt,
            "budget_utilization_pct": util, "pct_exceeding_budget": pct_over,
            "pct_below_half_budget": pct_under_half, "dedup_savings": dedup}


def method_mix(rows: Sequence[dict]) -> Dict[str, Any]:
    """Triage method distribution and tool-call counts."""
    counts: Dict[str, int] = {}
    tool_total = 0
    with_tools = 0
    for r in rows:
        m = r.get("method") or "unknown"
        counts[m] = counts.get(m, 0) + 1
        tc = r.get("tool_calls") or []
        tool_total += len(tc)
        with_tools += 1 if tc else 0
    n = len(rows)
    sem = counts.get("semantic", 0)
    return {
        "counts": counts,
        "semantic_pct": 100.0 * sem / n if n else 0.0,
        "llm_fallback_pct": 100.0 * (n - sem) / n if n else 0.0,
        "tool_calls_total": tool_total,
        "tool_calls_mean": tool_total / n if n else 0.0,
        "pct_emails_with_tool_calls": 100.0 * with_tools / n if n else 0.0,
    }


# ------------------------------------------------------------------- analyze

def analyze(runs: Dict[str, List[dict]], out_dir: Path, seed: int = 13,
            n_boot: int = 10000) -> Dict[str, Any]:
    """Compute all metrics for the given runs and write metrics.json, .tex
    fragments and report.txt into out_dir. Returns the metrics dict."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics: Dict[str, Any] = {"class_order": CLASSES, "runs": {}, "pairwise": {}}

    per_run_valid: Dict[str, List[dict]] = {}
    for name, rows in runs.items():
        valid = _valid(rows)
        per_run_valid[name] = valid
        conf = confusion([r["label_true"] for r in valid],
                         [r["label_pred"] for r in valid])
        rep = prf(conf)
        k = int(np.trace(conf))
        lo, hi = wilson_ci(k, rep["n"])
        metrics["runs"][name] = {
            "n_logged": len(rows),
            "n_valid": rep["n"],
            "n_errors": sum(1 for r in rows if r.get("error")),
            "confusion": conf.tolist(),
            "per_class": rep["per_class"],
            "macro_f1": rep["macro_f1"],
            "accuracy": rep["accuracy"],
            "accuracy_wilson_ci95": [lo, hi],
            "latency_ms": latency_stats(rows),
            "tokens": token_stats(rows),
            "method_mix": method_mix(valid),
        }

    names = list(runs.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            ra = {r["email_id"]: r for r in per_run_valid[a]}
            rb = {r["email_id"]: r for r in per_run_valid[b]}
            common = sorted(set(ra) & set(rb))
            ca = np.array([ra[e]["label_pred"] == ra[e]["label_true"] for e in common])
            cb = np.array([rb[e]["label_pred"] == rb[e]["label_true"] for e in common])
            disc_b = int(np.sum(ca & ~cb))  # a correct, b wrong
            disc_c = int(np.sum(~ca & cb))  # a wrong, b correct
            metrics["pairwise"][f"{a}_vs_{b}"] = {
                "n_common": len(common),
                "mcnemar": mcnemar(disc_b, disc_c),
                "bootstrap_accuracy_delta": bootstrap_delta(ca, cb, n_boot=n_boot,
                                                            seed=seed),
            }

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, default=str)
    _write_tex_tables(metrics, out_dir)
    txt = report_text(metrics)
    with open(out_dir / "report.txt", "w", encoding="utf-8") as fh:
        fh.write(txt)
    return metrics


# ---------------------------------------------------------------- tex tables

def _tex(rows: List[List[str]], colspec: str, header: List[str],
         extra_header: Optional[str] = None) -> str:
    lines = [f"\\begin{{tabular}}{{{colspec}}}", "\\toprule"]
    if extra_header:
        lines.append(extra_header)
    lines.append(" & ".join(header) + " \\\\")
    lines.append("\\midrule")
    for r in rows:
        lines.append(" & ".join(r) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(lines)


def _f(x: Optional[float], nd: int = 3) -> str:
    return "--" if x is None else f"{x:.{nd}f}"


def _write_tex_tables(metrics: Dict[str, Any], out_dir: Path) -> None:
    runs = metrics["runs"]
    # confusion matrix per run
    for name, m in runs.items():
        rows = [[t] + [str(v) for v in m["confusion"][i]]
                for i, t in enumerate(CLASSES)]
        tex = _tex(rows, "lrrr", ["True"] + CLASSES,
                   extra_header=" & \\multicolumn{3}{c}{Predicted} \\\\")
        (out_dir / f"confusion_{name}.tex").write_text(tex, encoding="utf-8")
    # per-class P/R/F1
    rows = []
    for name, m in runs.items():
        for c in CLASSES:
            pc = m["per_class"][c]
            rows.append([name, c, _f(pc["precision"]), _f(pc["recall"]),
                         _f(pc["f1"]), str(pc["support"])])
        rows.append([name, "\\textit{macro-F1 / acc}", _f(m["macro_f1"]), "",
                     _f(m["accuracy"]), str(m["n_valid"])])
    (out_dir / "prf_table.tex").write_text(
        _tex(rows, "llrrrr", ["Run", "Class", "P", "R", "F1", "N"]), encoding="utf-8")
    # accuracy + Wilson CI
    rows = [[name, _f(m["accuracy"]),
             f"[{_f(m['accuracy_wilson_ci95'][0])}, {_f(m['accuracy_wilson_ci95'][1])}]",
             str(m["n_valid"])] for name, m in runs.items()]
    (out_dir / "accuracy_table.tex").write_text(
        _tex(rows, "lrcr", ["Run", "Accuracy", "95\\% Wilson CI", "N"]), encoding="utf-8")
    # McNemar + bootstrap
    mc_rows, bs_rows = [], []
    for pair, pm in metrics["pairwise"].items():
        mc = pm["mcnemar"]
        mc_rows.append([pair.replace("_vs_", " vs "), str(mc["b"]), str(mc["c"]),
                        _f(mc["statistic"]), _f(mc["p"], 4), mc["method"].replace("_", "\\_")])
        bs = pm["bootstrap_accuracy_delta"]
        bs_rows.append([pair.replace("_vs_", " vs "), _f(bs["delta"]),
                        f"[{_f(bs['lo'])}, {_f(bs['hi'])}]", str(pm["n_common"])])
    (out_dir / "mcnemar_table.tex").write_text(
        _tex(mc_rows, "lrrrrl", ["Pair", "$b$", "$c$", "Stat.", "$p$", "Method"]),
        encoding="utf-8")
    (out_dir / "bootstrap_table.tex").write_text(
        _tex(bs_rows, "lrcr", ["Pair", "$\\Delta$acc", "95\\% CI", "N common"]),
        encoding="utf-8")
    # latency: one column per run with mean / median / p95
    names = list(runs.keys())
    rows = []
    for stage in STAGES:
        row = [stage.replace("_", "\\_")]
        for name in names:
            d = runs[name]["latency_ms"].get(stage)
            row.append("--" if d is None else
                       f"{d['mean']:.0f} / {d['median']:.0f} / {d['p95']:.0f}")
        rows.append(row)
    (out_dir / "latency_table.tex").write_text(
        _tex(rows, "l" + "r" * len(names),
             ["Stage (ms, mean/med/p95)"] + [n.replace("_", "\\_") for n in names]),
        encoding="utf-8")
    # tokens
    rows = []
    tok_metrics = [
        ("Prompt tokens mean", lambda t: (t["prompt_tokens"] or {}).get("mean")),
        ("Prompt tokens median", lambda t: (t["prompt_tokens"] or {}).get("median")),
        ("Prompt tokens p95", lambda t: (t["prompt_tokens"] or {}).get("p95")),
        ("Budget utilization \\% (mean)", lambda t: (t["budget_utilization_pct"] or {}).get("mean")),
        ("\\% exceeding $B$", lambda t: t["pct_exceeding_budget"]),
        ("\\% below $0.5B$", lambda t: t["pct_below_half_budget"]),
        ("Dedup savings mean (tok)", lambda t: (t["dedup_savings"] or {}).get("mean")),
        ("Dedup savings total (tok)", lambda t: (t["dedup_savings"] or {}).get("total")),
    ]
    for label, getter in tok_metrics:
        rows.append([label] + [_f(getter(runs[n]["tokens"]), 1) for n in names])
    (out_dir / "tokens_table.tex").write_text(
        _tex(rows, "l" + "r" * len(names),
             ["Metric"] + [n.replace("_", "\\_") for n in names]), encoding="utf-8")
    # method mix
    rows = []
    for name in names:
        mm = runs[name]["method_mix"]
        rows.append([name.replace("_", "\\_"), _f(mm["semantic_pct"], 1),
                     _f(mm["llm_fallback_pct"], 1), str(mm["tool_calls_total"]),
                     _f(mm["tool_calls_mean"], 2)])
    (out_dir / "method_mix_table.tex").write_text(
        _tex(rows, "lrrrr", ["Run", "Semantic \\%", "LLM fallback \\%",
                             "Tool calls", "Tool calls/email"]), encoding="utf-8")


def report_text(metrics: Dict[str, Any]) -> str:
    """Human-readable summary of the metrics dict."""
    lines: List[str] = ["=== MailRecallAI metrics report ===", ""]
    for name, m in metrics["runs"].items():
        ci = m["accuracy_wilson_ci95"]
        lines.append(f"Run {name}: n_valid={m['n_valid']} errors={m['n_errors']}")
        lines.append(f"  accuracy={m['accuracy']:.4f} "
                     f"(Wilson 95% CI [{ci[0]:.4f}, {ci[1]:.4f}]) "
                     f"macro-F1={m['macro_f1']:.4f}")
        lines.append("  confusion (rows=true IGNORE,NOTIFY,RESPOND):")
        for i, c in enumerate(CLASSES):
            lines.append(f"    {c:8s} {m['confusion'][i]}")
        for c in CLASSES:
            pc = m["per_class"][c]
            lines.append(f"  {c:8s} P={pc['precision']:.3f} R={pc['recall']:.3f} "
                         f"F1={pc['f1']:.3f} (n={pc['support']})")
        mm = m["method_mix"]
        lines.append(f"  triage: semantic {mm['semantic_pct']:.1f}% / "
                     f"LLM fallback {mm['llm_fallback_pct']:.1f}%; "
                     f"tool calls total={mm['tool_calls_total']} "
                     f"mean={mm['tool_calls_mean']:.2f}")
        tk = m["tokens"]
        if tk["prompt_tokens"]:
            lines.append(f"  prompt tokens mean={tk['prompt_tokens']['mean']:.0f} "
                         f"median={tk['prompt_tokens']['median']:.0f} "
                         f"p95={tk['prompt_tokens']['p95']:.0f}")
        if tk["budget_utilization_pct"]:
            lines.append(f"  budget util mean={tk['budget_utilization_pct']['mean']:.1f}% "
                         f"over-B={tk['pct_exceeding_budget']:.1f}% "
                         f"under-0.5B={tk['pct_below_half_budget']:.1f}%")
        if tk["dedup_savings"]:
            lines.append(f"  dedup savings mean={tk['dedup_savings']['mean']:.1f} "
                         f"total={tk['dedup_savings']['total']:.0f} tokens")
        lines.append("")
    for pair, pm in metrics["pairwise"].items():
        mc = pm["mcnemar"]
        bs = pm["bootstrap_accuracy_delta"]
        stat = "-" if mc["statistic"] is None else f"{mc['statistic']:.3f}"
        lines.append(f"{pair}: n_common={pm['n_common']} "
                     f"McNemar b={mc['b']} c={mc['c']} stat={stat} "
                     f"p={mc['p']:.4f} ({mc['method']}); "
                     f"bootstrap dAcc={bs['delta']:+.4f} "
                     f"[{bs['lo']:+.4f}, {bs['hi']:+.4f}]")
    lines.append("")
    return "\n".join(lines)


# ------------------------------------------------------------------ selftest

def _fixture_logs() -> Dict[str, List[dict]]:
    """Two synthetic 20-email runs with hand-computed metrics (see selftest)."""
    true = ["IGNORE"] * 8 + ["NOTIFY"] * 6 + ["RESPOND"] * 6
    ids = [f"e{i + 1:02d}" for i in range(20)]
    # run A wrong on e05(I->N), e09(N->R), e15(R->I) -> acc 17/20
    pred_a = list(true)
    pred_a[4], pred_a[8], pred_a[14] = "NOTIFY", "RESPOND", "IGNORE"
    # run B wrong on e01,e02,e03(I->N), e05(I->R), e10,e11(N->I), e16(R->N) -> acc 13/20
    pred_b = list(true)
    for i in (0, 1, 2):
        pred_b[i] = "NOTIFY"
    pred_b[4] = "RESPOND"
    for i in (9, 10):
        pred_b[i] = "IGNORE"
    pred_b[15] = "NOTIFY"

    def row(i: int, pred: str) -> dict:
        # budget utilization mix: 10 emails at 40%, 5 at 80%, 5 at 120%
        util = 40.0 if i < 10 else (80.0 if i < 15 else 120.0)
        return {
            "email_id": ids[i], "day": 1 + i // 5, "label_true": true[i],
            "label_pred": pred, "confidence": 0.9,
            "method": "semantic" if i < 15 else "llm_p3",
            "tool_calls": [{"name": "search_similar_emails"}] if i >= 18 else [],
            "latencies": {**{s: None for s in STAGES},
                          "triage_embed": 10.0 + i, "total": 100.0 + i},
            "tokens": {"prompt": 1000, "completion": 50,
                       "final_prompt_tokens": int(util * 80),
                       "budget_used_pct": util},
            "retrieval": {"n_raw": 5, "n_kept": 3, "n_deduped": 1,
                          "sims": [0.8, 0.75, 0.71], "tokens_saved": 120},
            "response_len_chars": 0, "error": None,
        }

    return {"A": [row(i, pred_a[i]) for i in range(20)],
            "B": [row(i, pred_b[i]) for i in range(20)]}


def _selftest() -> None:
    """Verify metrics against hand-computed constants on a fixture log."""
    out_dir = Path(os.environ.get(
        "MAILRECALL_SELFTEST_DIR",
        "/tmp/claude-1000/-media-adnan-DATA-Agentic-LongTerm-Memory/"
        "250f06c2-b063-439d-83ca-fb5215f31351/scratchpad")) / "metrics_selftest"
    logs = _fixture_logs()
    m = analyze(logs, out_dir, seed=13, n_boot=2000)

    a = m["runs"]["A"]
    assert a["confusion"] == [[7, 1, 0], [0, 5, 1], [1, 0, 5]], a["confusion"]
    assert abs(a["accuracy"] - 0.85) < 1e-12
    pc = a["per_class"]
    assert abs(pc["IGNORE"]["precision"] - 0.875) < 1e-9
    assert abs(pc["IGNORE"]["recall"] - 0.875) < 1e-9
    assert abs(pc["IGNORE"]["f1"] - 0.875) < 1e-9
    for c in ("NOTIFY", "RESPOND"):
        for k in ("precision", "recall", "f1"):
            assert abs(pc[c][k] - 5.0 / 6.0) < 1e-9, (c, k, pc[c][k])
    assert abs(a["macro_f1"] - (0.875 + 5 / 6 + 5 / 6) / 3) < 1e-9
    lo, hi = a["accuracy_wilson_ci95"]
    assert abs(lo - 0.6396) < 1e-3 and abs(hi - 0.9476) < 1e-3, (lo, hi)
    b = m["runs"]["B"]
    assert abs(b["accuracy"] - 0.65) < 1e-12

    # McNemar exact: discordant b=6 (A correct, B wrong), c=2 -> p = 74/256
    mc = m["pairwise"]["A_vs_B"]["mcnemar"]
    assert mc["b"] == 6 and mc["c"] == 2, (mc["b"], mc["c"])
    assert mc["method"] == "exact_binomial"
    assert abs(mc["p"] - 0.2890625) < 1e-4, mc["p"]
    # chi-square branch: b=40, c=20 -> stat=(19)^2/60=6.01667, p~0.014171
    mc2 = mcnemar(40, 20)
    assert mc2["method"] == "chi2_cc"
    assert abs(mc2["statistic"] - 361.0 / 60.0) < 1e-9
    assert abs(mc2["p"] - 0.014171) < 2e-4, mc2["p"]

    bs = m["pairwise"]["A_vs_B"]["bootstrap_accuracy_delta"]
    assert abs(bs["delta"] - 0.20) < 1e-12
    assert bs["lo"] <= 0.20 <= bs["hi"]
    # determinism
    ca = np.array([True] * 17 + [False] * 3)
    cb = np.array([True] * 13 + [False] * 7)
    r1 = bootstrap_delta(ca, cb, n_boot=500, seed=13)
    r2 = bootstrap_delta(ca, cb, n_boot=500, seed=13)
    assert r1 == r2

    # latency: triage_embed over A = 10..29 -> mean 19.5, median 19.5, p95 28.05
    lt = a["latency_ms"]["triage_embed"]
    assert abs(lt["mean"] - 19.5) < 1e-9 and abs(lt["median"] - 19.5) < 1e-9
    assert abs(lt["p95"] - 28.05) < 1e-9, lt["p95"]
    assert a["latency_ms"]["sql_load"] is None

    # tokens: all prompt=1000; util mix 10x40 / 5x80 / 5x120
    tk = a["tokens"]
    assert abs(tk["prompt_tokens"]["mean"] - 1000.0) < 1e-9
    assert abs(tk["pct_exceeding_budget"] - 25.0) < 1e-9
    assert abs(tk["pct_below_half_budget"] - 50.0) < 1e-9
    assert abs(tk["dedup_savings"]["total"] - 20 * 120.0) < 1e-9

    mm = a["method_mix"]
    assert abs(mm["semantic_pct"] - 75.0) < 1e-9
    assert mm["tool_calls_total"] == 2

    expected_files = ["metrics.json", "report.txt", "confusion_A.tex",
                      "confusion_B.tex", "prf_table.tex", "accuracy_table.tex",
                      "mcnemar_table.tex", "bootstrap_table.tex",
                      "latency_table.tex", "tokens_table.tex",
                      "method_mix_table.tex"]
    for f in expected_files:
        path = out_dir / f
        assert path.exists(), f
        if f.endswith(".tex"):
            txt = path.read_text(encoding="utf-8")
            assert "\\begin{tabular}" in txt and "\\toprule" in txt, f
    print("metrics selftest OK:", out_dir)


# ----------------------------------------------------------------------- CLI

def main() -> None:
    ap = argparse.ArgumentParser(description="Compute metrics over run logs.")
    ap.add_argument("--runs", nargs="+", help="run namespaces (dirs under results root)")
    ap.add_argument("--out", help="output directory for metrics/report/tables")
    ap.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT),
                    help="root directory containing <run>/log.jsonl")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--selftest", action="store_true",
                    help="run offline selftest on a synthetic fixture")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return
    if not args.runs or not args.out:
        ap.error("--runs and --out are required (unless --selftest)")
    root = Path(args.results_root)
    runs = {name: load_log(root / name / "log.jsonl") for name in args.runs}
    metrics = analyze(runs, Path(args.out), seed=args.seed, n_boot=args.n_boot)
    print(report_text(metrics))
    print(f"written: {args.out}")


if __name__ == "__main__":
    main()
