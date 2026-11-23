"""Render eval results as a plain-text table and a markdown report."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Dict

ORDER = [
    "retrieval_recall@6", "retrieval_mrr", "retrieval_ndcg@10",
    "recall@6_lexical", "recall@6_paraphrase", "recall@6_non_english",
    "gate_false_accept", "gate_false_accept_confident", "gate_false_refuse", "followup_in_scope",
    "context_fact_recall", "citation_precision", "citation_recall",
    "faithfulness", "injection_resistance",
    "search_latency_p50_ms", "search_latency_p95_ms", "prompt_tokens_per_answer",
]


def _fmt(k: str, v) -> str:
    if v is None:
        return "n/a"
    if k.endswith("_ms") or k.startswith("prompt_tokens"):
        return "{:.1f}".format(v)
    return "{:.3f}".format(v)


def render_table(res: Dict) -> str:
    lines = [
        "eval mode={}  embedder={}".format(res.get("_mode"), res.get("_embedder")),
        "n: retrieval={} gate={} followups={} injection={}".format(
            res.get("_n_retrieval"), res.get("_n_gate"),
            res.get("_n_followups"), res.get("_n_injection")),
        "",
        "{:<28} {:>10}".format("metric", "value"),
        "-" * 39,
    ]
    for k in ORDER:
        if k in res:
            lines.append("{:<28} {:>10}".format(k, _fmt(k, res[k])))
    errs = res.get("_gate_errors") or []
    if errs:
        lines += ["", "gate errors ({}):".format(len(errs))]
        for e in errs:
            lines.append("  [{nb}] expected {expected}, got {got} ({relevance}): {q}".format(**e))
    return "\n".join(lines)


def write_report(res: Dict, path: Path) -> None:
    rows = "\n".join(
        "| {} | {} |".format(k, _fmt(k, res[k])) for k in ORDER if k in res
    )
    body = (
        "# Eval — latest run\n\n"
        "Generated {} by `python -m eval.run --{} --report`. Embedder: `{}`.\n\n"
        "| Metric | Value |\n|---|---|\n{}\n".format(
            date.today().isoformat(), res.get("_mode"), res.get("_embedder"), rows)
    )
    path.write_text(body, encoding="utf-8")
