"""Evaluation harness.

    python -m eval.run --offline                 # stub LLM, free, deterministic
    python -m eval.run --offline --write-baseline
    python -m eval.run --offline --check-baseline   # CI regression gate
    python -m eval.run --live                    # real model (costs tokens)

Offline mode measures everything that does not need a real model: retrieval,
the scope gate, follow-up handling, context recall, structural injection
resistance, latency and prompt size. Citation precision/recall and judged
faithfulness need passage-marker citations (milestone M4) and a real model;
they print as n/a until then.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import Settings  # noqa: E402
from app.core.engine import ChatEngine  # noqa: E402
from app.core.guard import classify  # noqa: E402
from app.core.metadata import ScopeFilter  # noqa: E402
from app.core.persona import PRESETS  # noqa: E402
from app.core.store import DocumentStore  # noqa: E402
from eval import metrics as M  # noqa: E402
from eval.report import render_table, write_report  # noqa: E402

DATASETS = ROOT / "eval" / "datasets"
BASELINE = ROOT / "eval" / "baseline.json"
EX = ROOT / "examples"
TOP_K = 6

# metric -> True if higher is better. Used by the regression gate.
DIRECTION = {
    "retrieval_recall@6": True,
    "retrieval_mrr": True,
    "retrieval_ndcg@10": True,
    "recall@6_lexical": True,
    "recall@6_paraphrase": True,
    "recall@6_non_english": True,
    "gate_false_accept": False,
    "gate_false_accept_confident": False,
    "gate_false_refuse": False,
    "followup_in_scope": True,
    "context_fact_recall": True,
    "injection_resistance": True,
}
# Wall-clock and prompt size depend on the machine/corpus; reported, not gated.
UNGATED = {"search_latency_p50_ms", "search_latency_p95_ms", "prompt_tokens_per_answer"}


def load(name: str) -> List[Dict]:
    with open(DATASETS / name, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_notebooks() -> Dict[str, "NotebookStore"]:
    """One SQLite notebook per eval corpus, through the production storage
    path (NotebookRegistry), in a throwaway data directory."""
    from app.core.embedding import get_embedder
    from app.core.services import build_services

    syl = (EX / "course_ta" / "cs1428_syllabus.md").read_bytes()
    sup = (EX / "customer_care" / "returns_and_shipping.md").read_bytes()
    pd = EX / "public_domain"

    settings = Settings()
    settings.data_dir = Path(tempfile.mkdtemp(prefix="hsr-eval-"))
    svc = build_services(settings, get_embedder(settings.embedding_backend,
                                                settings.embedding_dimensions))
    spec = {
        "constitution": [("us_constitution.md", (pd / "us_constitution.md").read_bytes(), None)],
        "faraday": [("faraday_candle_lecture1.md", (pd / "faraday_candle_lecture1.md").read_bytes(), None)],
        "darwin": [("darwin_origin_ch3.md", (pd / "darwin_origin_ch3.md").read_bytes(), None)],
        "syllabus": [("cs1428_syllabus.md", syl, None)],
        "support": [("returns_and_shipping.md", sup, None)],
        "both": [("cs1428_syllabus.md", syl, {"course": "cs1428"}),
                 ("returns_and_shipping.md", sup, {"dept": "support"})],
    }
    out = {}
    for name, docs in spec.items():
        st = svc.registry.store(svc.registry.create(name).id)
        for fname, data, meta in docs:
            st.add(fname, data, metadata=meta)
        out[name] = st
    return out


def _settings() -> Settings:
    return Settings()


def eval_retrieval(nbs, out: Dict, latencies: List[float]) -> None:
    rows = load("retrieval.jsonl")
    ranks, rels, by_kind = [], [], {}
    for r in rows:
        store = nbs[r["notebook"]]
        t0 = time.perf_counter()
        hits = store.index.search(r["question"], top_k=10)
        latencies.append((time.perf_counter() - t0) * 1000)
        texts = [h.chunk.text for h in hits]
        rank = M.first_relevant_rank(texts, r["relevant_spans"])
        ranks.append(rank)
        rels.append([M.span_in(t, r["relevant_spans"]) for t in texts])
        by_kind.setdefault(r["kind"], []).append(rank)
    out["retrieval_recall@6"] = M.recall_at_k(ranks, TOP_K)
    out["retrieval_mrr"] = M.mrr(ranks)
    out["retrieval_ndcg@10"] = M.ndcg_at_k(rels, 10)
    for kind, rk in sorted(by_kind.items()):
        out["recall@6_{}".format(kind)] = M.recall_at_k(rk, TOP_K)
    out["_n_retrieval"] = len(rows)


def eval_gate(nbs, out: Dict, settings: Settings) -> List[Dict]:
    rows = load("gate.jsonl")
    expected, answered, confident, errors = [], [], [], []
    for r in rows:
        store = nbs[r["notebook"]]
        scope = ScopeFilter.parse(r["scope"]) if r.get("scope") else None
        hits = store.index.search(r["question"], top_k=TOP_K, scope=scope)
        res = classify(
            r["question"],
            hits,
            relevance_floor=settings.relevance_floor,
            corpus_empty=store.is_empty,
            scope=scope,
            scope_matched_documents=store.count_matching(scope),
            dense_rescue_floor=settings.dense_rescue_floor,
        )
        expected.append(r["expected"])
        answered.append(res.allows_llm)
        confident.append(res.decision == "in_scope")
        if (r["expected"] == "answer") != res.allows_llm:
            errors.append(
                {"q": r["question"], "nb": r["notebook"], "expected": r["expected"],
                 "got": res.decision, "relevance": round(res.best_score, 3)}
            )
    fa, fr = M.gate_rates(expected, answered)
    # Borderline questions reach the LLM *with* a verify-scope instruction, so
    # gate_false_accept is an upper bound on wrong answers. The confident rate
    # counts only off-topic questions the gate ruled in_scope outright.
    fa_confident, _ = M.gate_rates(expected, confident)
    out["gate_false_accept"] = fa
    out["gate_false_accept_confident"] = fa_confident
    out["gate_false_refuse"] = fr
    out["_n_gate"] = len(rows)
    return errors


def eval_followups(nbs, out: Dict, settings: Settings) -> None:
    """Baseline: the engine retrieves on the raw follow-up turn only."""
    rows = load("followups.jsonl")
    ok = 0
    for r in rows:
        store = nbs[r["notebook"]]
        q = r["turns"][-1]
        hits = store.index.search(q, top_k=TOP_K)
        res = classify(q, hits, relevance_floor=settings.relevance_floor,
                       corpus_empty=False,
                       dense_rescue_floor=settings.dense_rescue_floor)
        ok += 1 if res.allows_llm else 0
    out["followup_in_scope"] = ok / len(rows) if rows else 0.0
    out["_n_followups"] = len(rows)


def eval_answers(nbs, out: Dict, llm_factory) -> None:
    """Context fact recall: share of reference facts present in what the
    model is actually shown. Citation metrics need M4 and print as n/a."""
    from tests.fakes import FakeLLM  # noqa: F401  (import check for offline)

    rows = load("answers.jsonl")
    hit, total, tokens = 0, 0, []
    for r in rows:
        store = nbs[r["notebook"]]
        engine = ChatEngine(_settings(), store)
        llm = llm_factory()
        if llm is not None:
            engine._client = llm
        ans = engine.answer(r["question"], PRESETS["generic"])
        context = str(llm.last["messages"]) if llm is not None and llm.last else ""
        for fact in r["reference_facts"]:
            total += 1
            hit += 1 if M.span_in(context, [fact]) else 0
        if ans.usage:
            tokens.append(ans.usage.get("input_tokens", 0))
    out["context_fact_recall"] = hit / total if total else 0.0
    out["prompt_tokens_per_answer"] = sum(tokens) / len(tokens) if tokens else 0.0
    out["citation_precision"] = None
    out["citation_recall"] = None
    out["faithfulness"] = None


def eval_injection(out: Dict, llm_factory) -> List[Dict]:
    """Structural resistance (offline): off-topic probes must be refused with
    zero LLM calls, and injected text must never reach the system prompt."""
    rows = load("injection.jsonl")
    passed, failures, refused_in_scope = 0, [], 0
    for r in rows:
        store = DocumentStore(data_dir=Path(tempfile.mkdtemp(prefix="hsr-inj-")))
        store.add(r["source_name"], r["source"].encode("utf-8"))
        engine = ChatEngine(_settings(), store)
        llm = llm_factory()
        engine._client = llm
        ans = engine.answer(r["probe"], PRESETS["generic"])
        if llm.call_count == 0:
            # Nothing reached a model, so nothing could obey the injection.
            # An in-scope probe refused here is a gate false-refusal (the safe
            # direction), tracked separately rather than as a breach.
            ok = True
            if r["expect"] == "answer":
                refused_in_scope += 1
        else:
            ok = r["expect"] == "answer" and r["marker"] not in llm.last["system"] \
                and "untrusted" in llm.last["system"]
        passed += ok
        if not ok:
            failures.append({"probe": r["probe"], "decision": ans.decision})
    out["injection_resistance"] = passed / len(rows) if rows else 0.0
    out["_injection_probes_refused_in_scope"] = refused_in_scope
    out["_n_injection"] = len(rows)
    return failures


def run(offline: bool) -> Dict:
    if offline:
        from tests.fakes import FakeLLM

        def llm_factory():
            return FakeLLM()
    else:
        def llm_factory():
            return None  # engine builds the real client from ANTHROPIC_API_KEY

    settings = _settings()
    nbs = build_notebooks()
    out: Dict = {"_mode": "offline" if offline else "live",
                 "_embedder": nbs["syllabus"]._embedding_signature}
    lat: List[float] = []
    eval_retrieval(nbs, out, lat)
    out["search_latency_p50_ms"] = M.percentile(lat, 50)
    out["search_latency_p95_ms"] = M.percentile(lat, 95)
    out["_gate_errors"] = eval_gate(nbs, out, settings)
    eval_followups(nbs, out, settings)
    eval_answers(nbs, out, llm_factory)
    if offline:
        out["_injection_failures"] = eval_injection(out, llm_factory)
    return out


def check_baseline(cur: Dict, base: Dict, tol: float = 0.02) -> List[str]:
    problems = []
    for k, higher in DIRECTION.items():
        if base.get(k) is None or cur.get(k) is None:
            continue
        delta = cur[k] - base[k]
        if (higher and delta < -tol) or (not higher and delta > tol):
            problems.append("{}: {:.3f} -> {:.3f}".format(k, base[k], cur[k]))
    if cur.get("injection_resistance", 1.0) < 1.0:
        problems.append("injection_resistance below 100%")
    return problems


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--offline", action="store_true")
    g.add_argument("--live", action="store_true")
    p.add_argument("--write-baseline", action="store_true")
    p.add_argument("--check-baseline", action="store_true")
    p.add_argument("--report", action="store_true", help="write docs/eval-latest.md")
    a = p.parse_args(argv)

    res = run(offline=a.offline)
    print(render_table(res))

    if a.report:
        write_report(res, ROOT / "docs" / "eval-latest.md")
    gated = {k: v for k, v in res.items() if not k.startswith("_") and k not in UNGATED}
    if a.write_baseline:
        BASELINE.write_text(json.dumps(gated, indent=2, sort_keys=True) + "\n")
        print("wrote", BASELINE.relative_to(ROOT))
    if a.check_baseline:
        base = json.loads(BASELINE.read_text())
        problems = check_baseline(gated, base)
        if problems:
            print("\nREGRESSION vs eval/baseline.json:")
            for x in problems:
                print("  " + x)
            return 1
        print("\nNo regression vs eval/baseline.json (tolerance 2 points).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
