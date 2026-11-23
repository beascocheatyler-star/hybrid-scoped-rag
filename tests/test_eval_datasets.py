"""Every eval label must be checkable: spans exist in their corpus, notebooks
exist, and enum fields hold legal values. A typo in a span would otherwise
silently count as a retrieval miss."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.metrics import span_in

ROOT = Path(__file__).resolve().parent.parent
DS = ROOT / "eval" / "datasets"
EX = ROOT / "examples"
CORPUS = {
    "syllabus": [EX / "course_ta" / "cs1428_syllabus.md"],
    "support": [EX / "customer_care" / "returns_and_shipping.md"],
    "both": [EX / "course_ta" / "cs1428_syllabus.md",
             EX / "customer_care" / "returns_and_shipping.md"],
    "constitution": [EX / "public_domain" / "us_constitution.md"],
    "faraday": [EX / "public_domain" / "faraday_candle_lecture1.md"],
    "darwin": [EX / "public_domain" / "darwin_origin_ch3.md"],
}


def rows(name):
    return [json.loads(l) for l in (DS / name).read_text("utf-8").splitlines() if l.strip()]


def text_of(nb):
    return "\n".join(p.read_text("utf-8") for p in CORPUS[nb])


@pytest.mark.parametrize("r", rows("retrieval.jsonl"), ids=lambda r: r["question"][:40])
def test_retrieval_spans_exist(r):
    assert r["kind"] in {"lexical", "paraphrase", "non_english"}
    assert any(span_in(text_of(r["notebook"]), [s]) for s in r["relevant_spans"])


@pytest.mark.parametrize("r", rows("answers.jsonl"), ids=lambda r: r["question"][:40])
def test_reference_facts_exist(r):
    for fact in r["reference_facts"]:
        assert span_in(text_of(r["notebook"]), [fact]), fact


def test_gate_labels_are_legal_and_balanced():
    g = rows("gate.jsonl")
    assert {r["expected"] for r in g} == {"answer", "refuse"}
    assert all(r["notebook"] in CORPUS for r in g)
    assert sum(r["expected"] == "answer" for r in g) >= 80
    assert sum(r["expected"] == "refuse" for r in g) >= 80


def test_dataset_sizes_meet_roadmap_targets():
    r = rows("retrieval.jsonl")
    assert len(r) >= 120
    assert sum(x["kind"] == "paraphrase" for x in r) >= 30
    assert sum(x["kind"] == "non_english" for x in r) >= 15
    assert len(rows("followups.jsonl")) >= 15


def test_injection_markers_are_in_their_sources():
    for r in rows("injection.jsonl"):
        assert r["marker"] in r["source"]
        assert r["expect"] in {"answer", "refuse"}
