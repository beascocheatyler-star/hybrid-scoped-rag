#!/usr/bin/env python3
"""Ingest and search benchmarks.

    python scripts/bench.py ingest --docs 600
    python scripts/bench.py search --chunks 10000
    python scripts/bench.py all

ingest: per-upload wall time into one SQLite notebook, split into embedding
time and everything else (extract, chunk, insert, index). The M1 target is
that "everything else" stays under 50 ms and does not grow with corpus size.

search: p50/p95 hybrid search latency over a synthetic notebook of N chunks.
Roadmap target: p95 < 150 ms at 100k chunks with NumPy installed.
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings  # noqa: E402
from app.core.embedding import HashingEmbedder  # noqa: E402
from app.core.retrieval import Chunk  # noqa: E402
from app.core.services import build_services  # noqa: E402

WORDS = ("exam grade policy refund shipping warranty lecture candle flame senate "
         "species struggle seed elephant clover treaty court jury office hours "
         "assignment late penalty return label carrier package order cancel").split()


def _pct(xs, p):
    s = sorted(xs)
    return s[min(len(s) - 1, int(len(s) * p / 100))]


class TimedEmbedder(HashingEmbedder):
    """Records time spent embedding so it can be separated from storage."""

    spent = 0.0

    def embed_batch(self, texts):
        t0 = time.perf_counter()
        try:
            return super().embed_batch(texts)
        finally:
            TimedEmbedder.spent += time.perf_counter() - t0


def bench_ingest(n_docs: int, dims: int) -> None:
    rng = random.Random(7)
    with tempfile.TemporaryDirectory() as tmp:
        s = Settings()
        s.data_dir = Path(tmp)
        svc = build_services(s, TimedEmbedder(dimensions=dims))
        st = svc.registry.default_store()
        total, rest = [], []
        for i in range(n_docs):
            body = " ".join(rng.choice(WORDS) for _ in range(400)) + " doc{}".format(i)
            TimedEmbedder.spent = 0.0
            t0 = time.perf_counter()
            st.add("doc{}.md".format(i), body.encode())
            dt = time.perf_counter() - t0
            total.append(dt * 1000)
            rest.append((dt - TimedEmbedder.spent) * 1000)
        svc.close()
    k = max(1, n_docs // 10)
    print("INGEST  {} docs, hashing {}d, SQLite".format(n_docs, dims))
    print("  total per upload       p50 {:6.1f} ms  p95 {:6.1f} ms  max {:6.1f} ms".format(
        _pct(total, 50), _pct(total, 95), max(total)))
    print("  excluding embedding    p50 {:6.1f} ms  p95 {:6.1f} ms  max {:6.1f} ms".format(
        _pct(rest, 50), _pct(rest, 95), max(rest)))
    print("  first {} vs last {} (excl. embedding): {:.1f} ms -> {:.1f} ms".format(
        k, k, statistics.mean(rest[:k]), statistics.mean(rest[-k:])))
    ok = _pct(rest, 95) < 50
    print("  target p95 < 50 ms excluding embedding: {}".format("PASS" if ok else "FAIL"))


def _zipf_vocab(rng, size=5000):
    """Pseudo-words with Zipf-like frequencies, like real text (a uniform
    30-word vocabulary would make every chunk match every query)."""
    letters = "abcdefghijklmnopqrstuvwxyz"
    vocab = ["".join(rng.choice(letters) for _ in range(rng.randint(4, 9))) for _ in range(size)]
    weights = [1.0 / (r + 1) for r in range(size)]
    return vocab, weights


def bench_search(n_chunks: int, dims: int, queries: int = 50) -> None:
    rng = random.Random(11)
    vocab, weights = _zipf_vocab(rng)
    emb = HashingEmbedder(dimensions=dims)
    with tempfile.TemporaryDirectory() as tmp:
        s = Settings()
        s.data_dir = Path(tmp)
        svc = build_services(s, emb)
        st = svc.registry.default_store()
        chunks = [
            Chunk(chunk_id="c{}".format(i), doc_id="d{}".format(i // 10),
                  doc_name="d{}.md".format(i // 10), position=i % 10,
                  text=" ".join(rng.choices(vocab, weights, k=150)))
            for i in range(n_chunks)
        ]
        t0 = time.perf_counter()
        st.index.add_chunks(chunks)
        build = time.perf_counter() - t0
        lat = []
        for _ in range(queries):
            q = " ".join(rng.choices(vocab[50:2000], k=4))
            t0 = time.perf_counter()
            st.index.search(q, top_k=6)
            lat.append((time.perf_counter() - t0) * 1000)
        svc.close()
    try:
        import numpy  # noqa: F401
        mode = "NumPy cosine"
    except ImportError:
        mode = "pure-Python cosine (install numpy)"
    print("SEARCH  {} chunks, hashing {}d, Zipf vocabulary, {}".format(n_chunks, dims, mode))
    print("  index build {:.1f} s; search p50 {:.1f} ms  p95 {:.1f} ms".format(
        build, _pct(lat, 50), _pct(lat, 95)))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("what", choices=["ingest", "search", "all"])
    p.add_argument("--docs", type=int, default=600)
    p.add_argument("--chunks", type=int, default=10000)
    p.add_argument("--dims", type=int, default=1024)
    a = p.parse_args(argv)
    if a.what in ("ingest", "all"):
        bench_ingest(a.docs, a.dims)
    if a.what in ("search", "all"):
        bench_search(a.chunks, a.dims)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
