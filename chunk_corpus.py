#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Turn the rebuilt 29 CFR 1926 corpus into retrieval-sized chunks.

WHY THIS EXISTS
---------------
The corpus rebuild worked: 1926.602 now carries its real operative text,
including (a)(9)(ii), the reverse-signal-alarm rule that a question about
backing a truck up to unload actually turns on. None of that existed in the
scraped corpus.

But the retriever is still configured for the old, skeletal data. It hands the
model whole parent documents, and the documents are no longer small:

    mean section          7,424 chars   ~1,856 tokens
    1926.65 (HAZWOPER)  ~100,000 chars ~25,000 tokens

Two parents per query is 15k-30k tokens of context wrapped around one 40-word
answer. Three things follow, and all three are visible in the pipeline state:

  * doc_id_mapper returns failed_generation '' - the prompt overflows.
  * The QA ranker scores 0 - the answer is real but drowned.
  * 1926.65 keeps winning. It is the longest document in the part and it is
    full of generic safety vocabulary; it even has a paragraph literally
    titled "Material handling program". At document granularity, length is an
    advantage. At paragraph granularity it stops being one.

The fix is not more retrieval tuning. It is to index the unit that answers the
question. The corpus already carries it: every paragraph has its own text and
its own official citation.

WHAT THIS PRODUCES
------------------
One record per chunk:

    {
      "chunk_id":   "1926.602::c003",
      "citation":   "1926.602(a)(9)",              # or a span (a)(9)-(a)(10)
      "text":       "1926.602 Material handling equipment > Audible alarms\\n\\n
                     (9) Audible alarms. (i) All bidirectional machines ...",
      "section_id": "1926.602",
      "section_kind": "operative",
      "retrieval_weight": 1.0,
      ...
    }

The breadcrumb is inside `text` on purpose - the embedding should see which
section and heading a paragraph belongs to, because a bare "(ii) The vehicle is
backed up only when an observer signals that it is safe to do so." carries no
searchable subject on its own.

`retrieval_weight` is a hint, not a filter: operative text 1.0, scope and
definitions 0.5, administrative and non-mandatory appendices 0.3. Multiply it
into your scores, or ignore it and filter on `section_kind`. Reserved sections
are dropped entirely.

USAGE
-----
    python chunk_corpus.py --in corpus_1926.json --out chunks_1926.json
    python chunk_corpus.py --in corpus_1926.json --out chunks_1926.jsonl --jsonl
    python chunk_corpus.py --in corpus_1926.json --stats-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

TARGET_CHARS = 1200
MAX_CHARS = 2600
MIN_CHARS = 250

WEIGHTS = {
    "operative": 1.0,
    "appendix_mandatory": 0.9,
    "definitions": 0.5,
    "scope": 0.5,
    "appendix_nonmandatory": 0.3,
    "administrative": 0.3,
}
SKIP_KINDS = {"reserved"}


def group_subsections(rec: dict[str, Any], target: int, max_c: int,
                      min_c: int) -> list[list[dict[str, Any]]]:
    """
    Walk a section's paragraphs in order and pack them into groups.

    A new top-level paragraph - (a), (b), (c) - starts a new group once the
    current one is big enough to stand alone, so a chunk never straddles two
    unrelated subjects just to hit a size target.
    """
    subs = sorted(rec["subsections"].values(), key=lambda s: s["ordinal"])
    groups: list[list[dict[str, Any]]] = []
    cur: list[dict[str, Any]] = []
    cur_len = 0

    for s in subs:
        if s["source_type"] == "ecfr_citation":
            continue
        text = (s.get("text") or "").strip()
        if not text:
            continue

        opens_top = s.get("level") == 1 and bool(s.get("designator"))
        too_big = cur_len + len(text) > max_c
        if cur and ((opens_top and cur_len >= min_c) or too_big):
            groups.append(cur)
            cur, cur_len = [], 0

        cur.append(s)
        cur_len += len(text) + 2

        if cur_len >= target:
            groups.append(cur)
            cur, cur_len = [], 0

    if cur:
        groups.append(cur)
    return groups


def breadcrumb(rec: dict[str, Any], group: list[dict[str, Any]]) -> str:
    parts = [rec["title"]]
    heading = next((s.get("heading") for s in group if s.get("heading")), "")
    if heading and heading.lower() not in rec["title"].lower():
        parts.append(heading)
    return " > ".join(parts)


def citation_for(rec: dict[str, Any], group: list[dict[str, Any]]) -> str:
    ids = [s["official_subsection_id"] for s in group
           if s.get("official_subsection_id")]
    if not ids:
        return rec["section_id"]
    if len(ids) == 1 or ids[0] == ids[-1]:
        return ids[0]
    return f"{ids[0]}-{ids[-1]}"


def chunk_corpus(corpus: dict[str, Any], target: int, max_c: int,
                 min_c: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rec in corpus.values():
        kind = rec.get("section_kind", "operative")
        if kind in SKIP_KINDS:
            continue
        groups = group_subsections(rec, target, max_c, min_c)
        for i, group in enumerate(groups, 1):
            body = "\n\n".join((s.get("text") or "").strip() for s in group)
            head = breadcrumb(rec, group)
            out.append({
                "chunk_id": f"{rec['section_id']}::c{i:03d}",
                "citation": citation_for(rec, group),
                "text": f"{head}\n\n{body}",
                "body": body,
                "breadcrumb": head,
                "doc_id": rec.get("doc_id"),
                "section_id": rec["section_id"],
                "title": rec["title"],
                "subpart": rec.get("subpart", ""),
                "subpart_title": rec.get("subpart_title", ""),
                "section_kind": kind,
                "retrieval_weight": WEIGHTS.get(kind, 0.5),
                "source": rec.get("source", ""),
                "paragraph_count": len(group),
                "char_count": len(body),
            })
    return out


def report(chunks: list[dict[str, Any]], corpus: dict[str, Any]) -> None:
    if not chunks:
        print("no chunks produced")
        return
    sizes = sorted(c["char_count"] for c in chunks)
    n = len(sizes)

    def pct(p: float) -> int:
        return sizes[min(n - 1, int(n * p))]

    before = max((r["stats"]["char_count"] for r in corpus.values()), default=0)
    print(f"\n{'=' * 60}\nCHUNKING REPORT\n{'=' * 60}")
    print(f"sections in      {len(corpus)}")
    print(f"chunks out       {n}")
    print(f"chars  min/p50   {sizes[0]:,} / {pct(0.5):,}")
    print(f"       p90/max   {pct(0.9):,} / {sizes[-1]:,}")
    print(f"tokens p50/max   ~{pct(0.5) // 4:,} / ~{sizes[-1] // 4:,}")
    print(f"\nlargest document before chunking   {before:,} chars "
          f"(~{before // 4:,} tokens)")
    print(f"largest chunk after                {sizes[-1]:,} chars "
          f"(~{sizes[-1] // 4:,} tokens)")
    if before:
        print(f"worst-case context reduction       {before / max(sizes[-1], 1):.0f}x")

    kinds: dict[str, int] = {}
    for c in chunks:
        kinds[c["section_kind"]] = kinds.get(c["section_kind"], 0) + 1
    print("\nchunks by section_kind")
    for k, v in sorted(kinds.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<24} {v:>6}   weight {WEIGHTS.get(k, 0.5)}")

    worst = sorted(chunks, key=lambda c: -c["char_count"])[:5]
    print("\nlargest chunks")
    for c in worst:
        print(f"  {c['citation']:<26} {c['char_count']:>6} chars  "
              f"{c['breadcrumb'][:44]}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", default="corpus_1926.json")
    ap.add_argument("--out", default="chunks_1926.json")
    ap.add_argument("--target-chars", type=int, default=TARGET_CHARS)
    ap.add_argument("--max-chars", type=int, default=MAX_CHARS)
    ap.add_argument("--min-chars", type=int, default=MIN_CHARS)
    ap.add_argument("--jsonl", action="store_true", help="one JSON object per line")
    ap.add_argument("--stats-only", action="store_true")
    args = ap.parse_args(argv)

    corpus = json.loads(Path(args.inp).read_text(encoding="utf-8"))
    chunks = chunk_corpus(corpus, args.target_chars, args.max_chars, args.min_chars)
    report(chunks, corpus)

    if args.stats_only:
        return 0

    out = Path(args.out)
    if args.jsonl:
        with out.open("w", encoding="utf-8") as fh:
            for c in chunks:
                fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    else:
        out.write_text(json.dumps(chunks, indent=2, ensure_ascii=False),
                       encoding="utf-8")
    print(f"\n[write] {out} ({out.stat().st_size / 1e6:.1f} MB)")

    over = [c for c in chunks if c["char_count"] > args.max_chars]
    if over:
        print(f"WARNING {len(over)} chunks exceed --max-chars "
              f"(single paragraphs longer than the cap): "
              f"{[c['citation'] for c in over[:5]]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())