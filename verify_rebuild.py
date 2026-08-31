"""Gate the OSHA corpus/chunk rebuild before it gets re-indexed.

    python verify_rebuild.py corpus_1926.json chunks_1926.json

Checks two things and exits non-zero on either failure:

1. No rootless identifiers - no official_subsection_id in the corpus and no
   endpoint of any chunk citation span begins at a numeric level (a CFR
   paragraph path always starts with a lowercase letter).
2. Known paragraphs resolve - a fixed set of real, hand-verified 29 CFR 1926
   paragraphs must be present as official_subsection_id values, or the corpus
   itself is incomplete rather than just mis-cited.
"""
import json
import re
import sys

ROOTLESS = re.compile(r"^\d+\.\d+\(\d")

KNOWN_PARAGRAPHS = [
    "1926.601(b)(6)",
    "1926.601(b)(4)",
    "1926.601(b)(1)",
    "1926.651(e)",
    "1926.351(d)(5)",
]


def citation_endpoints(citation: str) -> list[str]:
    return citation.split("-") if "-" in citation else [citation]


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(f"usage: {argv[0]} <corpus.json> <chunks.json>")
        return 2

    corpus = json.loads(open(argv[1], encoding="utf-8").read())
    chunks = json.loads(open(argv[2], encoding="utf-8").read())

    failures: list[str] = []

    rootless_ids = sorted({
        s["official_subsection_id"]
        for rec in corpus.values()
        for s in rec["subsections"].values()
        if s.get("official_subsection_id") and ROOTLESS.match(s["official_subsection_id"])
    })
    rootless_citations = sorted({
        c["citation"]
        for c in chunks
        for endpoint in citation_endpoints(c.get("citation") or "")
        if endpoint and ROOTLESS.match(endpoint)
    })

    print("--- check 1: no rootless identifiers ---")
    if rootless_ids or rootless_citations:
        if rootless_ids:
            failures.append(f"{len(rootless_ids)} rootless official_subsection_id values")
            print(f"  FAIL  {len(rootless_ids)} rootless official_subsection_id, "
                  f"e.g. {rootless_ids[:5]}")
        if rootless_citations:
            failures.append(f"{len(rootless_citations)} rootless chunk citations")
            print(f"  FAIL  {len(rootless_citations)} rootless chunk citations, "
                  f"e.g. {rootless_citations[:5]}")
    else:
        print("  ok    no identifier begins at a numeric level")

    print("\n--- check 2: known paragraphs resolve ---")
    all_official = {
        s["official_subsection_id"]
        for rec in corpus.values()
        for s in rec["subsections"].values()
        if s.get("official_subsection_id")
    }
    missing = [p for p in KNOWN_PARAGRAPHS if p not in all_official]
    for p in KNOWN_PARAGRAPHS:
        print(f"  {'ok' if p not in missing else 'FAIL':<6}{p}")
    if missing:
        failures.append(f"{len(missing)} known paragraphs missing: {missing}")

    print()
    if failures:
        print(f"VERDICT: FAIL - {len(failures)} problem(s)")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("VERDICT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
