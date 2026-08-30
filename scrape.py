#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OSHA 29 CFR Part 1926 corpus builder.  v2

WHY THIS EXISTS
---------------
The original corpus was scraped from osha.gov HTML and silently lost almost all
operative text. The old 1926.601 record held 3 paragraphs; the real section has
14. Paragraphs (b)(1)-(b)(14) - brakes, obstructed rear view, reverse signal
alarm, dump body support, tailgate trip handles - were never indexed, so no
retriever could ever have found them.

Per the official GPO ECFR XML User Guide:

    "the numbering scheme is hardcoded in the content and there is no nesting
     of elements to preserve indentation levels"

osha.gov lays those flat paragraphs out with CSS indentation, so an HTML
scraper keyed on nesting keeps the top level and drops the rest. Appendices are
genuinely flat prose and survived intact. That asymmetry is what poisoned
retrieval: the appendices were complete and the enforceable rules were
skeletons, so the appendices won every search.

This builder reads the official GPO bulk XML instead. Same content - osha.gov
itself reports "GPO Source: e-CFR" - but complete.

v2 FIXES (found by inspecting v1 output)
----------------------------------------
1. DUPLICATE RECORDS / EMPTY SUBPARTS. v1 de-duplicated sections with a set of
   id(element). lxml creates element proxies on demand and recycles them, so
   id() is not stable and the set never matched. Every section was emitted
   twice - once with an empty subpart, once with the right one. Now resolved
   with iterancestors(), single pass, no dedup needed.

2. WRONG CITATIONS. GPO writes only the *local* designator on continuation
   paragraphs: "(a)" then "(1)" then "(2)". v1 took them literally and produced
   "1926.3(1)", which is not a valid citation - it should be "1926.3(a)(1)".
   The full path is now rebuilt with a stack that follows the CFR hierarchy
   (a) -> (1) -> (i) -> (A) -> (1) -> (i), disambiguating (i)/(v)/(x) as roman
   numerals or letters from context.

3. DROPPED TABLES. GPOTABLE elements were not extracted at all, losing the soil
   classification, scaffold capacity, and ladder rating tables - which carry
   binding requirements. Now rendered as pipe-delimited text.

4. part_title kept its "PART 1926-" prefix because the strip regex was
   case-sensitive and the XML is uppercase.

USAGE
-----
    pip install requests lxml
    python test_scraper.py                      # no network needed
    python osha_ecfr_scraper.py --out corpus_1926.json \
        --map-doc-ids-from old_corpus.json      # keeps existing doc_ids stable
    python osha_ecfr_scraper.py --validate-only --out corpus_1926.json

SOURCE
------
    https://www.govinfo.gov/bulkdata/ECFR/title-29/ECFR-title29.xml
GPO publishes this feed for bulk consumption. Prefer it over the ecfr.gov API
for full-part downloads - that site's robots.txt disallows the API XML path.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterator

try:
    from lxml import etree
except ImportError:  # pragma: no cover
    sys.exit("lxml is required:  pip install lxml")

BULK_XML_URL = "https://www.govinfo.gov/bulkdata/ECFR/title-29/ECFR-title29.xml"
OSHA_SECTION_URL = "https://www.osha.gov/laws-regs/regulations/standardnumber/1926/{slug}"
DEFAULT_PART = "1926"

PARAGRAPH_TAGS = {"P", "FP", "FP-1", "FP-2", "FP1-2", "FP2-2"}
HEADING_TAGS = {"HD1", "HD2", "HD3", "HD"}
CITATION_TAGS = {"CITA", "SECAUTH", "AUTH", "SOURCE"}
TABLE_TAGS = {"GPOTABLE"}
TEXT_TAGS = PARAGRAPH_TAGS | HEADING_TAGS | CITATION_TAGS | TABLE_TAGS
ITALIC_TAGS = {"I", "E"}

DESIGNATOR_RUN = re.compile(r"^\s*((?:\([0-9A-Za-z]{1,5}\)\s*){1,6})")
DESIGNATOR_GROUP = re.compile(r"\(([0-9A-Za-z]{1,5})\)")
# "General requirements. (1) All vehicles shall have ..." - a short topic
# phrase, then the first child's designator, all inside the parent's <P>.
# Matched on text rather than on <I> markup, because GPO italicises the topic
# phrase only some of the time.
# eCFR writes it with an em-dash at least as often as a period:
#     (a) General requirements—(1) The employer shall ensure ...
# v4 required a period, so every em-dash section kept losing its letter root.
MERGED_CHILD = re.compile(
    r"^(?P<lead>[^.()—–]{1,80})\s*[.—–]\s*"
    r"(?P<des>(?:\([0-9A-Za-z]{1,5}\)\s*){1,3})")

SCOPE_TITLES = {
    "scope", "purpose and scope", "purpose", "scope and application",
    "applicability", "scope, application", "coverage", "general",
    "scope and definitions", "effective dates", "incorporation by reference",
}
DEFINITION_TITLES = {"definitions", "definitions applicable to this subpart", "terms"}
# Administrative plumbing. Real regulation text, but never the answer to a
# site safety question, and it matches generic vocabulary like "safety and
# health standards". Tagged so retrieval can drop it.
ADMINISTRATIVE_TITLES = {
    "variances from safety and health standards",
    "inspections-right of entry", "inspections - right of entry",
    "rules of practice for administrative adjudications for enforcement of "
    "safety and health standards",
    "omb control numbers under the paperwork reduction act",
    "compliance duties owed to each employee",
}


# --------------------------------------------------------------------------
# designator hierarchy
# --------------------------------------------------------------------------

ROMANS = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x",
          "xi", "xii", "xiii", "xiv", "xv", "xvi", "xvii", "xviii", "xix", "xx"]
ROMAN_SET = set(ROMANS)

# CFR nests paragraphs by cycling numbering systems:
#   (a) -> (1) -> (i) -> (A) -> (1) -> (i) -> (A)
EXPECTED_BY_DEPTH = ["lower_alpha", "digit", "lower_roman", "upper_alpha",
                     "digit", "lower_roman", "upper_alpha", "digit"]
SYSTEM_FIRST = {"lower_alpha": "a", "digit": "1", "lower_roman": "i",
                "upper_alpha": "A", "upper_roman": "I"}
FIRST_TOKEN_VALUES = set(SYSTEM_FIRST.values())


def systems_for(tok: str) -> list[str]:
    """Which numbering systems could this token belong to, best guess first."""
    if tok.isdigit():
        return ["digit"]
    if not tok.isalpha():
        return []
    if tok.islower():
        return (["lower_roman", "lower_alpha"] if tok in ROMAN_SET
                else ["lower_alpha"])
    return (["upper_roman", "upper_alpha"] if tok.lower() in ROMAN_SET
            else ["upper_alpha"])


def _letter_next(tok: str) -> str | None:
    if len(tok) == 1 and tok.isalpha():
        return tok * 2 if tok.lower() == "z" else chr(ord(tok) + 1)
    if len(tok) > 1 and len(set(tok)) == 1 and tok[0].isalpha():
        return None if tok[0].lower() == "z" else chr(ord(tok[0]) + 1) * len(tok)
    return None


def _roman_next(tok: str) -> str | None:
    low = tok.lower()
    if low in ROMAN_SET:
        i = ROMANS.index(low)
        if i + 1 < len(ROMANS):
            return ROMANS[i + 1].upper() if tok.isupper() else ROMANS[i + 1]
    return None


def is_successor(system: str, prev: str, tok: str) -> bool:
    if system == "digit":
        return prev.isdigit() and tok.isdigit() and int(tok) == int(prev) + 1
    if system in ("lower_roman", "upper_roman"):
        return _roman_next(prev) == tok
    return _letter_next(prev) == tok


class DesignatorStack:
    """
    Rebuilds full CFR paragraph paths from the local designators GPO emits.

        (a)   -> ['a']            1926.3(a)
        (1)   -> ['a', '1']       1926.3(a)(1)
        (2)   -> ['a', '2']       1926.3(a)(2)
        (b)   -> ['b']            1926.3(b)

    Levels are keyed to the NUMBERING SYSTEM, not to succession alone. v2
    matched only successors, so a list that restarted - a second (a), or a
    fresh (1) under a new parent - looked like neither a sibling nor a known
    child and got pushed one level deeper every time. Across part 1926 that
    ran the stack to depth 22, which is impossible in the CFR.

    Order of resolution for a token:
      1. successor of an entry already on the stack, same system  -> sibling
      2. first element of a system already on the stack           -> restart
      3. first element of a system not yet on the stack           -> child
      4. otherwise                                                -> replace deepest
    """

    def __init__(self) -> None:
        self.entries: list[tuple[str, str]] = []  # (token, system)

    @property
    def stack(self) -> list[str]:
        return [t for t, _ in self.entries]

    def reset(self, tokens: list[str]) -> list[str]:
        """The paragraph carried an explicit full path - trust it."""
        self.entries = []
        for depth, tok in enumerate(tokens):
            cands = systems_for(tok) or ["digit"]
            want = EXPECTED_BY_DEPTH[depth] if depth < len(EXPECTED_BY_DEPTH) else None
            system = want if want in cands else cands[0]
            self.entries.append((tok, system))
        return self.stack

    def push(self, tok: str) -> list[str]:
        cands = systems_for(tok)
        if not cands:
            return self.stack

        for i in range(len(self.entries) - 1, -1, -1):
            prev_tok, prev_sys = self.entries[i]
            if prev_sys in cands and is_successor(prev_sys, prev_tok, tok):
                self.entries = self.entries[:i] + [(tok, prev_sys)]
                return self.stack

        for i in range(len(self.entries) - 1, -1, -1):
            _prev_tok, prev_sys = self.entries[i]
            if prev_sys in cands and SYSTEM_FIRST.get(prev_sys) == tok:
                self.entries = self.entries[:i] + [(tok, prev_sys)]
                return self.stack

        on_stack = {s for _, s in self.entries}
        for sys in cands:
            if SYSTEM_FIRST.get(sys) == tok and sys not in on_stack:
                self.entries.append((tok, sys))
                return self.stack

        # Out of sequence. If the token belongs to the SAME system as the
        # deepest entry it is a gap in that list, so replace. If it belongs to
        # a DIFFERENT system it cannot be a sibling at all - it is a child, and
        # replacing would drop its letter parent. v3 always replaced, which is
        # how "(a) General. (1) ..." followed by a bare "(2)" produced the
        # rootless citation 1926.102(2).
        if self.entries:
            deepest_sys = self.entries[-1][1]
            if cands[0] == deepest_sys:
                self.entries = self.entries[:-1] + [(tok, cands[0])]
            else:
                self.entries.append((tok, cands[0]))
        else:
            self.entries = [(tok, cands[0])]
        return self.stack


# --------------------------------------------------------------------------
# fetching
# --------------------------------------------------------------------------

def download_bulk_xml(cache_dir: Path, force: bool = False) -> Path:
    import requests

    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / "ECFR-title29.xml"
    if dest.exists() and not force and dest.stat().st_size > 1_000_000:
        print(f"[cache] using {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
        return dest

    print(f"[fetch] {BULK_XML_URL}")
    tmp = dest.with_suffix(".part")
    with requests.get(BULK_XML_URL, stream=True, timeout=180,
                      headers={"User-Agent": "osha-rag-corpus-builder/2.0"}) as r:
        r.raise_for_status()
        written = 0
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
                written += len(chunk)
                print(f"\r[fetch] {written / 1e6:8.1f} MB", end="", flush=True)
    print()
    tmp.replace(dest)
    return dest


# --------------------------------------------------------------------------
# text extraction
# --------------------------------------------------------------------------

def element_text(elem) -> str:
    text = "".join(elem.itertext()).replace(" ", " ")
    return re.sub(r"\s+", " ", text).strip()


def render_table(elem) -> str:
    """
    Flatten a table into pipe-delimited lines so its values are indexable.

    Falls back to flat text when the row/cell tags are not the ones we expect -
    a table whose numbers reach the index in a clumsy shape is far better than
    a table silently dropped, which is how the soil-classification and
    scaffold-capacity limits went missing.
    """
    lines: list[str] = []
    for tag in ("TTITLE", "TTL", "CAPTION"):
        node = elem.find(f".//{tag}")
        if node is not None:
            t = element_text(node)
            if t:
                lines.append(t)
            break
    boxhd = elem.find(".//BOXHD")
    if boxhd is not None:
        heads = [element_text(c) for c in boxhd.iter("CHED") if element_text(c)]
        if heads:
            lines.append(" | ".join(heads))
    for row_tag, cell_tag in (("ROW", "ENT"), ("TR", "TD"), ("TR", "TH")):
        for row in elem.iter(row_tag):
            cells = [element_text(c) for c in row.iter(cell_tag)]
            if any(cells):
                lines.append(" | ".join(cells))
        if lines:
            break
    if not lines:
        flat = element_text(elem)
        return flat
    return "\n".join(lines)


def split_designator(text: str) -> tuple[list[str], str]:
    m = DESIGNATOR_RUN.match(text)
    if not m:
        return [], text
    groups = DESIGNATOR_GROUP.findall(m.group(1))
    if not groups:
        return [], text
    return groups, text[m.end():].strip()


def extract_inline_heading(elem, body: str) -> str:
    """
    A paragraph's topic phrase is italicised immediately after the designator:
        <P>(a) <I>Coverage.</I> Motor vehicles ...</P>

    It must OPEN the body. v4 only required the italic run to appear somewhere
    in the first 120 characters, so an italicised trailing URL - "Web site:
    <E>http://techstreet.com</E>" - became the heading and then stuck to every
    following paragraph in the section.
    """
    for child in elem:
        if child.tag in ITALIC_TAGS:
            t = (child.text or "").strip()
            if not t or len(t) > 120:
                return ""
            if "://" in t or "@" in t or t.startswith("www."):
                return ""
            if body.startswith(t):
                return t.rstrip(". ").strip()
        break
    return ""


def first_sentences(text: str, max_chars: int = 320) -> str:
    if len(text) <= max_chars:
        return text
    out: list[str] = []
    total = 0
    for sent in re.split(r"(?<=[.;:])\s+", text):
        if total + len(sent) > max_chars and out:
            break
        out.append(sent)
        total += len(sent) + 1
    return " ".join(out) if out else text[:max_chars].rstrip()


# --------------------------------------------------------------------------
# identifiers
# --------------------------------------------------------------------------

def tidy_caps(text: str) -> str:
    """GPO writes part and subpart heads in caps; match the old corpus casing."""
    letters = [c for c in text if c.isalpha()]
    if letters and sum(c.isupper() for c in letters) / len(letters) > 0.85:
        small = {"a", "an", "and", "as", "at", "by", "for", "from", "in", "of",
                 "on", "or", "the", "to", "with"}
        words = text.lower().split()
        out = [w.capitalize() if (i == 0 or w not in small) else w
               for i, w in enumerate(words)]
        return " ".join(out)
    return text


def normalize_section_id(raw: str) -> str:
    s = (raw or "").replace("§", "").replace("§", "").strip()
    return re.sub(r"\s+", " ", s)


def appendix_section_id(head: str, part: str) -> str:
    head = re.sub(r"\s+", " ", (head or "").strip())
    m = re.search(r"Appendix\s+([A-Z0-9]+)\s+to\s+§?\s*(\d+\.\d+)", head, re.I)
    if m:
        return f"{m.group(2)} App {m.group(1).upper()}"
    m = re.search(r"Appendix\s+([A-Z0-9]+)\s+to\s+Subpart\s+([A-Z]+)", head, re.I)
    if m:
        return f"{part} Subpart {m.group(2).upper()} App {m.group(1).upper()}"
    m = re.search(r"Appendix\s+([A-Z0-9]+)", head, re.I)
    if m:
        return f"{part} App {m.group(1).upper()}"
    return head


def osha_url(section_id: str) -> str:
    return OSHA_SECTION_URL.format(slug=section_id.replace(" ", ""))


def classify(title_text: str, is_appendix: bool, head: str) -> str:
    if "[reserved]" in f"{head} {title_text}".lower():
        return "reserved"
    if is_appendix:
        blob = f"{head} {title_text}".lower()
        if "non-mandatory" in blob or "nonmandatory" in blob:
            return "appendix_nonmandatory"
        if "mandatory" in blob:
            return "appendix_mandatory"
        return "appendix_nonmandatory"
    t = title_text.strip().rstrip(".").lower()
    t = t.replace("—", "-")
    if t in DEFINITION_TITLES:
        return "definitions"
    if t in SCOPE_TITLES:
        return "scope"
    if t in ADMINISTRATIVE_TITLES:
        return "administrative"
    return "operative"


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------

def iter_part_sections(xml_path: Path, part: str) -> Iterator[dict[str, Any]]:
    """
    Stream the title XML, find DIV5[@N=part], then walk every DIV8 section and
    DIV9 appendix in document order, attributing each to its nearest DIV6
    ancestor. One pass, so a section can never be emitted twice.
    """
    context = etree.iterparse(str(xml_path), events=("end",), tag="DIV5",
                              huge_tree=True, recover=True)
    for _event, div5 in context:
        if (div5.get("N") or "").strip() != part:
            div5.clear()
            continue

        head_el = div5.find("HEAD")
        part_head = element_text(head_el) if head_el is not None else ""
        part_title = tidy_caps(
            re.sub(r"^part\s+[\d.]+\s*[-—–]?\s*", "", part_head,
                   flags=re.I).strip())

        for node in div5.iter():
            if node.tag not in ("DIV8", "DIV9"):
                continue

            div6 = None
            for anc in node.iterancestors():
                if anc.tag == "DIV6":
                    div6 = anc
                    break
                if anc.tag == "DIV5":
                    break

            if div6 is not None:
                sub_n = (div6.get("N") or "").strip()
                sh = div6.find("HEAD")
                sub_head = element_text(sh) if sh is not None else ""
                subpart_title = tidy_caps(
                    re.sub(r"^subpart\s+[A-Z]+\s*[-—–]?\s*", "",
                           sub_head, flags=re.I).strip())
                subpart = f"{part} Subpart {sub_n}" if sub_n else ""
            else:
                subpart, subpart_title = "", ""

            yield {
                "element": node,
                "is_appendix": node.tag == "DIV9",
                "part": part,
                "part_title": part_title,
                "subpart": subpart,
                "subpart_title": subpart_title,
            }
        div5.clear()
        break
    del context


def parse_section(raw: dict[str, Any]) -> dict[str, Any]:
    elem = raw["element"]
    is_appendix = raw["is_appendix"]

    head_el = elem.find("HEAD")
    head = element_text(head_el) if head_el is not None else ""

    if is_appendix:
        section_id = appendix_section_id(head, raw["part"])
        title_text = re.sub(r"^.*?[-—]\s*", "", head).strip() or head
    else:
        n = normalize_section_id(elem.get("N") or "")
        if not n:
            m = re.match(r"§?\s*([\d.]+)", head)
            n = m.group(1) if m else ""
        section_id = n
        title_text = re.sub(r"^§?\s*[\d.]+\s*", "", head).strip()

    title = f"{section_id} - {title_text}".strip(" -")

    subsections: dict[str, dict[str, Any]] = {}
    body_chunks: list[str] = []
    prose_chunks: list[str] = []
    ordinal = 0
    current_heading = ""
    designated = 0
    tables = 0
    stack = DesignatorStack()

    for node in elem.iter():
        if node.tag not in TEXT_TAGS or node is head_el:
            continue

        if node.tag in TABLE_TAGS:
            text = render_table(node)
            if not text:
                continue
            tables += 1
            ordinal += 1
            key = f"{section_id}::p{ordinal:03d}"
            subsections[key] = {
                "subsection_id": key,
                "official_subsection_id": "",
                "level": len(stack.stack) + 1,
                "ordinal": ordinal,
                "designator": "",
                "heading": current_heading,
                "text": text,
                "summary": first_sentences(text),
                "source_type": "ecfr_table",
            }
            body_chunks.append(text)
            prose_chunks.append(text)
            continue

        text = element_text(node)
        if not text:
            continue

        groups, body = split_designator(text)
        inline_heading = extract_inline_heading(node, body)

        # GPO frequently folds a parent and its first child into one <P>:
        #     (a) General requirements. (1) All vehicles shall have ...
        # Seeing only "(a)" here leaves the following "(2)" with no "(1)"
        # sibling on the stack, which is what produced rootless citations like
        # 1926.102(2). v3 keyed this off the italic heading; GPO italicises the
        # topic phrase only sometimes, so match the text shape instead.
        if groups:
            m = MERGED_CHILD.match(body)
            if m:
                extra = DESIGNATOR_GROUP.findall(m.group("des"))
                # A merged child always OPENS its list, so its designator must
                # be the first element of some numbering system. Without this
                # guard "…; telephone: (877) 413-5184" reads as a child (877).
                if extra and extra[0] not in FIRST_TOKEN_VALUES:
                    extra = []
                if extra and len(groups) + len(extra) <= 6:
                    groups = groups + extra
                    body = body[m.end():].strip()
                    if not inline_heading:
                        inline_heading = m.group("lead").rstrip(". ").strip()

        if groups:
            path = stack.reset(groups) if len(groups) > 1 else stack.push(groups[0])
            official = section_id + "".join(f"({g})" for g in path)
            level = len(path)
            designated += 1
        else:
            path, official, level = [], "", max(len(stack.stack), 1)

        if inline_heading:
            current_heading = inline_heading

        if node.tag in CITATION_TAGS or re.match(r"^\[\d+\s+FR\s", text):
            source_type = "ecfr_citation"
        elif node.tag in HEADING_TAGS or (inline_heading and not body):
            source_type = "ecfr_heading"
        else:
            source_type = "ecfr_paragraph"

        ordinal += 1
        key = f"{section_id}::p{ordinal:03d}"
        subsections[key] = {
            "subsection_id": key,
            "official_subsection_id": official,
            "level": level,
            "ordinal": ordinal,
            "designator": "".join(f"({g})" for g in (path if groups else [])),
            "heading": current_heading,
            "text": text,
            "summary": first_sentences(body or text),
            "source_type": source_type,
        }
        body_chunks.append(text)
        if source_type != "ecfr_citation":
            prose_chunks.append(text)

    full_text = "\n\n".join(body_chunks)
    kind = classify(title_text, is_appendix, head)
    if not full_text.strip():
        kind = "reserved"

    return {
        "part": raw["part"],
        "part_title": raw["part_title"],
        "subpart": raw["subpart"],
        "subpart_title": raw["subpart_title"],
        "section_id": section_id,
        "title": title,
        "source": osha_url(section_id),
        "gpo_source": "e-CFR",
        "section_kind": kind,
        "is_appendix": is_appendix,
        "full_text": full_text,
        "section_summary": first_sentences("\n\n".join(prose_chunks), 600),
        "subsections": subsections,
        "stats": {
            "char_count": len(full_text),
            "paragraph_count": len(subsections),
            "designated_paragraph_count": designated,
            "table_count": tables,
            "max_depth": max((s["level"] for s in subsections.values()), default=0),
        },
    }


def legacy_header(rec: dict[str, Any]) -> str:
    return (
        f"Part Number:{rec['part']}\n\n"
        f"Part Number Title:{rec['part_title']}\n\n"
        f"Subpart:{rec['subpart']}\n\n"
        f"Subpart Title:{rec['subpart_title']}\n\n"
        f"Standard Number:{rec['section_id']}\n\n"
        f"Title:{rec['title'].split(' - ', 1)[-1]}\n\n"
        f"GPO Source:e-CFR\n\n"
    )


def build_corpus(xml_path: Path, part: str, use_legacy_header: bool,
                 doc_id_map: dict[str, str] | None) -> dict[str, dict[str, Any]]:
    corpus: dict[str, dict[str, Any]] = {}
    next_id = 0
    used: set[str] = set(doc_id_map.values()) if doc_id_map else set()

    for raw in iter_part_sections(xml_path, part):
        rec = parse_section(raw)
        if not rec["section_id"]:
            continue

        if doc_id_map and rec["section_id"] in doc_id_map:
            doc_id = doc_id_map[rec["section_id"]]
        else:
            while str(next_id) in used:
                next_id += 1
            doc_id = str(next_id)
            used.add(doc_id)

        if use_legacy_header:
            rec["full_text"] = legacy_header(rec) + rec["full_text"]

        rec["doc_id"] = doc_id
        rec["metadata"] = {
            "doc_id": doc_id,
            "part": rec["part"],
            "subpart": rec["subpart"],
            "subpart_title": rec["subpart_title"],
            "section_id": rec["section_id"],
            "title": rec["title"],
            "source": rec["source"],
            "section_kind": rec["section_kind"],
        }
        corpus[doc_id] = rec

    return corpus


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------

GOLDEN_ASSERTIONS: list[tuple[str, str]] = [
    ("1926.601", "reverse signal alarm"),
    ("1926.601", "trip handles for tailgates"),
    ("1926.601", "positive means of support"),
    ("1926.601", "service brake system"),
    ("1926.600", "fully lowered or blocked"),
    ("1926.602", "powered industrial truck operator training"),
    ("1926.501", "unprotected sides and edges"),
    ("1926.652", "protective system"),
    ("1926.451", "scaffold"),
    ("1926.1053", "ladder"),
]

# Citations that must resolve to a paragraph, proving path reconstruction works.
GOLDEN_DESIGNATORS = [
    "1926.601(b)(4)",
    "1926.3(a)(1)",
    "1926.501(b)(1)",
    "1926.652(a)(1)",
]

MIN_OPERATIVE_CHARS = 400
MIN_DESIGNATED_RATIO = 0.5
MIN_SECTIONS = 250


def validate(corpus: dict[str, dict[str, Any]]) -> int:
    failures: list[str] = []
    warnings: list[str] = []

    print(f"\n{'=' * 68}\nCORPUS VALIDATION\n{'=' * 68}")

    total_paras = sum(r["stats"]["paragraph_count"] for r in corpus.values())
    designated = sum(r["stats"]["designated_paragraph_count"] for r in corpus.values())
    total_chars = sum(r["stats"]["char_count"] for r in corpus.values())
    tables = sum(r["stats"].get("table_count", 0) for r in corpus.values())
    depth = max((r["stats"].get("max_depth", 0) for r in corpus.values()), default=0)

    print(f"sections            {len(corpus)}")
    print(f"paragraphs          {total_paras}")
    print(f"with official id    {designated} ({designated / max(total_paras, 1):.0%})")
    print(f"tables extracted    {tables}")
    print(f"deepest nesting     level {depth}")
    print(f"total characters    {total_chars:,}")
    print(f"mean chars/section  {total_chars // max(len(corpus), 1):,}")

    kinds: dict[str, int] = {}
    for r in corpus.values():
        kinds[r["section_kind"]] = kinds.get(r["section_kind"], 0) + 1
    print("section_kind        " + ", ".join(f"{k}={v}" for k, v in sorted(kinds.items())))

    print("\n--- structural integrity ---")
    seen: dict[str, list[str]] = {}
    for doc_id, r in corpus.items():
        seen.setdefault(r["section_id"], []).append(doc_id)
    dupes = {k: v for k, v in seen.items() if len(v) > 1}
    if dupes:
        for sid, ids in list(dupes.items())[:10]:
            failures.append(f"DUPLICATE {sid} as doc_ids {ids}")
            print(f"  FAIL  duplicate section_id {sid} -> {ids}")
        if len(dupes) > 10:
            print(f"  ... and {len(dupes) - 10} more duplicated section_ids")
    else:
        print(f"  ok    no duplicate section_ids ({len(seen)} unique)")

    no_subpart = [r for r in corpus.values() if not r["subpart"]]
    frac = len(no_subpart) / max(len(corpus), 1)
    if frac > 0.05:
        failures.append(f"{frac:.0%} of sections have no subpart - ancestry lookup broken")
        print(f"  FAIL  {len(no_subpart)} sections ({frac:.0%}) have an empty subpart")
    else:
        print(f"  ok    subpart populated on {1 - frac:.0%} of sections")

    if len(corpus) < MIN_SECTIONS:
        failures.append(f"only {len(corpus)} sections - part 1926 should have many more")
        print(f"  FAIL  only {len(corpus)} sections parsed")

    print("\n--- golden text ---")
    by_section = {r["section_id"]: r for r in corpus.values()}
    for section_id, needle in GOLDEN_ASSERTIONS:
        rec = by_section.get(section_id)
        if rec is None:
            failures.append(f"MISSING SECTION  {section_id}")
            print(f"  FAIL  {section_id:<12} section not in corpus")
        elif needle.lower() in rec["full_text"].lower():
            print(f"  ok    {section_id:<12} {needle!r}")
        else:
            failures.append(f"MISSING TEXT     {section_id}: {needle!r}")
            print(f"  FAIL  {section_id:<12} {needle!r} not found "
                  f"({rec['stats']['char_count']} chars)")

    print("\n--- citation reconstruction ---")
    all_official = {s["official_subsection_id"]
                    for r in corpus.values() for s in r["subsections"].values()}
    for cite in GOLDEN_DESIGNATORS:
        if cite in all_official:
            print(f"  ok    {cite}")
        else:
            warnings.append(f"citation {cite} not reconstructed")
            print(f"  warn  {cite} not found")

    bad_paths = [o for o in all_official if re.match(r"^\d+\.\d+\(\d", o)]
    if bad_paths:
        failures.append(
            f"{len(bad_paths)} citations start at a numeric level, e.g. "
            f"{sorted(bad_paths)[:3]} - a paragraph cannot begin at (1)")
        print(f"  FAIL  {len(bad_paths)} citations skip their letter parent, "
              f"e.g. {sorted(bad_paths)[:3]}")
    else:
        print("  ok    no citation begins at a numeric level")

    if depth > 8:
        failures.append(
            f"deepest nesting is level {depth}; the CFR does not nest that far, "
            "so the designator stack is running away")
        deep = sorted(((r["stats"].get("max_depth", 0), r["section_id"])
                       for r in corpus.values()), reverse=True)[:5]
        print(f"  FAIL  nesting reaches level {depth}, deepest sections {deep}")
    else:
        print(f"  ok    nesting depth {depth} is within CFR limits")

    print("\n--- skeletal section scan ---")
    print(f"  (reserved/removed sections excluded: "
          f"{sum(1 for r in corpus.values() if r['section_kind'] == 'reserved')})")
    skeletal = [r for r in corpus.values()
                if r["section_kind"] == "operative"
                and r["stats"]["char_count"] < MIN_OPERATIVE_CHARS]
    for r in sorted(skeletal, key=lambda x: x["stats"]["char_count"])[:15]:
        warnings.append(f"thin: {r['section_id']}")
        print(f"  warn  {r['section_id']:<14} {r['stats']['char_count']:>5} chars  "
              f"{r['title'][:42]}")
    if not skeletal:
        print("  none")
    elif len(skeletal) > 15:
        print(f"  ... and {len(skeletal) - 15} more")

    ratio = designated / max(total_paras, 1)
    if ratio < MIN_DESIGNATED_RATIO:
        failures.append(f"only {ratio:.0%} of paragraphs carry a designator")

    print(f"\n{'=' * 68}")
    if failures:
        print(f"FAILED  {len(failures)} blocking problem(s):")
        for f in failures[:20]:
            print(f"  - {f}")
        print("Do NOT index this corpus.")
        return 1
    print(f"PASSED  {len(warnings)} warning(s). Corpus is safe to index.")
    return 0


# --------------------------------------------------------------------------
# cli
# --------------------------------------------------------------------------

def explain_bad_paths(corpus: dict[str, dict[str, Any]], limit: int = 12) -> int:
    """
    Print the paragraphs whose citation lost its letter root, each with the
    paragraph before it. The preceding paragraph is where the root was
    supposed to come from, so its raw shape is the evidence we need.
    """
    shown = 0
    print(f"\n{'=' * 68}\nROOTLESS CITATIONS - RAW CONTEXT\n{'=' * 68}")
    for rec in corpus.values():
        subs = sorted(rec["subsections"].values(), key=lambda s: s["ordinal"])
        for idx, s in enumerate(subs):
            o = s["official_subsection_id"]
            if not o or not re.match(r"^\d+\.\d+\(\d", o):
                continue
            prev = subs[idx - 1] if idx else None
            print(f"\n--- {o}   ({rec['section_id']}, ordinal {s['ordinal']}) ---")
            if prev:
                print(f"  PREV [{prev['official_subsection_id'] or '-'}] "
                      f"{prev['text'][:260]}")
            else:
                print("  PREV (this is the first paragraph in the section)")
            print(f"  THIS {s['text'][:260]}")
            shown += 1
            if shown >= limit:
                print(f"\n... stopping at {limit}. "
                      "Send this block back and the shape will be obvious.")
                return 0
    if not shown:
        print("  none - every citation has a letter root")
    return 0


def tag_census(xml_path: Path, part: str) -> int:
    """
    Print every element tag that appears inside the part, with counts.

    Use this when something is coming back empty - "tables extracted 0" means
    either the tables are named something other than GPOTABLE in this feed, or
    they are not where we are looking. Guessing is slower than counting.
    """
    from collections import Counter

    tags: Counter = Counter()
    inside_table: Counter = Counter()
    for raw in iter_part_sections(xml_path, part):
        for node in raw["element"].iter():
            tags[node.tag] += 1
            if node.tag in ("GPOTABLE", "TABLE"):
                for child in node.iter():
                    inside_table[child.tag] += 1

    print(f"\nelement tags inside part {part}")
    print("-" * 44)
    for tag, n in tags.most_common(60):
        mark = "  <-- table?" if "TAB" in tag.upper() or "ROW" in tag.upper() else ""
        print(f"  {tag:<24} {n:>7}{mark}")
    if inside_table:
        print("\ntags inside table elements")
        print("-" * 44)
        for tag, n in inside_table.most_common(30):
            print(f"  {tag:<24} {n:>7}")
    else:
        print("\nno GPOTABLE/TABLE elements found at all")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="corpus_1926.json")
    ap.add_argument("--part", default=DEFAULT_PART)
    ap.add_argument("--cache-dir", default=".ecfr_cache", type=Path)
    ap.add_argument("--xml", type=Path, help="use a local XML file instead of downloading")
    ap.add_argument("--force-download", action="store_true")
    ap.add_argument("--legacy-header", action="store_true")
    ap.add_argument("--map-doc-ids-from", type=Path)
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--tag-census", action="store_true",
                    help="print the element-tag histogram inside the part and exit")
    ap.add_argument("--explain-bad-paths", action="store_true",
                    help="dump rootless citations with their preceding paragraph")
    args = ap.parse_args(argv)

    out_path = Path(args.out)

    if args.validate_only or args.explain_bad_paths:
        corpus = json.loads(out_path.read_text(encoding="utf-8"))
        if args.explain_bad_paths:
            return explain_bad_paths(corpus)
        return validate(corpus)

    if args.tag_census:
        xml_path = args.xml or download_bulk_xml(args.cache_dir, args.force_download)
        return tag_census(xml_path, args.part)

    doc_id_map = None
    if args.map_doc_ids_from:
        old = json.loads(args.map_doc_ids_from.read_text(encoding="utf-8"))
        doc_id_map = {rec["section_id"]: str(rec.get("doc_id", k))
                      for k, rec in old.items() if rec.get("section_id")}
        print(f"[ids] reusing {len(doc_id_map)} doc_ids from {args.map_doc_ids_from}")

    xml_path = args.xml or download_bulk_xml(args.cache_dir, args.force_download)

    print(f"[parse] part {args.part} from {xml_path}")
    corpus = build_corpus(xml_path, args.part, args.legacy_header, doc_id_map)
    print(f"[parse] {len(corpus)} sections")

    out_path.write_text(json.dumps(corpus, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"[write] {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    return validate(corpus)


if __name__ == "__main__":
    sys.exit(main())