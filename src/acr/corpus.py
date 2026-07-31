"""Corpus access layer.

The platform stores one flat directory per patient containing plain-text documents named

    <DocType>_<YYYY-MM-DD>[__<n>].txt

This module turns that convention into a queryable chart. It is deliberately the *only*
place that knows about the on-disk layout: swap this class to point at a different backend
(FHIR server, object store, database) and the agent and tools are unchanged.

Nothing here calls an LLM. Listing, searching and sectioning are cheap, deterministic
operations — that is what makes coverage attestation auditable.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path

FILENAME_RE = re.compile(
    r"^(?P<doc_type>.+?)_(?P<date>\d{4}-\d{2}-\d{2})(?:__(?P<seq>\d+))?$"
)

#: Separators a clinical phrase may be written with, in the query and in the text.
#:
#: Any run of whitespace (a hard line wrap included), any dash, or a solidus. Used only to relate
#: one SPELLING of a phrase to another -- never one WORD to another. See `PatientChart.search`.
#:
#: The class started as `[\s\-]+`, which is ASCII-only, and that is a real gap the synthetic
#: corpus cannot show: this generator writes ASCII hyphens, while text that has been through
#: Word, a PDF or a dictation front end carries U+2013 and U+2014 routinely. `non-small` did not
#: find `non\u2013small`.
#:
#: The solidus is here because clinical prose runs on it -- `c/w`, `s/p`, `w/o`, `R/O` -- and a
#: query written either way should reach text written the other. It costs nothing: it can only
#: relate two spellings of the same token sequence, never two different sequences.
_SEP_CHARS = r"\s\-\u2010-\u2015\u2212/"
_SEPARATORS = re.compile(f"[{_SEP_CHARS}]+")

#: Apostrophes and quotes that denote the same mark. Word turns `'` into U+2019 on the way in, so
#: `patient's` in a query and `patient\u2019s` in the text are the same phrase written twice.
#:
#: Expressed as a CHARACTER CLASS IN THE PATTERN and never by normalising the text, which is the
#: constraint every tolerance here works under: `quote()` slices the file by the offsets `search`
#: reported, so a matcher that rewrote the text would hand back offsets into a string nobody
#: else has.
_QUOTE_CHARS = "'\u2018\u2019\u02bc\u00b4`"
_QUOTE_CLASS = "[" + re.escape(_QUOTE_CHARS) + "]"

_MONTHS = ("january", "february", "march", "april", "may", "june",
           "july", "august", "september", "october", "november", "december")
_ISO_DATE = re.compile(r"\A(\d{4})-(\d{2})-(\d{2})\Z")


def _date_alternation(query: str) -> str | None:
    """If the whole query is an ISO date, a pattern matching how that day is actually written.

    WHY THIS IS NOTATION AND NOT VOCABULARY, which is the line this module refuses to cross
    everywhere else. A synonym set is open, contested and incomplete -- 23.9% of this corpus's
    diagnosis-bearing documents contain none of the seven commonest words for cancer, so any
    fixed list guesses on the model's behalf and hides the miss inside a hit. The renderings of
    one calendar day are none of those things: the set is closed, decidable, and every member
    denotes exactly the same thing. Nothing is being folded that a reader would distinguish.
    
    It earns its place because `STORE.390`'s answer IS a date. A run that finds the retrospective
    remark "the 3/12/19 nodule is this same tumour" and then searches `2019-03-12` for the study
    it names gets nothing, and the failure looks like an absent document.

    Returns None for anything that is not a bare ISO date, so every other query is untouched.
    """
    m = _ISO_DATE.match(query.strip())
    if not m:
        return None
    y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if not (1 <= mo <= 12 and 1 <= d <= 31):
        return None
    name = _MONTHS[mo - 1]
    sep = f"[{_SEP_CHARS}.]+"          # date parts also get written with dots: 12.03.2019
    forms = [
        rf"{y}{sep}0?{mo}{sep}0?{d}",                       # 2019-03-12, 2019/3/12
        rf"0?{mo}{sep}0?{d}{sep}{y}",                       # 3/12/2019
        rf"0?{mo}{sep}0?{d}{sep}{y % 100:02d}",             # 3/12/19
        rf"0?{d}{sep}0?{mo}{sep}{y}",                       # 12/3/2019, day-first
        rf"{y}{mo:02d}{d:02d}",                             # 20190312
        rf"{name}{sep}0?{d}(?:,)?{sep}{y}",                 # March 12, 2019
        rf"{name[:3]}\.?{sep}0?{d}(?:,)?{sep}{y}",          # Mar 12, 2019 / Mar. 12 2019
        rf"0?{d}{sep}{name}{sep}{y}",                       # 12 March 2019
    ]
    return "(?:" + "|".join(forms) + ")"


#: A token this short is not a word, it is a fragment, and joining fragments with a permissive
#: separator matches nearly everywhere. Measured the moment the solidus went into the separator
#: class: `s/p` became `s[sep]+p`, which found 43 documents where a literal found one — and every
#: one of the 43 was noise like "lungs Plan" and "masses present". The gain column said +43 and
#: the tolerance was worse than useless.
#:
#: So a pattern whose shortest token is a fragment gets word boundaries. `s/p` then reaches
#: "s/p", "s p" and "s-p" and stops reaching the inside of other words. Anchors are NOT added
#: otherwise, because the documented substring behaviour depends on their absence: `lobe` must
#: keep matching inside `lobes`, and a trailing `\b` would end that.
_FRAGMENT_LEN = 2


def _notation_tolerant(query: str) -> str:
    """A literal query, with its separators allowed to be spelled any of the ways this corpus
    spells them. Returns a regex source string.

    Single-token queries come back as an escaped literal with only its quote marks widened, so
    substring behaviour is unchanged: `lobe` still matches inside `lobes`, as it always did, and
    `nonsmall` still does not find `non-small` -- that needs a word list, which is the thing
    this module does not have.
    """
    if (dates := _date_alternation(query)) is not None:
        return dates
    raw = [tok for tok in _SEPARATORS.split(query.strip()) if tok]
    if not raw:
        return _widen_quotes(re.escape(query))
    joined = _SEPARATORS.pattern.join(_widen_quotes(re.escape(tok)) for tok in raw)
    if len(raw) > 1 and min(len(tok) for tok in raw) <= _FRAGMENT_LEN:
        return rf"\b{joined}\b"
    return joined


#: Matches any quote mark, and is applied in ONE pass. A loop of `str.replace` per character
#: corrupts its own output: the first pass inserts a class that CONTAINS the marks the later
#: passes go looking for, so the second replacement rewrites the inside of the class the first
#: one built and the result matches nothing it should.
_ANY_QUOTE = re.compile("[" + re.escape(_QUOTE_CHARS) + "]")


def _widen_quotes(escaped: str) -> str:
    """Let any apostrophe in an already-escaped literal match any of the marks it is written as.

    Runs on the ESCAPED literal, where a quote mark is still a single literal character on every
    Python version — `re.escape` has not escaped apostrophes since 3.7, but it is not this
    function's business to know that.
    """
    return _ANY_QUOTE.sub(_QUOTE_CLASS.replace("\\", "\\\\"), escaped)


@dataclass(frozen=True)
class DocMeta:
    note_id: str          # filename stem — stable, human-readable, citable
    doc_type: str         # e.g. "Surgical-Pathology-Report"
    date: date
    seq: int              # 1 unless the filename carried a __N suffix
    n_chars: int

    def to_dict(self) -> dict:
        d = asdict(self)
        d["date"] = self.date.isoformat()
        return d


@dataclass(frozen=True)
class SearchHit:
    note_id: str
    doc_type: str
    date: str
    start: int
    end: int
    snippet: str


class ChartNotFoundError(FileNotFoundError):
    pass


def parse_filename(stem: str) -> tuple[str, date, int] | None:
    """`Head-CT-WWO-Contr_2023-05-23__2` -> ("Head-CT-WWO-Contr", date(2023,5,23), 2)."""
    m = FILENAME_RE.match(stem)
    if not m:
        return None
    y, mo, d = (int(x) for x in m.group("date").split("-"))
    return m.group("doc_type"), date(y, mo, d), int(m.group("seq") or 1)


class PatientChart:
    """All documents for one patient, indexed by the filename convention."""

    def __init__(self, patient_dir: Path):
        self.dir = Path(patient_dir)
        if not self.dir.is_dir():
            raise ChartNotFoundError(f"no such patient directory: {self.dir}")
        self.patient_id = self.dir.name
        self._docs: dict[str, DocMeta] = {}
        self._paths: dict[str, Path] = {}
        for p in sorted(self.dir.glob("*.txt")):
            parsed = parse_filename(p.stem)
            if parsed is None:
                continue  # unparseable filenames are skipped, not guessed at
            doc_type, dt, seq = parsed
            self._docs[p.stem] = DocMeta(p.stem, doc_type, dt, seq, p.stat().st_size)
            self._paths[p.stem] = p

    # -- metadata -----------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._docs)

    @property
    def doc_types(self) -> list[str]:
        return sorted({d.doc_type for d in self._docs.values()})

    def list_documents(
        self,
        doc_type_contains: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[DocMeta], int]:
        """Metadata only — never returns document text. Returns (page, total_matching)."""
        out = list(self._docs.values())
        if doc_type_contains:
            needle = doc_type_contains.lower()
            out = [d for d in out if needle in d.doc_type.lower()]
        if date_from:
            out = [d for d in out if d.date.isoformat() >= date_from]
        if date_to:
            out = [d for d in out if d.date.isoformat() <= date_to]
        out.sort(key=lambda d: (d.date, d.doc_type, d.seq))
        return out[offset : offset + limit], len(out)

    def type_summary(self) -> list[dict]:
        """Cheap triage view: counts and date span per document type."""
        by: dict[str, list[DocMeta]] = {}
        for d in self._docs.values():
            by.setdefault(d.doc_type, []).append(d)
        rows = []
        for t, ds in by.items():
            dates = sorted(d.date for d in ds)
            rows.append(
                {
                    "doc_type": t,
                    "count": len(ds),
                    "earliest": dates[0].isoformat(),
                    "latest": dates[-1].isoformat(),
                }
            )
        rows.sort(key=lambda r: (-r["count"], r["doc_type"]))
        return rows

    def timeline(self, doc_type_contains: str | None = None, limit: int = 200) -> list[dict]:
        docs, _ = self.list_documents(doc_type_contains=doc_type_contains, limit=limit)
        return [{"date": d.date.isoformat(), "doc_type": d.doc_type, "note_id": d.note_id} for d in docs]

    # -- content ------------------------------------------------------------------
    @lru_cache(maxsize=512)
    def _text(self, note_id: str) -> str:
        if note_id not in self._paths:
            raise KeyError(note_id)
        return self._paths[note_id].read_text(encoding="utf-8", errors="replace")

    def read(self, note_id: str, offset: int = 0, limit: int = 4000) -> dict:
        """Paginated read with stable character offsets so spans stay citable."""
        text = self._text(note_id)
        chunk = text[offset : offset + limit]
        meta = self._docs[note_id]
        return {
            "note_id": note_id,
            "doc_type": meta.doc_type,
            "date": meta.date.isoformat(),
            "offset": offset,
            "returned_chars": len(chunk),
            "total_chars": len(text),
            "truncated": offset + len(chunk) < len(text),
            "text": chunk,
        }

    # ------------------------------------------------------------------ sections: REMOVED
    # `sections()` and `read_section()` are gone, and so is the `SECTION_RE` they ran on:
    #
    #     SECTION_RE = re.compile(r"^(?P<name>[A-Z][A-Z0-9 /&'\-]{2,60}):\s*$", re.MULTILINE)
    #
    # It required an ALL-CAPS line with nothing after the colon. Measured over the 12,221
    # diagnosis-bearing documents in this corpus -- documents containing the phrase, against
    # documents where `read_section` could actually address it:
    #
    #     final diagnosis               7390 contain      170 addressable    2.3%
    #     final pathologic diagnosis    6262 contain      332 addressable    5.3%
    #     microscopic description       5914 contain      479 addressable    8.1%
    #     diagnosis                    12082 contain       42 addressable    0.3%
    #     addendum                      2401 contain        0 addressable    0.0%
    #
    # The reason is in the real spellings. Title Case dominates and the regex admits none of it:
    #
    #     no    3899  'Pre-Operative Diagnosis:'      YES   308  'FINAL PATHOLOGIC DIAGNOSIS:'
    #     no    2807  'Final Diagnosis:'              YES   104  'FINAL DIAGNOSIS:'
    #     no    2154  'Final Cytologic Diagnosis:'    no     28  '***DIAGNOSIS***'
    #
    # Two consequences, and the second is worse than low recall. `ADDENDUM` was addressable in
    # 0 of 2,401 documents while the thread machinery was refusing answers 40 times and telling
    # the agent to go chase an addendum -- an obligation whose tool could never reach its
    # target. And `read_section` with no section returned `available_sections`, so in 97.7% of
    # documents the model asked what was in the file and got back a list with the final
    # diagnosis missing from it. That is not a tool failing to help; it is a tool answering
    # wrongly.
    #
    # THE REPLACEMENT IS ALREADY HERE and needs no vocabulary: `search` returns the offset of
    # every hit, and `read` takes an offset and a limit. "Jump to the diagnosis" is a search
    # inside one document followed by a read around the hit, composed by the model, and it works
    # whatever the heading looks like -- or whether there is a heading at all.
    #
    # It also removes a state: `read_section` reported true offsets and no document length, and
    # that was the ONLY source of `coverage_planner.READ_STATE_LENGTH_UNKNOWN` -- a run that had
    # only ever read sections could not tell that it had left a hole. Every read is now a `read`
    # with offsets, so total length is always known and `truncated` is always computable.

    def search(
        self,
        query: str,
        regex: bool = False,
        doc_type_contains: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        context: int = 250,
        max_hits: int = 40,
    ) -> list[SearchHit]:
        """Find `query` in this patient's documents. The model chooses the term; this finds it.

        NOTATION-TOLERANT, NOT SYNONYM-AWARE, and the line between those two is the point.

        A literal `re.escape(query)` made three different searches out of one clinical phrase,
        because this corpus writes the same thing three ways: `non-small cell`, `non small
        cell`, and the same phrase broken across a hard line wrap. So a run of whitespace or
        hyphen in the QUERY now matches any run of whitespace or hyphen in the text, and the
        match stays case-insensitive.

        Measured over the 12,190 diagnosis-bearing documents in this corpus, documents found:

            non-small cell            2251 -> 2476   +225  (+10.0%)
            small cell carcinoma      1537 -> 1552    +15
            right upper lobe          1270 -> 1277     +7
            right lower lobe           876 ->  883     +7
            squamous cell carcinoma   2615 -> 2623     +8
            ten phrases, total       16797 -> 17064   +267   (+1.6%)

        So the line-wrap case is small and the hyphen case is real. Both are free.

        WHAT THIS DELIBERATELY DOES NOT DO is fold synonyms. Same measurement, single terms,
        share of diagnosis-bearing documents each one appears in:

            carcinoma 57.5%   malignan 50.1%   tumor 36.8%   cancer 32.7%
            adenocarcinoma 27.6%   neoplasm 4.3%   tumour 0.0%

        677 documents (5.6%) contain `cancer` and not `carcinoma`, and 23.9% contain none of
        those seven. No fixed list is close to complete, so a matcher that silently expanded
        `cancer` into a synonym set would be guessing on the model's behalf and hiding the miss
        rate inside a hit. Recall comes from the model issuing several searches and reading the
        hits -- which is what it has this tool for -- not from a vocabulary compiled in here.

        Nor does it split inside a token: query `nonsmall` still will not find `non-small`.
        That needs a word list, which is the thing above.

        `regex=True` is honoured verbatim: a caller that wrote its own pattern gets it.
        """
        pattern = re.compile(query if regex else _notation_tolerant(query), re.IGNORECASE)
        docs, _ = self.list_documents(
            doc_type_contains=doc_type_contains, date_from=date_from, date_to=date_to, limit=10_000
        )
        hits: list[SearchHit] = []
        for meta in docs:
            text = self._text(meta.note_id)
            for m in pattern.finditer(text):
                s = max(0, m.start() - context)
                e = min(len(text), m.end() + context)
                hits.append(
                    SearchHit(
                        note_id=meta.note_id,
                        doc_type=meta.doc_type,
                        date=meta.date.isoformat(),
                        start=m.start(),
                        end=m.end(),
                        snippet=text[s:e].replace("\n", " ").strip(),
                    )
                )
                if len(hits) >= max_hits:
                    return hits
        return hits

    def quote(self, note_id: str, start: int, end: int) -> str:
        return self._text(note_id)[start:end]


class Corpus:
    """A directory of patient directories."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def patient_ids(self) -> list[str]:
        return sorted(p.name for p in self.root.iterdir() if p.is_dir())

    def chart(self, patient_id: str) -> PatientChart:
        return PatientChart(self.root / patient_id)

    # -- corpus-wide document-type vocabulary -------------------------------------
    VOCAB_CACHE = ".doc_type_vocab.json"

    def doc_type_vocabulary(self, use_cache: bool = True) -> list[str]:
        """Every document type appearing anywhere in the corpus.

        Needed so that "this patient has no pathology report" (a finding) stays separable
        from "no such document type exists" (a typo). But it is a property of the CORPUS,
        not of the patient under review, so it must not be recomputed per run.

        The obvious implementation — building a PatientChart for every patient and unioning
        `.doc_types` — costs one stat() PER FILE, because PatientChart records
        `p.stat().st_size` for each document. On the real corpus that is ~276,000 stats,
        and Lustre metadata here runs ~8.5 ms per op: about 39 minutes, to run ONE patient.
        Measured: a single-patient run sat 17+ minutes in stat() before its first API call.

        A document's type comes from `parse_filename(p.stem)` — the NAME alone. So this
        reads directory entries and never stats them, then caches the result next to the
        corpus. Cold: seconds. Warm: a file read.

        Delete `<root>/.doc_type_vocab.json` if documents are added with new types.
        """
        import json
        import os

        cache = self.root / self.VOCAB_CACHE
        if use_cache and cache.is_file():
            try:
                v = json.loads(cache.read_text())
                if isinstance(v, list) and v:
                    return v
            except (json.JSONDecodeError, OSError):
                pass  # a corrupt cache should be rebuilt, not fatal

        types: set[str] = set()
        with os.scandir(self.root) as patients:
            for pent in patients:
                if not pent.is_dir():
                    continue
                try:
                    with os.scandir(pent.path) as docs:
                        for dent in docs:
                            name = dent.name
                            if not name.endswith(".txt"):
                                continue
                            parsed = parse_filename(name[:-4])
                            if parsed is not None:
                                types.add(parsed[0])
                except OSError:
                    continue  # unreadable patient dir is not a reason to fail the run

        vocab = sorted(types)
        if use_cache:
            try:
                tmp = cache.with_suffix(".tmp")
                tmp.write_text(json.dumps(vocab, indent=0))
                os.replace(tmp, cache)   # atomic: never leave a half-written cache
            except OSError:
                pass  # a read-only corpus is fine; we just pay the scan next time
        return vocab

    COUNT_CACHE = ".doc_type_counts.json"

    def doc_type_counts(self, use_cache: bool = True) -> dict[str, int]:
        """Every document type in the corpus, and how many documents carry it.

        Same scan and the same reason as `doc_type_vocabulary` -- directory entries only,
        never `stat()`, because a stat per file is ~276,000 metadata ops and 39 minutes on
        this filesystem. The counts are what `site_mapping` needs on top of the names: they
        order a 1,516-row review table by how much of the corpus each row decides, so a
        registrar with one hour spends it on `Surgical-Pathology-Document` (3,849 documents)
        rather than on `ELECTRONYSTAGMOGRAM` (1).

        Cached separately from the vocabulary so an existing warm vocabulary cache is not
        invalidated by asking a new question of the same scan.
        """
        import json
        import os
        from collections import Counter

        cache = self.root / self.COUNT_CACHE
        if use_cache and cache.is_file():
            try:
                v = json.loads(cache.read_text())
                if isinstance(v, dict) and v:
                    return {str(k): int(n) for k, n in v.items()}
            except (json.JSONDecodeError, OSError, TypeError, ValueError):
                pass

        counts: Counter[str] = Counter()
        with os.scandir(self.root) as patients:
            for pent in patients:
                if not pent.is_dir():
                    continue
                try:
                    with os.scandir(pent.path) as docs:
                        for dent in docs:
                            name = dent.name
                            if not name.endswith(".txt"):
                                continue
                            parsed = parse_filename(name[:-4])
                            if parsed is not None:
                                counts[parsed[0]] += 1
                except OSError:
                    continue

        out = dict(sorted(counts.items()))
        if use_cache:
            try:
                tmp = cache.with_suffix(".tmp")
                tmp.write_text(json.dumps(out, indent=0))
                os.replace(tmp, cache)
            except OSError:
                pass
        return out
