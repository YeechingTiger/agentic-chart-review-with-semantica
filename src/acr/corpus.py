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
from dataclasses import dataclass, asdict
from datetime import date
from functools import lru_cache
from pathlib import Path

FILENAME_RE = re.compile(
    r"^(?P<doc_type>.+?)_(?P<date>\d{4}-\d{2}-\d{2})(?:__(?P<seq>\d+))?$"
)

# A section header is an ALL-CAPS (or Title-cased all-caps-ish) line ending in a colon.
SECTION_RE = re.compile(r"^(?P<name>[A-Z][A-Z0-9 /&'\-]{2,60}):\s*$", re.MULTILINE)


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

    def sections(self, note_id: str) -> list[str]:
        return [m.group("name").strip() for m in SECTION_RE.finditer(self._text(note_id))]

    def read_section(self, note_id: str, section: str) -> dict:
        """Return one named section (e.g. IMPRESSION, FINAL DIAGNOSIS) with true offsets."""
        text = self._text(note_id)
        marks = list(SECTION_RE.finditer(text))
        target = section.strip().upper()
        for i, m in enumerate(marks):
            if m.group("name").strip().upper() == target:
                start = m.end()
                end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
                meta = self._docs[note_id]
                return {
                    "note_id": note_id,
                    "doc_type": meta.doc_type,
                    "date": meta.date.isoformat(),
                    "section": m.group("name").strip(),
                    "start": start,
                    "end": end,
                    "text": text[start:end].strip("\n"),
                }
        return {
            "note_id": note_id,
            "section": section,
            "error": "section not found",
            "available_sections": [m.group("name").strip() for m in marks],
        }

    def search(
        self,
        query: str,
        regex: bool = False,
        doc_type_contains: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        context: int = 160,
        max_hits: int = 40,
    ) -> list[SearchHit]:
        pattern = re.compile(query if regex else re.escape(query), re.IGNORECASE)
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
