"""Recover the WHY comments that `yaml.safe_load` throws away.

The house rule in this repo is that a decision is documented at the point it is made, in a
comment. Those comments are the only record of why most of these choices were made, and the
parser discards every one — so a review document built from the parsed spec alone can state
what was decided and never on what basis, which is the one thing a reviewer needs in order
to disagree with it.
"""
from __future__ import annotations

import re
from typing import Sequence

from .prose import plain


_ELEMENT_START = re.compile(r"^\s*(-\s|[A-Za-z_][\w.\"']*\s*:)")
_CODEY = re.compile(r"[`(]\)|`|->|\.py\b|\bP3d\b|§|_check|\bre\.|\(\)")


def comment_map(text: str) -> tuple[list[str], dict[int, str]]:
    """Line -> the comment block sitting directly above it.

    The house rule in this repo is that a decision is documented at the point it is made, in
    a comment. Those comments are the only record of WHY most of these choices were made, and
    `yaml.safe_load` throws every one of them away — so a review document built from the
    parsed spec alone can state what was decided and never on what basis.
    """
    lines = text.splitlines()
    buf: list[str] = []
    attached: dict[int, str] = {}
    for i, ln in enumerate(lines):
        s = ln.strip()
        if s.startswith("#"):
            buf.append(s.lstrip("#").strip())
            continue
        if not s:
            buf = []
            continue
        if buf:
            attached[i] = " ".join(x for x in buf if x).strip()
        buf = []
    return lines, attached


def basis_at(lines: Sequence[str], attached: dict[int, str], needle: str) -> str:
    """The comment above whichever line the content first appears on."""
    probe = re.sub(r"\s+", " ", str(needle)).strip()[:44]
    if len(probe) < 6:
        return ""
    hit = -1
    for i, ln in enumerate(lines):
        if probe in re.sub(r"\s+", " ", ln):
            hit = i
            break
    if hit < 0:
        return ""
    j = hit
    while j >= 0 and not _ELEMENT_START.match(lines[j]):
        j -= 1
    return _clean_comment(attached.get(j if j >= 0 else hit, ""))


def _clean_comment(raw: str) -> str:
    """Keep the sentences a clinician can use; drop the ones addressed to a maintainer.

    A comment like "graph._check_gate loops over required_keywords" is true, load-bearing and
    completely useless to the person being asked whether radiology may localise a primary.
    Sentences naming a function, a file or a backticked identifier go; the rest survives and
    then goes through the jargon map like everything else.
    """
    if not raw:
        return ""
    keep = [s for s in re.split(r"(?<=[.;])\s+", raw)
            if s.strip() and not _CODEY.search(s)]
    out = plain(" ".join(keep))
    return out[:420].rstrip()
