"""The reviewer-facing side of a spec: one import site for the review document, the
statements it is made of, and the sign-offs taken against them.

Almost every genuinely clinical decision in this system is written in YAML. `establishes:
[primary_site]` on a group of imaging documents IS the ruling "radiology may tell us where
the tumour started". It is a sentence a thoracic oncologist would settle in four seconds,
and it was wrong for weeks because it was never a sentence — it was a key. P03 was coded
C349 (lung, NOS) while "right upper lobe" sat in seven other note types, and the spec's own
prose said outright that radiology can localise a mass. Nobody clinical had ever read the
file, because the file is 179 lines of regexes, group names and Clopper-Pearson bounds.

So this package answers one question: what would a certified registrar have to be shown in
order to catch that? Four modules own the four properties that follow, and
`tests/test_specview.py` asserts each one rather than trusting the prose:

  * `prose`       — nothing engineering-shaped reaches the page.
  * `statements`  — the made-up list is complete, and defaults to "we made this up".
  * `signoff`     — a sign-off dies when the text it approved changes.
  * `measurements`— a number is reprinted only beside the configuration it was measured on.

`basis`, `decisions` and `render` build on those: what the file's own comments say a choice
was based on, which choices a clinician could make differently, and the markdown itself.

This file re-exports rather than defines. Callers name `acr.usecase.specview` — the CLI, the tests,
and the docstrings in `acr.contract.spec` that point here — and splitting the module must not turn
into a rewrite of every one of them; a facade is cheaper than a churn of import lines that
would bury the change worth reading.
"""
from __future__ import annotations

from .decisions import Decision, decisions, who_decided
from .measurements import (
                           CORPUS_HEADER,
                           MEASUREMENTS,
                           SAMPLING_ARITHMETIC,
                           UNMEASURED_NOTE,
                           Measurement,
)
from .prose import ICDO_NAMES, JARGON
from .render import SECTION_TITLES, render_review
from .signoff import SIGNED, STALE, UNSIGNED, load_signoffs, record_signoff, signoff_status
from .statements import (
                           MODEL_AUTHORED,
                           Element,
                           SourceGroup,
                           editorial_index,
                           element_ids,
                           elements,
                           provenance_findings,
)

__all__ = [
    "CORPUS_HEADER", "ICDO_NAMES", "JARGON", "MEASUREMENTS", "MODEL_AUTHORED",
    "SAMPLING_ARITHMETIC", "SECTION_TITLES", "SIGNED", "STALE", "UNMEASURED_NOTE", "UNSIGNED",
    "Decision", "Element", "Measurement", "SourceGroup",
    "decisions", "editorial_index", "element_ids", "elements", "load_signoffs",
    "provenance_findings", "record_signoff", "render_review", "signoff_status", "who_decided",
]
