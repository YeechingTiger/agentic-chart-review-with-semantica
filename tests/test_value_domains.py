"""Enumerated value domains at the tool interface.

An empty result has two possible causes with opposite remedies: the name you asked for does
not exist (retry), or it exists and this patient has none (that is a finding, often the
answer). Returning `[]` for both is how a typo becomes a silent false negative — an agent
that asks for "Biopsy", gets nothing, and concludes there is no pathology is wrong and has
no way to notice.

The domain has to be corpus-wide. Scoping it to the patient's own types inverts the whole
point: "this patient has no pathology" would come back as a query error.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from acr.corpus import Corpus
from acr.coverage import CoverageLedger
from acr.state import EvidenceLedger
from acr.tools import Toolbox

CORPUS = Path(__file__).resolve().parents[1] / "corpus" / "patients"


@pytest.fixture(scope="module")
def vocabulary() -> list[str]:
    c = Corpus(CORPUS)
    return sorted({t for pid in c.patient_ids() for t in c.chart(pid).doc_types})


@pytest.fixture
def tb(vocabulary):
    # SYN0002's biopsy was done at an outside hospital, so it has NO pathology documents.
    chart = Corpus(CORPUS).chart("SYN0002")
    return Toolbox(chart, EvidenceLedger(), CoverageLedger(list(chart._docs.values()), []), known_doc_types=vocabulary)


def test_out_of_domain_type_is_an_error_not_an_empty_list(tb):
    out, _ = tb.dispatch("list_documents", {"doc_type_contains": "Biopsy"})
    assert out["error"] == "UNKNOWN_DOC_TYPE"
    assert out["queried"] == "Biopsy"
    assert out["known_types"], "the caller must be handed the actual domain"
    assert "not evidence" in out["message"].lower()


def test_in_domain_but_patient_has_none_is_a_finding(tb):
    """The case that matters: SYN0002 really has no pathology, and that is the answer."""
    out, _ = tb.dispatch("list_documents", {"doc_type_contains": "Pathology"})
    assert "error" not in out
    assert out["total_matching"] == 0
    assert out["type_filter_valid"] is True
    assert out["type_exists_but_empty"] is True


def test_in_domain_with_matches(tb):
    out, _ = tb.dispatch("list_documents", {"doc_type_contains": "Abd-Pelvis-CT"})
    assert out["total_matching"] >= 1
    assert out["type_exists_but_empty"] is False


def test_domain_is_corpus_wide_not_patient_scoped(vocabulary):
    """Without a corpus vocabulary the two cases become indistinguishable again — so the
    fallback has to admit that in the error rather than pretend to be authoritative."""
    chart = Corpus(CORPUS).chart("SYN0002")
    narrow = Toolbox(chart, EvidenceLedger(), CoverageLedger(list(chart._docs.values()), []))          # no vocabulary
    wide = Toolbox(chart, EvidenceLedger(), CoverageLedger(list(chart._docs.values()), []), known_doc_types=vocabulary)

    n_out, _ = narrow.dispatch("list_documents", {"doc_type_contains": "Pathology"})
    w_out, _ = wide.dispatch("list_documents", {"doc_type_contains": "Pathology"})

    assert n_out.get("error") == "UNKNOWN_DOC_TYPE"          # the failure mode being fixed
    assert n_out["domain"] == "patient_chart_only"
    assert "cannot be told apart" in n_out["message"]
    assert "error" not in w_out and w_out["type_exists_but_empty"] is True


@pytest.mark.parametrize("tool", ["search_notes", "timeline"])
def test_every_tool_taking_a_doc_type_enforces_the_domain(tb, tool):
    args = {"doc_type_contains": "Biopsy"}
    if tool == "search_notes":
        args["query"] = "carcinoma"
    out, _ = tb.dispatch(tool, args)
    assert out["error"] == "UNKNOWN_DOC_TYPE"


def test_fabricated_note_id_is_rejected_with_the_domain(tb):
    out, _ = tb.dispatch("read_document", {"note_id": "Surgical-Pathology-Report_2022-09-01"})
    assert out["error"] == "UNKNOWN_NOTE_ID"
    assert "do not construct one" in out["message"]


# REMOVED 2026-07-30 with `read_section` itself: `test_unknown_section_is_distinguished_from_an
# _empty_one`. It asserted that an unknown section name came back with `available_sections` so a
# retry could succeed -- and that list was the defect. `SECTION_RE` required an ALL-CAPS line
# with nothing after the colon, so over the 12,221 diagnosis-bearing documents in this corpus it
# could address `FINAL DIAGNOSIS` in 170 of the 7,390 that contain it (2.3%) and `ADDENDUM` in 0
# of 2,401. The domain this test insisted on handing back was therefore a list with the final
# diagnosis missing from it, in 97.7% of documents.
#
# The replacement needs no vocabulary and is asserted below: `search_notes` returns the offset
# of every hit and `read_document` takes an offset.
def test_search_offsets_are_the_addressing_scheme_that_replaced_section_names(tb):
    """search -> offset -> read is the whole replacement, and it needs no heading vocabulary.

    The term is taken FROM the document rather than assumed, because this synthetic fixture is
    part of how the regex survived so long: it writes ALL-CAPS headings (`CLINICAL HISTORY:`,
    `FINDINGS:`) that `SECTION_RE` matched perfectly, while the real corpus writes
    `Final Diagnosis:` 2,807 times and the regex admitted none of it.
    """
    docs, _ = tb.chart.list_documents(limit=1)
    text = tb.chart.read(docs[0].note_id, 0, 4000)["text"]
    term = next(w for w in text.split() if len(w) > 5 and w.isalpha())

    out, _ = tb.dispatch("search_notes", {"query": term, "max_hits": 5})
    assert out["n_hits"] >= 1, f"{term!r} was taken from the document, so it must be findable"
    hit = out["hits"][0]
    assert {"note_id", "doc_type", "date", "start", "end", "snippet"} <= set(hit)

    read, _ = tb.dispatch("read_document", {"note_id": hit["note_id"],
                                            "offset": hit["start"], "limit": 200})
    assert term.lower() in read["text"].lower(), "the offset must land on the hit"
    assert read["total_chars"] > 0, "every read reports length, so `truncated` is computable"
