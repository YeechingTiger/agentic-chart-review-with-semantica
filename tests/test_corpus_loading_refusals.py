"""What the chart loader does with files it cannot read as documents.

Two defects found by the 2026-08-03 pipeline audit, both in the enumeration FEEDING
`unreadable_filenames` rather than in the mechanism itself:

1. `FILENAME_RE` matches on digit SHAPE and then `date(y, mo, d)` was constructed unguarded, so
   `Note_0000-00-00.txt` — the standard MySQL/EHR null-date sentinel — raised an uncaught
   `ValueError` from `PatientChart.__init__`, making the whole chart unconstructible. One such file
   in one patient killed `acr run`, `acr batch`, `acr chart` and `acr check-corpus` for that
   patient, with a traceback instead of a report.

2. The universe was `glob("*.txt")`, so a file that is not lowercase-`.txt` (`.TXT`, `.text`,
   `.doc`) was never ENUMERATED — it could not reach `unreadable_filenames`, and a chart consisting
   entirely of such files reported as clean with zero documents. "Report every document the loader
   cannot see" was false for exactly the files most likely to exist in a real export.
"""

from __future__ import annotations

import pytest

from acr.chartstore.corpus import PatientChart, parse_filename


@pytest.mark.parametrize("stem", [
    "Note_0000-00-00",        # the null-date sentinel EHR exports actually contain
    "Note_2021-13-45",        # month out of range
    "Note_2021-02-30",        # day out of range for month
])
def test_a_calendar_invalid_date_is_unreadable_not_a_crash(stem):
    assert parse_filename(stem) is None


def test_a_chart_with_a_null_date_file_still_loads(tmp_path):
    d = tmp_path / "P1"
    d.mkdir()
    (d / "Note_0000-00-00.txt").write_text("x", encoding="utf-8")
    (d / "Path_2023-01-02.txt").write_text("y", encoding="utf-8")

    chart = PatientChart(d)   # must not raise

    assert len(chart) == 1
    assert chart.unreadable_filenames == ["Note_0000-00-00.txt"]


def test_a_wrong_suffix_is_reported_not_invisible(tmp_path):
    """`.TXT` and `.text` were outside the glob, so a chart of only such files read as CLEAN."""
    d = tmp_path / "P2"
    d.mkdir()
    (d / "Discharge_2021-01-02.TXT").write_text("x", encoding="utf-8")
    (d / "Path_2021-01-03.text").write_text("y", encoding="utf-8")

    chart = PatientChart(d)

    assert len(chart) == 0
    assert sorted(chart.unreadable_filenames) == [
        "Discharge_2021-01-02.TXT", "Path_2021-01-03.text"]


def test_sidecars_and_hidden_files_are_not_documents(tmp_path):
    """`_ground_truth.json` is this corpus's own sidecar convention and `.DS_Store` is every
    macOS directory's. Flagging either would make the shipped corpus — and any real Mac corpus —
    report unreadable files forever, which trains people to ignore the report."""
    d = tmp_path / "P3"
    d.mkdir()
    (d / "Path_2021-01-03.txt").write_text("y", encoding="utf-8")
    (d / "_ground_truth.json").write_text("{}", encoding="utf-8")
    (d / ".DS_Store").write_bytes(b"\x00")

    chart = PatientChart(d)

    assert len(chart) == 1
    assert chart.unreadable_filenames == []
