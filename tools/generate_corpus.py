#!/usr/bin/env python3
"""Generate a synthetic, PHI-free chart-review corpus.

Filename convention mirrors the target platform exactly:

    <DocType>_<YYYY-MM-DD>[__<n>].txt

where `__<n>` disambiguates multiple documents of the same type on the same date.
One flat directory per patient. No structured data — notes only, as on the real platform.

Each patient is built around a deliberate *evidence pattern* so the corpus can
exercise the extraction specs (tissue confirmation, evidence gaps, date-boundary
rules, and the 00 / 70 / 99 recurrence trichotomy).

Deterministic: seeded per patient, so regenerating yields byte-identical output.

    python tools/generate_corpus.py --out corpus/patients
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

# --------------------------------------------------------------------------------------
# Document-type vocabulary (matches the real platform's naming)
# --------------------------------------------------------------------------------------
PROGRESS_SPECIALTIES = [
    "Onc-Med-MD-OP-Progress-Note",
    "GI-Gen-MD-OP-Progress-Note",
    "Endo-Diab-MD-OP-Progress-Note",
    "Pulm-MD-OP-Progress-Note",
    "Transfusion-MD-OP-Progress-Note",
]
IMAGING_TYPES = [
    "Chest-CT-W-Contr",
    "Chest-CT-WWO-Contr",
    "Head-CT-W-Contr",
    "Head-CT-WWO-Contr",
    "Abd-Pelvis-CT-W-Contr",
    "PET-CT-Skull-Base-Thigh",
    "Spine-Cervical-2-3V-XR",
]
PATH_TYPES = ["Surgical-Pathology-Report", "Surgical-Pathology-Document"]
OTHER_TYPES = ["EKG", "Endoscopy", "Procedure-Note", "Fluoroscopy-Up-to-1-Hr"]
RX_TYPE = "Prescriptions-Filled-RxHub"

FIRST = ["Jordan", "Alex", "Casey", "Riley", "Morgan", "Avery", "Quinn", "Reese"]
LAST = ["Whitaker", "Alvarez", "Okafor", "Lindqvist", "Baptiste", "Moreau", "Nakamura", "Ferris"]

BACKGROUND_YEARS = 10          # decade of routine care before the index diagnosis


# --------------------------------------------------------------------------------------
# Patient blueprints — each encodes what the ground truth should be and why
# --------------------------------------------------------------------------------------
@dataclass
class Blueprint:
    pid: str
    pattern: str            # human-readable label for the evidence pattern
    site_text: str          # how the primary site is described in prose
    site_code: str          # ICD-O-3 topography ground truth
    histology_text: str
    histology_code: str     # ICD-O-3 morphology ground truth ("" if no tissue)
    behavior: str           # "3" invasive, "2" in situ, "" unknown
    dx_date: str            # ground-truth Date of Initial Diagnosis (CCYYMMDD)
    dx_date_why: str
    tissue: bool            # is there a pathology report in the chart?
    recurrence_type: str    # STORE [1880] ground truth
    recurrence_date: str    # CCYYMMDD or ""
    imaging: list[str] = field(default_factory=list)
    notes: str = ""         # designer commentary carried into ground_truth.json
    # Follow-up shape. This is what the observable-period / empty-window logic is tested on.
    #   normal     regular surveillance to the end
    #   interior   records, then a 2-year hole, then records again  -> true gap
    #   truncated  records stop at last contact and never resume    -> NOT a gap
    followup: str = "normal"
    gap_years: tuple[int, int] = (2, 4)     # years after dx that the interior hole spans
    truncate_after_years: float = 2.0       # for followup == "truncated"
    expect: dict = field(default_factory=dict)   # machine-assertable expectations


BLUEPRINTS = [
    Blueprint(
        pid="SYN0001", pattern="tissue-confirmed; cytology precedes pathology",
        site_text="right upper lobe of lung", site_code="C341",
        histology_text="adenocarcinoma", histology_code="8140", behavior="3",
        dx_date="20230412",
        dx_date_why=("Cytology 2023-04-12 read as 'suspicious for adenocarcinoma' AND an oncology "
                     "note of the same admission records a clinical impression of malignancy, so the "
                     "cytology date is diagnostic per STORE [390]. Core biopsy 2023-04-27 confirms."),
        tissue=True, recurrence_type="00", recurrence_date="",
        imaging=["Chest-CT-W-Contr", "PET-CT-Skull-Base-Thigh"],
        notes="Exercises the [390] cytology-vs-pathology boundary case and a clean 00 (disease-free, no recurrence).",
    ),
    Blueprint(
        pid="SYN0002", pattern="evidence gap — biopsy performed at outside hospital",
        site_text="sigmoid colon", site_code="C187",
        histology_text="", histology_code="", behavior="",
        dx_date="20220906",
        dx_date_why="CT 2022-09-06 shows obstructing mass; GI note same day states 'known colon cancer'.",
        tissue=False, recurrence_type="99", recurrence_date="",
        imaging=["Abd-Pelvis-CT-W-Contr"],
        notes=("Histology must come back EVIDENCE_INSUFFICIENT: the diagnosis is asserted in prose but "
               "the confirming pathology was done elsewhere and is not in the record. Recurrence status "
               "is genuinely unknown (99), not 00."),
    ),
    Blueprint(
        pid="SYN0003", pattern="never disease-free — metastatic at presentation",
        site_text="pancreatic head", site_code="C250",
        histology_text="adenocarcinoma", histology_code="8140", behavior="3",
        dx_date="20210118",
        dx_date_why="EUS-FNA pathology 2021-01-18 is the first diagnosis of record.",
        tissue=True, recurrence_type="70", recurrence_date="",
        imaging=["Abd-Pelvis-CT-W-Contr", "Chest-CT-W-Contr"],
        notes=("Distant metastasis at diagnosis -> STORE [1880] code 70 (never disease-free). "
               "An agent that codes 00 here is wrong: there was never a disease-free interval."),
    ),
    Blueprint(
        pid="SYN0004", pattern="documented recurrence after a disease-free interval",
        site_text="left breast, upper outer quadrant", site_code="C504",
        histology_text="infiltrating ductal carcinoma", histology_code="8500", behavior="3",
        dx_date="20190305",
        dx_date_why="Lumpectomy pathology 2019-03-05.",
        tissue=True, recurrence_type="52", recurrence_date="20220714",
        imaging=["Chest-CT-W-Contr", "PET-CT-Skull-Base-Thigh"],
        notes=("Requires the agent to establish BOTH (a) a disease-free interval and (b) the later "
               "pulmonary recurrence. Distant recurrence in lung only -> 52."),
    ),
    Blueprint(
        pid="SYN0005", pattern="in situ disease — behavior 2",
        site_text="urinary bladder", site_code="C679",
        histology_text="urothelial carcinoma in situ", histology_code="8120", behavior="2",
        dx_date="20240220",
        dx_date_why="TURBT pathology 2024-02-20.",
        tissue=True, recurrence_type="00", recurrence_date="",
        imaging=["Abd-Pelvis-CT-W-Contr"],
        notes=("Behavior must be 2, not 3. Tests that the agent reads the behavior off the pathology "
               "wording rather than defaulting to invasive."),
    ),
    Blueprint(
        pid="SYN0006", pattern="patient declined biopsy — radiographic diagnosis only",
        site_text="right lung", site_code="C349",
        histology_text="", histology_code="", behavior="",
        dx_date="20230802",
        dx_date_why="CT chest 2023-08-02 spiculated mass; pulmonary note asserts probable NSCLC.",
        tissue=False, recurrence_type="99", recurrence_date="",
        imaging=["Chest-CT-WWO-Contr"],
        notes="Second evidence-gap probe with a different mechanism (patient refusal, documented in prose).",
    ),
    Blueprint(
        pid="SYN0007", pattern="focal invasion inside an in-situ lesion — behavior 3",
        site_text="right breast", site_code="C509",
        histology_text="intraductal carcinoma with focal invasion", histology_code="8500", behavior="3",
        dx_date="20200917",
        dx_date_why="Excisional biopsy pathology 2020-09-17.",
        tissue=True, recurrence_type="00", recurrence_date="",
        imaging=["Chest-CT-W-Contr"],
        notes=("STORE [523] boundary case: 'Code 3 if any malignant invasion is present, no matter how "
               "limited.' The prose says intraductal (which reads in situ) but reports focal invasion."),
    ),
    Blueprint(
        pid="SYN0008", pattern="conflicting notes — consult contradicts the imaging impression",
        site_text="stomach, antrum", site_code="C163",
        histology_text="adenocarcinoma", histology_code="8140", behavior="3",
        dx_date="20221103",
        dx_date_why="Endoscopic biopsy pathology 2022-11-03.",
        tissue=True, recurrence_type="99", recurrence_date="",
        imaging=["Abd-Pelvis-CT-W-Contr", "Chest-CT-W-Contr"],
        notes=("Oncology consult claims 'CT shows liver metastases' while the CT IMPRESSION explicitly "
               "says no hepatic lesions. Tests conflict handling and whether the agent cites the "
               "primary report over the downstream restatement."),
    ),

    # ---------------------------------------------------------------------------------
    # Observable-period controls. These four exist to pin down one rule: clipping changes
    # the SCOPE of a coverage claim, never the VERDICT inside that scope.
    #
    # 09 and 10 must be INDISTINGUISHABLE to the agent — the only difference is an event
    # that happened at an outside hospital and left no trace in this chart. If an
    # implementation gets either one "right" it is using information it cannot have.
    # 11 and 12 are the positive/negative pair for truncation: a truncated record is not
    # a gap, so 11 must still yield a valid negative, and 12 must still surface the
    # recurrence that is plainly visible before the truncation.
    # ---------------------------------------------------------------------------------
    Blueprint(
        pid="SYN0009", pattern="interior follow-up gap; genuinely no recurrence",
        site_text="left breast", site_code="C504",
        histology_text="infiltrating ductal carcinoma", histology_code="8500", behavior="3",
        dx_date="20180614", dx_date_why="Lumpectomy pathology 2018-06-14.",
        tissue=True, recurrence_type="99", recurrence_date="",
        imaging=["Chest-CT-W-Contr"], followup="interior", gap_years=(2, 4),
        notes=("Two-year hole in the middle of surveillance. Nothing recurred, but the chart "
               "cannot show that. Must be Unknown (99)."),
        expect={"recurrence_status": "EVIDENCE_INSUFFICIENT", "recurrence_type": "99",
                "reason": "interior_gap", "indistinguishable_from": "SYN0010"},
    ),
    Blueprint(
        pid="SYN0010", pattern="interior follow-up gap; recurrence happened elsewhere during it",
        site_text="left breast", site_code="C504",
        histology_text="infiltrating ductal carcinoma", histology_code="8500", behavior="3",
        dx_date="20180614", dx_date_why="Lumpectomy pathology 2018-06-14.",
        tissue=True, recurrence_type="99", recurrence_date="",
        imaging=["Chest-CT-W-Contr"], followup="interior", gap_years=(2, 4),
        notes=("Same shape as SYN0009, but the patient recurred during the hole and was treated "
               "at another hospital. NOTHING about that appears here. Also Unknown (99) — and an "
               "implementation that distinguishes it from SYN0009 is cheating."),
        expect={"recurrence_status": "EVIDENCE_INSUFFICIENT", "recurrence_type": "99",
                "reason": "interior_gap", "indistinguishable_from": "SYN0009"},
    ),
    Blueprint(
        pid="SYN0011", pattern="follow-up truncated; no recurrence before truncation",
        site_text="sigmoid colon", site_code="C187",
        histology_text="adenocarcinoma", histology_code="8140", behavior="3",
        dx_date="20190222", dx_date_why="Colonoscopic biopsy pathology 2019-02-22.",
        tissue=True, recurrence_type="00", recurrence_date="",
        imaging=["Abd-Pelvis-CT-W-Contr"], followup="truncated", truncate_after_years=2.0,
        notes=("ACCEPTANCE TEST for the clipping fix. Surveillance is clean, then the patient "
               "simply stops coming. Everything after last contact is out of scope, NOT a gap. "
               "Must yield a valid negative with through_date = last contact. A one-size "
               "'reject every empty window' implementation fails this."),
        expect={"recurrence_status": "FOUND", "recurrence_type": "00",
                "through_date": "last_contact", "finality": "Provisional",
                "finality_reason": "OBSERVATION_TRUNCATED"},
    ),
    Blueprint(
        pid="SYN0012", pattern="follow-up truncated; recurrence documented BEFORE truncation",
        site_text="sigmoid colon", site_code="C187",
        histology_text="adenocarcinoma", histology_code="8140", behavior="3",
        dx_date="20190222", dx_date_why="Colonoscopic biopsy pathology 2019-02-22.",
        tissue=True, recurrence_type="54", recurrence_date="20201109",
        imaging=["Abd-Pelvis-CT-W-Contr"], followup="truncated", truncate_after_years=2.0,
        notes=("NEGATIVE CONTROL for SYN0011. Same truncation, but a hepatic recurrence is "
               "plainly documented before it. Clipping must narrow the claim's SCOPE without "
               "touching the VERDICT inside that scope — so this must report the recurrence, "
               "not abstain and not say 00. Without this case, an implementation that waves "
               "through every truncated record would pass SYN0011."),
        expect={"recurrence_status": "FOUND", "recurrence_type": "54",
                "recurrence_date": "20201109", "through_date": "last_contact"},
    ),
]


# --------------------------------------------------------------------------------------
# Prose builders
# --------------------------------------------------------------------------------------
def _hdr(bp: Blueprint, name: str, sex: str, dob: str, doctype: str, d: date) -> str:
    return (
        f"MRN: {bp.pid}\n"
        f"Patient: {name}\n"
        f"DOB: {dob}   Sex: {sex}\n"
        f"Document Type: {doctype.replace('-', ' ')}\n"
        f"Service Date: {d.isoformat()}\n"
        f"{'-' * 64}\n"
    )


def imaging_note(bp, name, sex, dob, dtype, d, rng, *, mention_mets: bool) -> str:
    body = [
        "CLINICAL HISTORY:",
        f"  {rng.choice(['Eval of', 'Follow-up', 'Restaging for'])} {bp.site_text} lesion.",
        "",
        "TECHNIQUE:",
        f"  {dtype.replace('-', ' ')} performed per department protocol.",
        "",
        "FINDINGS:",
        f"  There is a mass involving the {bp.site_text}, measuring "
        f"{rng.randint(15, 62)} x {rng.randint(12, 55)} mm.",
        "  No acute osseous abnormality. Visualized bowel gas pattern unremarkable.",
    ]
    if mention_mets:
        body.append("  Multiple hypodense hepatic lesions compatible with metastatic disease.")
    else:
        body.append("  No hepatic lesions identified. No definite hepatic metastatic disease.")
    body += [
        "",
        "IMPRESSION:",
        f"  1. Mass in the {bp.site_text}, as above. Neoplasm is favored.",
        "  2. " + ("Hepatic metastases." if mention_mets else "No evidence of hepatic metastasis."),
    ]
    return _hdr(bp, name, sex, dob, dtype, d) + "\n".join(body) + "\n"


def pathology_note(bp, name, sex, dob, dtype, d) -> str:
    body = [
        "SPECIMEN:",
        f"  A. {bp.site_text} — biopsy",
        "",
        "GROSS DESCRIPTION:",
        "  Received in formalin, multiple tan-pink soft tissue fragments aggregating 1.2 cm.",
        "",
        "MICROSCOPIC DESCRIPTION:",
        f"  Sections demonstrate {bp.histology_text}.",
    ]
    if bp.behavior == "3" and "focal invasion" in bp.histology_text:
        body.append("  Focal stromal invasion is identified, limited in extent but unequivocal.")
    elif bp.behavior == "2":
        body.append("  No stromal invasion is identified. The process is confined to the epithelium.")
    body += [
        "",
        "FINAL DIAGNOSIS:",
        f"  A. {bp.site_text.upper()}, BIOPSY — {bp.histology_text.upper()}.",
    ]
    if bp.behavior == "2":
        body.append("     No invasive carcinoma identified.")
    body += ["", "Electronically signed by Pathology.", ""]
    return _hdr(bp, name, sex, dob, dtype, d) + "\n".join(body) + "\n"


def cytology_note(bp, name, sex, dob, d) -> str:
    body = [
        "SPECIMEN:",
        f"  Fine needle aspirate, {bp.site_text}",
        "",
        "INTERPRETATION:",
        f"  Atypical cells present, suspicious for {bp.histology_text or 'malignancy'}.",
        "  Correlation with tissue biopsy recommended.",
        "",
        "IMPRESSION:",
        f"  SUSPICIOUS FOR {(bp.histology_text or 'MALIGNANCY').upper()}.",
        "",
    ]
    return _hdr(bp, name, sex, dob, "Surgical-Pathology-Document", d) + "\n".join(body) + "\n"


def progress_note(bp, name, sex, dob, dtype, d, rng, *, kind: str) -> str:
    """kind: initial | interval | disease_free | recurrence | gap_outside | gap_declined | conflict"""
    lines = ["SUBJECTIVE:"]
    if kind == "initial":
        lines += [
            f"  Pt seen today to discuss recent imaging of the {bp.site_text}.",
            "  Reports fatigue, appetite down. No fevers.",
            "",
            "ASSESSMENT AND PLAN:",
            f"  1. Mass, {bp.site_text}. Clinically this represents malignancy;",
            "     proceeding to tissue sampling for confirmation.",
        ]
    elif kind == "disease_free":
        lines += [
            "  Pt doing well. Denies new sx. Completed adjuvant therapy.",
            "",
            "ASSESSMENT AND PLAN:",
            "  1. No evidence of disease on todays exam and recent imaging.",
            "     Pt is disease free. Continue surveillance q6mo.",
        ]
    elif kind == "recurrence":
        lines += [
            "  Pt reports new cough x 6 wks. Restaging obtained.",
            "",
            "ASSESSMENT AND PLAN:",
            "  1. Recurrent disease, now with pulmonary metastases. Prior disease-free interval",
            "     since completion of therapy. Recurrence confined to lung.",
            "  2. Will initiate systemic therapy.",
        ]
    elif kind == "gap_outside":
        lines += [
            "  Pt transferred care to us. Records from outside hospital are incomplete.",
            "",
            "ASSESSMENT AND PLAN:",
            f"  1. Known {bp.site_text} carcinoma — biopsy was performed at the outside facility;",
            "     the pathology report is NOT available in our system. Requested by ROI.",
        ]
    elif kind == "gap_declined":
        lines += [
            "  Discussed biopsy at length. Pt declines bronchoscopic sampling.",
            "",
            "ASSESSMENT AND PLAN:",
            f"  1. {bp.site_text.capitalize()} mass, radiographically suspicious for NSCLC.",
            "     Pt DECLINED biopsy. No tissue diagnosis available. Dx is clinical/radiographic.",
        ]
    elif kind == "conflict":
        lines += [
            "  Interval hx reviewed.",
            "",
            "ASSESSMENT AND PLAN:",
            f"  1. {bp.site_text.capitalize()} adenocarcinoma. CT shows liver mets — stage IV.",
            "     (Note: attending restatement; see radiology IMPRESSION of record.)",
        ]
    else:  # interval
        lines += [
            "  Interval visit. Tolerating therapy. Labs reviewed.",
            "",
            "ASSESSMENT AND PLAN:",
            "  1. Continue current regimen. RTC 4 wks.",
        ]
    lines += [
        "",
        "MEDS: " + ", ".join(rng.sample(
            ["ondansetron", "dexamethasone", "pantoprazole", "lisinopril", "metformin", "atorvastatin"], 3)),
        "",
    ]
    return _hdr(bp, name, sex, dob, dtype, d) + "\n".join(lines) + "\n"


def rx_note(bp, name, sex, dob, d, rng) -> str:
    meds = rng.sample(
        ["METFORMIN HCL 500MG TAB", "ATORVASTATIN 20MG TAB", "LISINOPRIL 10MG TAB",
         "ONDANSETRON 8MG TAB", "PANTOPRAZOLE 40MG TAB", "OXYCODONE 5MG TAB"], 3)
    body = ["FILLED PRESCRIPTIONS (external claims feed):", ""]
    for m in meds:
        body.append(f"  {d.isoformat()}   {m:32s} qty {rng.choice([30, 60, 90])}")
    body.append("")
    return _hdr(bp, name, sex, dob, RX_TYPE, d) + "\n".join(body) + "\n"


def ancillary_note(bp, name, sex, dob, dtype, d, rng) -> str:
    if dtype == "EKG":
        body = ["INTERPRETATION:", f"  {rng.choice(['Normal sinus rhythm', 'Sinus tachycardia'])}, "
                f"rate {rng.randint(62, 104)}. No acute ST-T changes.", ""]
    elif dtype == "Endoscopy":
        body = ["FINDINGS:", f"  Mucosal lesion noted in the {bp.site_text}. Biopsies obtained.", ""]
    else:
        body = ["FINDINGS:", "  Procedure completed without immediate complication.", ""]
    return _hdr(bp, name, sex, dob, dtype, d) + "\n".join(body) + "\n"


# --------------------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------------------
def _d(s: str) -> date:
    return date(int(s[:4]), int(s[4:6]), int(s[6:8]))


def build_patient(bp: Blueprint, out_root: Path) -> dict:
    rng = random.Random(int(bp.pid[3:]) * 7919)
    name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
    sex = rng.choice(["M", "F"])
    dx = _d(bp.dx_date)
    dob = date(dx.year - rng.randint(48, 82), rng.randint(1, 12), rng.randint(1, 28)).isoformat()

    pdir = out_root / bp.pid
    pdir.mkdir(parents=True, exist_ok=True)
    for old in pdir.glob("*.txt"):
        old.unlink()

    written: list[tuple[str, date]] = []

    def emit(doctype: str, d: date, text: str) -> None:
        base = f"{doctype}_{d.isoformat()}"
        path = pdir / f"{base}.txt"
        n = 2
        while path.exists():                      # produce the real "__2" collision suffix
            path = pdir / f"{base}__{n}.txt"
            n += 1
        path.write_text(text, encoding="utf-8")
        written.append((path.name, d))

    # --- longitudinal background -------------------------------------------------
    # A decade of routine care. This is the bulk of a real chart and the reason
    # exhaustive review is infeasible: hundreds of documents, almost none of which can
    # establish anything about the tumour. These populate the cannot_establish stratum.
    # Type mix follows the distribution actually observed in the reference corpus
    # (RxHub and routine progress notes dominate; imaging and pathology are rare).
    start = dx - timedelta(days=365 * BACKGROUND_YEARS)
    day = start
    while day < dx - timedelta(days=10):
        if rng.random() < 0.80:
            emit(RX_TYPE, day, rx_note(bp, name, sex, dob, day, rng))
        if rng.random() < 0.55:
            t = rng.choice(PROGRESS_SPECIALTIES)
            d2 = day + timedelta(days=rng.randint(0, 20))
            emit(t, d2, progress_note(bp, name, sex, dob, t, d2, rng, kind="interval"))
        if rng.random() < 0.35:
            t = rng.choice(OTHER_TYPES)
            d2 = day + timedelta(days=rng.randint(0, 25))
            emit(t, d2, ancillary_note(bp, name, sex, dob, t, d2, rng))
        if rng.random() < 0.20:      # incidental imaging, NOT about the tumour
            t = rng.choice([x for x in IMAGING_TYPES if x not in bp.imaging])
            d2 = day + timedelta(days=rng.randint(0, 25))
            emit(t, d2, ancillary_note(bp, name, sex, dob, t, d2, rng))
        day += timedelta(days=rng.randint(18, 30))

    # --- index imaging, a few days before diagnosis ---
    for i, itype in enumerate(bp.imaging):
        idate = dx - timedelta(days=6 - i)
        emit(itype, idate, imaging_note(bp, name, sex, dob, itype, idate, rng, mention_mets=(bp.pattern.startswith("never disease-free") and i == 0)))

    # --- the diagnostic event itself ---
    if bp.pid == "SYN0001":
        emit("Surgical-Pathology-Document", dx, cytology_note(bp, name, sex, dob, dx))
        onc = "Onc-Med-MD-OP-Progress-Note"
        emit(onc, dx, progress_note(bp, name, sex, dob, onc, dx, rng, kind="initial"))
        later = dx + timedelta(days=15)
        emit("Surgical-Pathology-Report", later, pathology_note(bp, name, sex, dob, "Surgical-Pathology-Report", later))
    elif bp.tissue:
        emit(rng.choice(PATH_TYPES), dx, pathology_note(bp, name, sex, dob, "Surgical-Pathology-Report", dx))
        onc = "Onc-Med-MD-OP-Progress-Note"
        emit(onc, dx + timedelta(days=2), progress_note(bp, name, sex, dob, onc, dx + timedelta(days=2), rng, kind="initial"))
    else:
        kind = "gap_outside" if "outside" in bp.pattern else "gap_declined"
        t = "GI-Gen-MD-OP-Progress-Note" if "outside" in bp.pattern else "Pulm-MD-OP-Progress-Note"
        emit(t, dx, progress_note(bp, name, sex, dob, t, dx, rng, kind=kind))

    if bp.pid == "SYN0008":
        onc = "Onc-Med-MD-OP-Progress-Note"
        cd = dx + timedelta(days=5)
        emit(onc, cd, progress_note(bp, name, sex, dob, onc, cd, rng, kind="conflict"))

    # --- follow-up arc, encoding the recurrence ground truth ---
    if bp.followup in ("interior", "truncated"):
        # Observable-period controls. Surveillance is emitted on a schedule and then either
        # holed out in the middle (interior) or stopped dead (truncated).
        onc = "Onc-Med-MD-OP-Progress-Note"
        g0, g1 = bp.gap_years
        horizon = 5.0 if bp.followup == "interior" else bp.truncate_after_years
        months = 0
        while months / 12.0 < horizon:
            months += 6
            yrs = months / 12.0
            if bp.followup == "interior" and g0 <= yrs < g1:
                continue                                    # the hole
            fd = dx + timedelta(days=int(30.44 * months))
            emit(onc, fd, progress_note(bp, name, sex, dob, onc, fd, rng, kind="disease_free"))
            if months % 12 == 0:                            # annual surveillance imaging
                it = bp.imaging[0] if bp.imaging else "Chest-CT-W-Contr"
                emit(it, fd + timedelta(days=4),
                     imaging_note(bp, name, sex, dob, it, fd + timedelta(days=4), rng, mention_mets=False))
        # a recurrence that IS visible in this chart (SYN0012)
        if bp.recurrence_date:
            rd = _d(bp.recurrence_date)
            emit("Abd-Pelvis-CT-W-Contr", rd - timedelta(days=3),
                 imaging_note(bp, name, sex, dob, "Abd-Pelvis-CT-W-Contr",
                              rd - timedelta(days=3), rng, mention_mets=True))
            emit(onc, rd, progress_note(bp, name, sex, dob, onc, rd, rng, kind="recurrence"))
        # SYN0010's recurrence happened during the hole, at another hospital: emit NOTHING.
    elif bp.recurrence_type == "00":
        for k in (1, 2, 3):
            fd = dx + timedelta(days=180 * k)
            t = "Onc-Med-MD-OP-Progress-Note"
            emit(t, fd, progress_note(bp, name, sex, dob, t, fd, rng, kind="disease_free"))
    elif bp.recurrence_type == "52":
        for k in (1, 2):
            fd = dx + timedelta(days=200 * k)
            t = "Onc-Med-MD-OP-Progress-Note"
            emit(t, fd, progress_note(bp, name, sex, dob, t, fd, rng, kind="disease_free"))
        rd = _d(bp.recurrence_date)
        emit("Chest-CT-W-Contr", rd - timedelta(days=3),
             imaging_note(bp, name, sex, dob, "Chest-CT-W-Contr", rd - timedelta(days=3), rng, mention_mets=False))
        t = "Onc-Med-MD-OP-Progress-Note"
        emit(t, rd, progress_note(bp, name, sex, dob, t, rd, rng, kind="recurrence"))
    elif bp.recurrence_type == "70":
        for k in (1, 2):
            fd = dx + timedelta(days=90 * k)
            t = "Onc-Med-MD-OP-Progress-Note"
            emit(t, fd, progress_note(bp, name, sex, dob, t, fd, rng, kind="interval"))
    else:  # 99 — sparse, ambiguous follow-up
        fd = dx + timedelta(days=240)
        t = "GI-Gen-MD-OP-Progress-Note"
        emit(t, fd, progress_note(bp, name, sex, dob, t, fd, rng, kind="interval"))

    # post-diagnosis pharmacy noise, honouring the same follow-up shape so that a gap is a
    # gap in EVERY document type, not just the oncology notes
    last_day = max(d for _, d in written)
    pd_ = dx + timedelta(days=25)
    while pd_ <= last_day:
        yrs = (pd_ - dx).days / 365.25
        in_hole = bp.followup == "interior" and bp.gap_years[0] <= yrs < bp.gap_years[1]
        past_end = bp.followup == "truncated" and yrs > bp.truncate_after_years
        if not (in_hole or past_end):
            emit(RX_TYPE, pd_, rx_note(bp, name, sex, dob, pd_, rng))
        pd_ += timedelta(days=rng.randint(26, 40))

    obs_end = max(d for _, d in written)
    gt = {
        "patient_id": bp.pid,
        "evidence_pattern": bp.pattern,
        "designer_notes": bp.notes,
        "n_documents": len(written),
        "date_range": [min(d for _, d in written).isoformat(), max(d for _, d in written).isoformat()],
        "ground_truth": {
            "STORE.390.date_of_initial_diagnosis": {
                "value": bp.dx_date, "status": "FOUND", "why": bp.dx_date_why},
            "STORE.400_522_523.site_histology_behavior": {
                "primary_site": bp.site_code,
                "histology": bp.histology_code or None,
                "behavior": bp.behavior or None,
                "status": "FOUND" if bp.tissue else "EVIDENCE_INSUFFICIENT",
                "why": ("Tissue diagnosis present in the chart." if bp.tissue else
                        "No pathology in the record; histology must NOT be inferred from imaging."),
            },
            "STORE.1860_1880.first_recurrence": {
                "type": bp.recurrence_type,
                "date": bp.recurrence_date or None,
                "status": "FOUND" if bp.recurrence_type != "99" else "EVIDENCE_INSUFFICIENT",
                # observable period: the right edge is the last document of any kind.
                # A coverage claim is only meaningful relative to this.
                "observable_period": {"start": min(d for _, d in written).isoformat(),
                                      "end": obs_end.isoformat()},
                "expected_through_date": obs_end.isoformat(),
                "followup_shape": bp.followup,
            },
            "STORE.610.class_of_case": {"status": "SPEC_INSUFFICIENT", "why": "Not derivable from notes."},
            "STORE.580.date_of_first_contact": {"status": "SPEC_INSUFFICIENT", "why": "Not derivable from notes."},
            "STORE.1760_1750.vital_status": {"status": "SPEC_INSUFFICIENT", "why": "Not derivable from notes."},
        },
        # Machine-assertable expectations. Tests read this; nobody eyeballs it.
        "expect": bp.expect,
    }
    (pdir / "_ground_truth.json").write_text(json.dumps(gt, indent=2) + "\n", encoding="utf-8")
    return gt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="corpus/patients", type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    index = [build_patient(bp, a.out) for bp in BLUEPRINTS]
    (a.out.parent / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    total = sum(g["n_documents"] for g in index)
    print(f"generated {len(index)} patients, {total} documents -> {a.out}")
    for g in index:
        print(f"  {g['patient_id']}  {g['n_documents']:3d} docs  "
              f"{g['date_range'][0]}..{g['date_range'][1]}  {g['evidence_pattern']}")


if __name__ == "__main__":
    main()
