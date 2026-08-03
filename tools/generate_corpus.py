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
import hashlib
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

#: WHY A CANDIDATE LOSES, as a closed vocabulary. Prose reasons cannot be compared across
#: charts, and a reasoner may be reliable at one kind and unreliable at another — a note's own
#: service date is easy, a conflict rule is not — which one number over both would hide.
REJECTION_CODES = (
    "DOCUMENT_SERVICE_DATE",
    "NONQUALIFYING_RADIOLOGY",
    "SUPERSEDED_BY_EARLIER",
    "TREATMENT_OR_FOLLOWUP_DATE",
    "NOT_THE_TARGET_ENTITY",
    "AMBIGUOUS_CYTOLOGY_UNSUPPORTED",
    "OUT_OF_SCOPE_DATE",
)


# --------------------------------------------------------------------------------------
# Patient blueprints — each encodes what the ground truth should be and why
#
# THREE POPULATIONS, and `informed_module_design` is what separates them.
#
#   SYN0001-0012  built before any method card existed, to vary the CLINICAL situation —
#                 in situ, metastatic at presentation, the recurrence trichotomy. No card was
#                 written from them. But they appear in every ladder run the cards were then
#                 revised against, and nobody has traced which card moved because of which
#                 chart. They therefore carry the DEFAULT, `informed_module_design=True`,
#                 with `designed_from` left empty — which reads as "nobody recorded where this
#                 came from, so it is treated as informed". Marking twelve charts clean on a
#                 belief nobody checked is the failure this flag exists to prevent.
#   SYNX / SYNK   informed, and the provenance is known: each names the run behaviour that
#                 produced its trap.
#   SYNY01-Y06    held out. Every trap derived from a contract clause no other chart
#                 exercises, named in `designed_from` and checkable against the contract.
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
    #   terminal   NOTHING after the index date. For SYNY01, where the index date is the
    #              patient's death: the generic arcs below emitted surveillance and pharmacy
    #              claims for eight months afterwards, which is a corpus bug that reads from
    #              outside exactly like a chart nobody checked.
    followup: str = "normal"
    gap_years: tuple[int, int] = (2, 4)     # years after dx that the interior hole spans
    truncate_after_years: float = 2.0       # for followup == "truncated"
    expect: dict = field(default_factory=dict)   # machine-assertable expectations

    # ---- WHAT THE ANSWER IS, when it is not a plain date ----------------------------
    #: The recorded STORE.390 status. `FOUND` for every chart that has an answer; a chart
    #: whose CORRECT answer is an abstention has to be able to say so, and until SYNY05 none
    #: could — so `CORPUS_INSUFFICIENT` was an outcome the contract declared, the tool offered,
    #: and no chart in the corpus could ever score as right.
    #: How many years of routine care precede the index date. `BACKGROUND_YEARS` for every
    #: chart but one: SYNY05's whole premise is that this institution's record BEGINS after the
    #: diagnosis, and a decade of prior notes would make that false — the diagnosis would fall
    #: inside the observed window and the correct answer would be EVIDENCE_INSUFFICIENT rather
    #: than CORPUS_INSUFFICIENT. The chart is short as a result, and the shortness IS the
    #: finding: a record this thin cannot establish a date and saying so is the right answer.
    background_years: int = BACKGROUND_YEARS
    #: Which ancillary types may appear in the background noise. `OTHER_TYPES` for every chart
    #: but two, and the exception is a defect found by writing the tests rather than by reading
    #: the generator: `ancillary_note`'s Endoscopy branch writes "Mucosal lesion noted in the
    #: {site_text}. Biopsies obtained." — background noise that names the TUMOUR SITE and
    #: claims tissue. On SYNY01 that put fourteen notes saying biopsies were obtained into a
    #: chart whose death summary says none were, and whose whole premise is that no tissue
    #: exists. A chart that contradicts its own ground truth is not a hard case; it is a wrong
    #: one.
    #:
    #: NARROWED PER CHART rather than fixed in `ancillary_note`, and that is a deliberate
    #: refusal. Rewriting that branch would change the bytes of all twenty-one existing charts,
    #: and this module promises byte-identical regeneration because every recorded pilot number
    #: was measured on those bytes. The latent version of the same problem on SYN0002 (tissue
    #: at an outside hospital, and Endoscopy notes here claiming biopsies) is recorded and NOT
    #: fixed for that reason.
    background_types: tuple[str, ...] = ()
    dx_status: str = "FOUND"
    #: The imputation flags the answer carries. Empty means all three false. A chart that
    #: forces an approximated year or a season-derived month is the only way to find out
    #: whether the three flags are filled in or left at their default.
    dx_flags: dict = field(default_factory=dict)

    # ---- WHETHER THIS CHART MAY BE SCORED AS A HEADLINE NUMBER ----------------------
    #: True when the chart's design was informed by watching a module perform.
    #:
    #: SYNX01-06 were built by watching runs fail, and the search cards were then written from
    #: the same failures — SYNX06's own designer note says it tests "precisely the shorter-stem
    #: move the controller-reactive card advises". Scoring those cards on those charts is
    #: scoring them on their own development set, and until 2026-08-03 nothing in the tree
    #: recorded that. `docs/MODULE_LADDER_EXPERIMENT.md:138` named the flag and no code had it.
    #:
    #: DEFAULTS TO TRUE, and the direction is the point: the failure mode being guarded against
    #: is a contaminated chart silently counted as clean, so a chart that does not claim to be
    #: held out is treated as informed. Claiming it requires naming where the design came from.
    informed_module_design: bool = True
    # ---- GOLD FOR THE CANDIDATE MECHANISM, not only for the answer ------------------
    #: EVERY VALUE A CAREFUL READER WOULD PUT ON THE TABLE, including the ones that lose. A
    #: gold ANSWER cannot tell three failures apart — never saw the candidate, saw it and
    #: wrongly ruled it out, saw it and resolved it correctly — and those have three different
    #: owners. Empty means nobody has stated it, and the analyser EXCLUDES such a chart from
    #: the candidate metrics rather than scoring it as a miss.
    gold_candidates: tuple[str, ...] = ()
    #: value -> why it loses, as a CODE from `REJECTION_CODES`. Prose cannot be compared across
    #: charts, and which KIND of rejection a reasoner gets wrong is the question worth asking.
    gold_rejections: dict = field(default_factory=dict)
    #: What the answerability axis should say, kept apart from the values by construction.
    gold_answerability: str = "VALUE_AVAILABLE"
    #: `clear` / `competing` / `no_answer`. The `clear` stratum is where false competition is
    #: measured; without it, candidate recall has no counterweight and rewards declaring
    #: alternatives everywhere.
    candidate_stratum: str = ""

    #: Where the trap came from, in words a reader can check against the artefact named.
    #: `contract_clause: <clause>` for a held-out chart — the clause must be one the contract
    #: states and no other chart exercises. Required whenever `informed_module_design` is False.
    designed_from: str = ""

    # ---- ADVERSARIAL LAYOUT ---------------------------------------------------------
    #: Which retrieval trap this chart is built around; "" is the ordinary layout above.
    #:
    #: The first twelve patients vary the CLINICAL situation — in situ, metastatic at
    #: presentation, recurrence shapes — and in all of them the establishing document is where
    #: anyone would look. That is why a run with no retrieval guidance at all scored 11 of 12 on
    #: `STORE.390` in the 2026-07-31 pilot, leaving ONE case of headroom in the whole cohort and
    #: no way to tell four traversal arms apart.
    #:
    #: These vary WHERE THE ANSWER IS instead. Every trap below is built on a rule the spec
    #: already states, so a chart tests whether a run can find and follow that rule, never
    #: whether it happens to know some clinical fact. Each also has a specific WRONG date that a
    #: naive pass produces — so a failure is a measurable MISMATCH rather than an abstention,
    #: and the two are different failures with different owners.
    trap: str = ""

    #: WHEN THE RECORDED ANSWER IS NOT THE CHART'S ANSWER.
    #:
    #: A registry value is what a human abstractor wrote down, and abstractors read outside
    #: records, mistype, and apply the wrong rule. README section 2.5 already refuses to call
    #: such a value gold: it is staged as a REGISTRY_REFERENCE and only a human adjudication of
    #: field-level chart derivability turns it into anything stronger.
    #:
    #: On these charts `_ground_truth.json` carries the REGISTRY value — including when it is
    #: wrong — because that is what a deployment actually has. An agent that reads the chart
    #: correctly therefore scores MISMATCH, and that is the point: what is being tested is not
    #: the agent but whether the EVALUATION can tell "the agent erred" from "the key did".
    #:
    #: `dispute` names which of those it is. Three kinds, and conflating them is the failure:
    #:   OUTSIDE_EVIDENCE  the key is right and the chart cannot show it. Abstaining is correct;
    #:                     scoring it as a miss teaches the agent to guess from outside knowledge
    #:                     it does not have.
    #:   KEY_ERROR         the key is wrong. The agent is right and is being marked down.
    #:   CHART_AMBIGUOUS   two readings are both supportable. Neither side erred, and any
    #:                     verdict that names a winner is manufacturing certainty.
    dispute: dict = field(default_factory=dict)

    #: The date the chart is LAID OUT around: index imaging, background years, follow-up arc.
    #: Defaults to `dx_date`, which is the answer. They come apart exactly when the answer is
    #: earlier than the workup that found it — `retrospective` is the case — and conflating
    #: them would put a decade of background notes after the event they precede.
    index_date: str = ""


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
        # `clear` FOR THIS TARGET. SYN0002's evidence gap is about HISTOLOGY — the biopsy was
        # done elsewhere — and STORE.390's date is plainly in the chart. A stratum is per
        # target, not per patient, and filing it under no_answer would have put a chart with a
        # findable date into the population that measures abstention.
        candidate_stratum="clear",
        gold_candidates=('20220906',),
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
        candidate_stratum="clear",
        gold_candidates=('20210118',),
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
        candidate_stratum="clear",
        gold_candidates=('20240220',),
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
        candidate_stratum="clear",
        gold_candidates=('20200917',),
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
        dx_date="20190222",
        candidate_stratum="clear",
        gold_candidates=('20190222',), dx_date_why="Colonoscopic biopsy pathology 2019-02-22.",
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

    # ==================================================================================
    # RETRIEVAL-ADVERSARIAL CHARTS (SYNX01-X06)
    #
    # The twelve above vary the clinical situation and put the establishing document where
    # anyone would look. On `STORE.390` that left one case of headroom in the whole cohort and
    # four traversal arms scoring 10, 11, 12 and 12 of 12 — a spread too small to rank.
    #
    # These vary WHERE THE ANSWER IS. Each is built on a rule the spec already states, so what
    # is tested is finding and following that rule, never knowing a clinical fact. Each puts a
    # plausible WRONG date where a naive pass will read it, so a failure is a MISMATCH the
    # scorer counts rather than an abstention.
    #
    # `expect.naive_answer` is the date a run gets by taking the obvious document. A chart on
    # which some arm returns exactly that is a chart that trap is working on.
    # ==================================================================================
    Blueprint(
        pid="SYNX01", pattern="retrospective diagnosis — the answer predates the workup by two years",
        trap="retrospective",
        site_text="right upper lobe of lung", site_code="C341",
        histology_text="adenocarcinoma", histology_code="8140", behavior="3",
        dx_date="20190312",
        candidate_stratum="competing",
        gold_candidates=('20190312', '20210608'),
        gold_rejections={'20210608': 'SUPERSEDED_BY_EARLIER'}, index_date="20210608",
        dx_date_why=("An oncology note of 2021-06-15 states that the 8 mm nodule on the "
                     "2019-03-12 CT is this same tumour in retrospect. STORE.390 decision_rule: "
                     "'If a physician states that in retrospect the patient had cancer at an "
                     "earlier date, use the earlier date.'"),
        tissue=True, recurrence_type="00", recurrence_date="",
        imaging=["Chest-CT-W-Contr"],
        notes=("The biopsy date 2021-06-08 is where every ordinary path leads and it is WRONG. "
               "Reaching 2019-03-12 needs two hops: find the retrospective sentence in a "
               "post-diagnosis oncology note, then go back for the CT it names. A sweep that "
               "stops at the pathology never sees the sentence; a chase that never opens the "
               "follow-up notes never starts."),
        expect={"dx_date": "20190312", "naive_answer": "20210608",
                "requires": "two hops: retrospective remark -> the imaging it names"},
        # INFORMED BY A RUN, and the flag defaults to True so this is the recorded
        # state either way. `designed_from` is set because the provenance is known
        # and specific, and a reader deciding whether to trust a number over this
        # chart needs the sentence, not the boolean.
        designed_from="observed_failure: naive passes took the workup date; the retrospective clause was the fix",
    ),
    Blueprint(
        pid="SYNX02", pattern="first-course treatment precedes any documented diagnosis",
        trap="treatment_first",
        site_text="left lower lobe of lung", site_code="C343",
        histology_text="adenocarcinoma", histology_code="8140", behavior="3",
        dx_date="20200510", index_date="20200620",
        dx_date_why=("Carboplatin/pemetrexed cycle 1 was administered 2020-05-10, six weeks "
                     "before the first document that establishes a diagnosis. STORE.390 "
                     "decision_rule: 'If first-course treatment began before a diagnosis was "
                     "documented, use the treatment start date.'"),
        tissue=True, recurrence_type="00", recurrence_date="",
        imaging=["Chest-CT-W-Contr"],
        notes=("The pathology of 2020-06-20 is the obvious answer and it is WRONG. The infusion "
               "record carries no narrative impression — it is an administration, under "
               "Procedure-Note — so a run that searches diagnostic vocabulary cannot reach it "
               "and must instead have swept the document inventory by type."),
        expect={"dx_date": "20200510", "naive_answer": "20200620",
                "requires": "sweeping a type that states no diagnosis"},
        # INFORMED BY A RUN, and the flag defaults to True so this is the recorded
        # state either way. `designed_from` is set because the provenance is known
        # and specific, and a reader deciding whether to trust a number over this
        # chart needs the sentence, not the boolean.
        designed_from="observed_failure: runs abstained where treatment preceded any documented diagnosis",
    ),
    Blueprint(
        pid="SYNX03", pattern="ambiguous cytology with NO clinical impression — the biopsy dates it",
        trap="cytology_no_impression",
        site_text="right lower lobe of lung", site_code="C342",
        histology_text="squamous cell carcinoma", histology_code="8070", behavior="3",
        dx_date="20220309",
        candidate_stratum="competing",
        gold_candidates=('20220309', '20220214'),
        gold_rejections={'20220214': 'AMBIGUOUS_CYTOLOGY_UNSUPPORTED'}, index_date="20220309",
        dx_date_why=("Cytology of 2022-02-14 is ambiguous ('suspicious for') and NO physician's "
                     "clinical impression of cancer accompanies it, so STORE.390's second "
                     "conflict_rule applies and the biopsy date governs."),
        tissue=True, recurrence_type="00", recurrence_date="",
        imaging=["Chest-CT-W-Contr"],
        notes=("THE MIRROR OF SYN0001, and the branch the corpus has never tested. SYN0001 has "
               "the same ambiguous cytology WITH a same-day clinical impression, so its answer "
               "is the cytology date. Here the impression is absent and the answer is the "
               "biopsy date. A run that has generalised 'ambiguous cytology dates the case' "
               "answers 2022-02-14 and is exactly wrong — the two conflict_rules differ by one "
               "document, and only reading both charts distinguishes them."),
        expect={"dx_date": "20220309", "naive_answer": "20220214",
                "requires": "noticing an ABSENT document, not finding a present one"},
        # INFORMED BY A RUN, and the flag defaults to True so this is the recorded
        # state either way. `designed_from` is set because the provenance is known
        # and specific, and a reader deciding whether to trust a number over this
        # chart needs the sentence, not the boolean.
        designed_from="observed_failure: the cytology-vs-biopsy ordering was applied without checking the impression",
    ),
    Blueprint(
        pid="SYNX04", pattern="pathology deferred to an addendum filed under another type",
        trap="deferred_addendum",
        site_text="right upper lobe of lung", site_code="C341",
        histology_text="adenocarcinoma", histology_code="8140", behavior="3",
        dx_date="20230824", index_date="20230803",
        dx_date_why=("The 2023-08-03 report DEFERRED its diagnosis pending stains and therefore "
                     "established nothing. The 2023-08-24 addendum establishes it."),
        tissue=True, recurrence_type="00", recurrence_date="",
        imaging=["Chest-CT-W-Contr"],
        notes=("P05's 8046 error given somewhere further to go. The deferred report is the only "
               "Surgical-Pathology-Report in the chart, so an arm that sweeps the pathology type "
               "finds a document that answers nothing; the addendum is filed under "
               "Surgical-Pathology-Document three weeks later. Following the words 'SEE "
               "ADDENDUM' is the only cheap route, which is what thread-chasing is for."),
        expect={"dx_date": "20230824", "naive_answer": "20230803",
                "requires": "following a pointer out of the document that raised it"},
        # INFORMED BY A RUN, and the flag defaults to True so this is the recorded
        # state either way. `designed_from` is set because the provenance is known
        # and specific, and a reader deciding whether to trust a number over this
        # chart needs the sentence, not the boolean.
        designed_from="observed_failure: addenda filed under another type were never opened",
    ),
    Blueprint(
        pid="SYNX05", pattern="the first diagnosis is a clinical impression buried in a diabetes note",
        trap="buried_late",
        site_text="pancreatic head", site_code="C250",
        histology_text="adenocarcinoma", histology_code="8140", behavior="3",
        dx_date="20181107",
        candidate_stratum="competing",
        gold_candidates=('20181107', '20190215'),
        gold_rejections={'20190215': 'SUPERSEDED_BY_EARLIER'}, index_date="20190215",
        dx_date_why=("An endocrinology follow-up of 2018-11-07 records the physician's "
                     "assessment that the pancreatic head lesion is malignant. STORE.390 takes "
                     "the FIRST date, clinically or histologically established, and a "
                     "physician's clinical impression of cancer counts as evidence."),
        tissue=True, recurrence_type="00", recurrence_date="",
        imaging=["Abd-Pelvis-CT-W-Contr"],
        notes=("The pathology of 2019-02-15 is three months later and is the obvious answer. "
               "The real one is one clause inside a routine diabetes visit — a specialty nobody "
               "searches for an oncology diagnosis, in a note type whose prior says it can "
               "establish nothing. Rewards sweeping the inventory by type over trusting a "
               "type prior."),
        expect={"dx_date": "20181107", "naive_answer": "20190215",
                "requires": "reading a type the prior calls unable to establish"},
        # INFORMED BY A RUN, and the flag defaults to True so this is the recorded
        # state either way. `designed_from` is set because the provenance is known
        # and specific, and a reader deciding whether to trust a number over this
        # chart needs the sentence, not the boolean.
        designed_from="observed_failure: a type prior demoted the note that held the answer",
    ),
    Blueprint(
        pid="SYNX06", pattern="the diagnosis exists only in dictation shorthand",
        trap="search_resistant",
        site_text="right upper lobe of lung", site_code="C341",
        histology_text="adenocarcinoma", histology_code="8140", behavior="3",
        dx_date="20210917", index_date="20210917",
        dx_date_why=("The pulmonary note of 2021-09-17 records 'Bx +ve. Path c/w adenoCA, RUL.' "
                     "That is a physician's statement of a tissue-confirmed diagnosis and it is "
                     "the first one in the chart."),
        tissue=False, recurrence_type="00", recurrence_date="",
        # NO index imaging. The standard `imaging_note` writes "neoplasm is favored", which
        # hands a keyword search the foothold this chart exists to withhold — caught by
        # `tests/test_adversarial_corpus.py`, not by reading the generator. The trap emitter
        # writes an incidental-nodule study instead: same clinical shape, no diagnostic word.
        imaging=[],
        notes=("No full word appears anywhere: not adenocarcinoma, not carcinoma, not "
               "malignant, not diagnosis. A run searching the contract's own vocabulary comes "
               "back empty and concludes the chart is silent. 'adeno' hits and "
               "'adenocarcinoma' does not, which is precisely the shorter-stem move the "
               "controller-reactive card advises — so this chart tests whether that advice is "
               "followed rather than merely rendered."),
        expect={"dx_date": "20210917", "naive_answer": "EVIDENCE_INSUFFICIENT",
                "requires": "widening to a stem after the contract's own words miss"},
        # INFORMED BY A RUN, and the flag defaults to True so this is the recorded
        # state either way. `designed_from` is set because the provenance is known
        # and specific, and a reader deciding whether to trust a number over this
        # chart needs the sentence, not the boolean.
        designed_from="observed_failure: the contract's own vocabulary missed; informed the stem-widening advice in controller-reactive",
    ),

    # ==================================================================================
    # CHARTS WHERE THE RECORDED ANSWER IS IN DISPUTE (SYNK01-K03)
    #
    # Everything above assumes the key is right and asks whether the agent can reach it. These
    # ask the other question. `ground_truth` here carries the REGISTRY value — wrong on K02 —
    # because that is what a deployment has, so a correct agent scores MISMATCH and the
    # evaluation has to be the thing that notices.
    #
    # The measurement is on the EVAL, and `key_dispute.kind` is its answer key. An evaluation
    # that returns KEY_ERROR on K01 has taught the agent to guess from outside knowledge; one
    # that returns OUTSIDE_EVIDENCE on K02 has laundered an abstractor's typo into truth; one
    # that names a winner on K03 has manufactured certainty. Three different harms, and a
    # single "the answer disagrees" verdict cannot tell them apart.
    # ==================================================================================
    Blueprint(
        pid="SYNK01", pattern="the key rests on an outside report that is not in this chart",
        trap="outside_evidence",
        site_text="left upper lobe of lung", site_code="C341",
        histology_text="adenocarcinoma", histology_code="8140", behavior="3",
        dx_date="20210315", index_date="20210802",
        dx_date_why=("REGISTRY VALUE. The abstractor had the outside facility's biopsy report of "
                     "2021-03-15. That report is not in this chart and no document here "
                     "establishes any diagnosis before 2021-08-02."),
        tissue=False, recurrence_type="00", recurrence_date="",
        imaging=["Chest-CT-W-Contr"],
        notes=("The key is CORRECT and unreachable. The only in-chart reference is a transfer "
               "note saying the biopsy was done elsewhere in March. An agent that answers "
               "EVIDENCE_INSUFFICIENT has read the chart correctly and will be scored as having "
               "missed — and an evaluation that calls that an agent error is teaching it to "
               "produce dates it cannot support."),
        dispute={
            "kind": "OUTSIDE_EVIDENCE",
            "registry_value": "20210315",
            "chart_supports": None,
            "why": ("The chart names an outside biopsy and its month but carries neither the "
                    "report nor a physician's diagnostic statement predating 2021-08-02."),
            "correct_eval_verdict": "KEY_NOT_DERIVABLE_FROM_CHART",
            "harm_if_missed": ("scoring the abstention as a miss trains the agent to guess on "
                               "exactly the subpopulation where records are incomplete"),
        },
        expect={"chart_answer": "EVIDENCE_INSUFFICIENT", "registry_value": "20210315"},
        # INFORMED BY A RUN, and the flag defaults to True so this is the recorded
        # state either way. `designed_from` is set because the provenance is known
        # and specific, and a reader deciding whether to trust a number over this
        # chart needs the sentence, not the boolean.
        designed_from="observed_failure: evaluation called an unverifiable key an agent error",
    ),
    Blueprint(
        pid="SYNK02", pattern="the key is a transcription error — no document carries that date",
        trap="key_typo",
        site_text="sigmoid colon", site_code="C187",
        histology_text="adenocarcinoma", histology_code="8140", behavior="3",
        dx_date="20200714", index_date="20200614",
        dx_date_why=("REGISTRY VALUE, AND IT IS WRONG. Every document in the chart places the "
                     "diagnosis at 2020-06-14; no document exists on 2020-07-14 and nothing "
                     "supports it. The month digit was mistyped."),
        tissue=True, recurrence_type="00", recurrence_date="",
        imaging=["Abd-Pelvis-CT-W-Contr"],
        notes=("The agent is right and the key is wrong. This is the cheapest dispute to detect "
               "and the one most worth detecting automatically: the key names a date on which "
               "the chart holds NO DOCUMENT AT ALL, which is decidable without reading a word "
               "of clinical text. An evaluation that cannot catch this one will not catch any."),
        dispute={
            "kind": "KEY_ERROR",
            "registry_value": "20200714",
            "chart_supports": "20200614",
            "why": ("Pathology, the oncology note and the index imaging all fall in June 2020. "
                    "The chart contains no document dated 2020-07-14."),
            "correct_eval_verdict": "KEY_CONTRADICTED_BY_CHART",
            "harm_if_missed": ("a correct run is recorded as a failure, and repeated, the "
                               "measured accuracy of a working system decays toward the "
                               "abstractor's error rate"),
        },
        expect={"chart_answer": "20200614", "registry_value": "20200714"},
        # INFORMED BY A RUN, and the flag defaults to True so this is the recorded
        # state either way. `designed_from` is set because the provenance is known
        # and specific, and a reader deciding whether to trust a number over this
        # chart needs the sentence, not the boolean.
        designed_from="observed_failure: evaluation could not report that the key itself was wrong",
    ),
    Blueprint(
        pid="SYNK03", pattern="two defensible readings — the chart itself does not settle it",
        trap="genuinely_ambiguous",
        site_text="right lower lobe of lung", site_code="C342",
        histology_text="adenocarcinoma", histology_code="8140", behavior="3",
        dx_date="20210510", index_date="20210510",
        dx_date_why=("REGISTRY VALUE, and defensible. The 2021-04-05 cytology reads POSITIVE "
                     "FOR MALIGNANT CELLS — not an ambiguous term — with no clinical impression "
                     "beside it. Whether an unambiguous cytology establishes the diagnosis "
                     "without tissue, or whether the 2021-05-10 biopsy does, is a reading the "
                     "spec's own wording supports both ways."),
        tissue=True, recurrence_type="00", recurrence_date="",
        imaging=["Chest-CT-W-Contr"],
        notes=("NEITHER SIDE IS WRONG, and that is what makes it a control. STORE.390 counts 'a "
               "pathology or cytology report whose interpretation establishes the diagnosis' "
               "and separately discounts 'cytology carrying only an ambiguous term'. POSITIVE "
               "FOR MALIGNANT CELLS is not ambiguous, so the first clause reaches it; but the "
               "conflict_rules are written around the ambiguous case and say nothing here. "
               "Without this chart an evaluation can score full marks by calling every "
               "disagreement a defect, which is the same failure as calling none of them one."),
        dispute={
            "kind": "CHART_AMBIGUOUS",
            "registry_value": "20210510",
            "chart_supports": ["20210405", "20210510"],
            "why": ("An unambiguous cytology and a confirmatory biopsy, and a spec whose "
                    "conflict rules only cover the ambiguous-cytology case."),
            "correct_eval_verdict": "HUMAN_ADJUDICATION_REQUIRED",
            "harm_if_missed": ("naming a winner here manufactures certainty, and a spec gap "
                               "reported as an agent error never reaches whoever owns the spec"),
        },
        expect={"chart_answer": "AMBIGUOUS", "registry_value": "20210510"},
        # INFORMED BY A RUN, and the flag defaults to True so this is the recorded
        # state either way. `designed_from` is set because the provenance is known
        # and specific, and a reader deciding whether to trust a number over this
        # chart needs the sentence, not the boolean.
        designed_from="observed_failure: evaluation named a winner where the contract supports both readings",
    ),

    # ==================================================================================
    # HELD OUT (SYNY01-Y06)
    #
    # WHY A SECOND ADVERSARIAL SET EXISTS. SYNX01-06 were designed by watching runs fail, and
    # the search cards were then written from the same failures — SYNX06's own designer note
    # says it tests "precisely the shorter-stem move the controller-reactive card advises".
    # Scoring those cards on those charts is scoring them on their own development set. That
    # was true from the day the cards were written and nothing in the tree recorded it;
    # `docs/MODULE_LADDER_EXPERIMENT.md:138` named the flag and no code had it.
    #
    # THE RULE THAT MAKES THESE HELD OUT, and it is checkable rather than promised: every trap
    # is derived from a CLAUSE THE CONTRACT STATES AND NO OTHER CHART EXERCISES, named in
    # `designed_from`. Not one came from a run result, a card's failure mode, or an arm's
    # score. A reader who doubts it can open the contract and the other twenty-one charts.
    #
    # The six unexercised clauses, and what each one costs a run that has not read it:
    #
    #   Y01  decision_rule[4]            death-certificate-only: the date of death IS the date
    #   Y02  decision_rule[5]            an unidentifiable year must be APPROXIMATED
    #   Y03  conflict_rules[4]           earliest wins AND every conflicting source is cited
    #   Y04  date_imputation             a season is the only thing the record offers
    #   Y05  abstention.CORPUS_INSUFFICIENT   the right answer is that the record starts too late
    #   Y06  evidence_rules.does_not_count[2]  suspicious imaging is not a diagnosis
    #
    # Y06 is in the set for a second reason. Y01, and SYNX01/02/05 before it, all reward
    # reaching for the EARLIEST date; a card written from those learns "earlier wins". Y06 runs
    # that backwards: the earliest candidate is inadmissible and the later one is the answer.
    # ==================================================================================
    Blueprint(
        pid="SYNY01", pattern="death certificate only — nothing ante-mortem names a cancer",
        trap="death_certificate_only",
        site_text="head of pancreas", site_code="C250",
        histology_text="adenocarcinoma", histology_code="8140", behavior="3",
        dx_date="20220419",
        candidate_stratum="competing",
        gold_candidates=('20220419',), index_date="20220419",
        dx_date_why=("Nothing before the death names a malignancy: the three preceding visits "
                     "record decline, weight loss and jaundice and no more. The Death-Summary "
                     "of 2022-04-19 states metastatic adenocarcinoma of the pancreas as the "
                     "cause of death, which is a physician's diagnostic statement, and "
                     "decision_rule[4] puts the date of a death-certificate-only case at the "
                     "date of death."),
        tissue=False, recurrence_type="99", recurrence_date="",
        imaging=[], followup="terminal",
        # No Endoscopy in the background: it would claim biopsies in a chart whose premise is
        # that no tissue was ever obtained. See `background_types`.
        background_types=("EKG", "Procedure-Note", "Fluoroscopy-Up-to-1-Hr"),
        notes=("No tissue was ever obtained and no oncology referral was completed, so there "
               "is nothing earlier to find. A run that reads the decline notes and stops "
               "returns EVIDENCE_INSUFFICIENT and is wrong by one document — one whose type "
               "appears nowhere else in the corpus, so no type prior can point at it."),
        expect={"dx_date": "20220419", "naive_answer": "EVIDENCE_INSUFFICIENT",
                "requires": "opening a document type that occurs once in the whole corpus"},
        informed_module_design=False,
        designed_from="contract_clause: decision_rule[4] (autopsy / death certificate only)",
    ),
    Blueprint(
        pid="SYNY02", pattern="the year cannot be read anywhere and must be approximated",
        trap="year_only_approximate",
        site_text="sigmoid colon", site_code="C187",
        histology_text="adenocarcinoma", histology_code="8140", behavior="3",
        dx_date="20159999",
        candidate_stratum="competing",
        gold_candidates=('20159999',), index_date="20190604",
        dx_date_why=("No document states a diagnosis date. The establishing-care note of "
                     "2019-06-04 says the cancer was treated 'roughly four years ago' at an "
                     "outside hospital, and the pharmacy feed carries an adjuvant capecitabine "
                     "course beginning 2015-08-17 — which cannot establish a diagnosis but does "
                     "bound the year. decision_rule[5]: the year is APPROXIMATED to 2015 and "
                     "month and day are then unknown, so 20159999 with year_imputed true."),
        tissue=False, recurrence_type="00", recurrence_date="",
        imaging=[],
        dx_status="FOUND",
        dx_flags={"year_imputed": True, "month_imputed": False, "day_imputed": False},
        # No Endoscopy in the background: this chart's ground truth says there is no
        # pathology in the record, and that branch claims biopsies. See `background_types`.
        background_types=("EKG", "Procedure-Note", "Fluoroscopy-Up-to-1-Hr"),
        notes=("THE CHART THE `20999999` DEFECT NEEDED. Two E4 runs put 99 in the YEAR slot "
               "because decision_rule[5] orders an approximation and no field could record "
               "that one had been made; the value space now can, and nothing tested it. Note "
               "which flags are true: the year was approximated, the month and day are simply "
               "not recorded — a distinction one boolean could not make, and the reason there "
               "are three. The hard anchor is in a claims feed, which is a source class a run "
               "reading only clinical note types never opens."),
        expect={"dx_date": "20159999", "naive_answer": "EVIDENCE_INSUFFICIENT",
                "requires": "approximating a year from a non-clinical source class"},
        informed_module_design=False,
        designed_from="contract_clause: decision_rule[5] (the year must be approximated)",
    ),
    Blueprint(
        pid="SYNY03", pattern="three admissible sources give three different dates",
        trap="three_sources_disagree",
        site_text="left breast", site_code="C504",
        histology_text="invasive ductal carcinoma", histology_code="8500", behavior="3",
        dx_date="20200302",
        candidate_stratum="competing",
        gold_candidates=('20200302', '20200326', '20200401', '20200410'),
        gold_rejections={'20200326': 'NONQUALIFYING_RADIOLOGY', '20200401': 'SUPERSEDED_BY_EARLIER', '20200410': 'SUPERSEDED_BY_EARLIER'}, index_date="20200401",
        dx_date_why=("Three sources, three dates. Cytology 2020-03-02 reads 'suspicious for "
                     "invasive ductal carcinoma' AND an oncology note of the same day records "
                     "a clinical impression of malignancy, so conflict_rules[1] makes the "
                     "cytology date diagnostic. Pathology 2020-04-01 confirms. A later "
                     "oncology note asserts the diagnosis was made 2020-04-10. Earliest that "
                     "satisfies the rules wins."),
        tissue=True, recurrence_type="00", recurrence_date="",
        imaging=["Mammography-Diagnostic-Bilat"],
        notes=("The arithmetic is not the test — SYN0001 already has a two-source version and "
               "runs get it right. conflict_rules[4] additionally demands that EVERY "
               "conflicting source be cited and the conflict resolved, which is the half of "
               "the proof obligation nothing has ever exercised. A run that answers 20200302 "
               "citing only the cytology has the right value and has not met the obligation, "
               "and those are two different results."),
        expect={"dx_date": "20200302", "naive_answer": "20200401",
                "requires": "citing all three conflicting sources, not only the winner"},
        informed_module_design=False,
        designed_from="contract_clause: conflict_rules[4] (cite every conflicting source)",
    ),
    Blueprint(
        pid="SYNY04", pattern="a season and a year are all the record offers",
        trap="seasonal_phrase",
        site_text="ascending colon", site_code="C182",
        histology_text="adenocarcinoma", histology_code="8140", behavior="3",
        dx_date="20191099",
        candidate_stratum="competing",
        gold_candidates=('20191099',), index_date="20200127",
        dx_date_why=("The transfer-of-care note of 2020-01-27 says the cancer was diagnosed "
                     "'in the fall of 2019' at another institution, the outside pathology "
                     "never arrived, and nothing else in the chart names a date. "
                     "date_imputation maps fall to month 10 and the day is unknown, giving "
                     "20191099 with month_imputed and day_imputed true and year_imputed "
                     "FALSE — the year was stated outright."),
        tissue=False, recurrence_type="00", recurrence_date="",
        imaging=[],
        dx_status="FOUND",
        dx_flags={"year_imputed": False, "month_imputed": True, "day_imputed": True},
        # No Endoscopy in the background: this chart's ground truth says there is no
        # pathology in the record, and that branch claims biopsies. See `background_types`.
        background_types=("EKG", "Procedure-Note", "Fluoroscopy-Up-to-1-Hr"),
        notes=("The contract's third boundary case is this exact shape ('diagnosed in the "
               "spring of 2010' -> 20100499) and no chart has ever produced one, so the "
               "season table has been rendered into every prompt for weeks and applied to "
               "nothing. The flags are the second half: a run that answers 20191099 and "
               "leaves all three false has given a date it invented two components of and "
               "said it read them."),
        expect={"dx_date": "20191099", "naive_answer": "20200127",
                "requires": "applying the season table and flagging which components it made up"},
        informed_module_design=False,
        designed_from="contract_clause: date_imputation (seasonal phrase -> month, flagged)",
    ),
    Blueprint(
        pid="SYNY05", pattern="the record begins after the diagnosis and never says when",
        trap="record_starts_after",
        site_text="rectum", site_code="C209",
        histology_text="adenocarcinoma", histology_code="8140", behavior="3",
        dx_date="",
        candidate_stratum="no_answer",
        gold_answerability="CORPUS_INSUFFICIENT", index_date="20210308",
        dx_date_why=("CORPUS_INSUFFICIENT. Every note is surveillance and every one refers to "
                     "the diagnosis only as history; the chart states no date, no season, no "
                     "interval and no treatment start, and the outside records are named as "
                     "absent. This is not 'the chart was searched and does not establish one' "
                     "— it is that the documents which would carry it are not in this corpus."),
        tissue=False, recurrence_type="99", recurrence_date="",
        imaging=[],
        dx_status="CORPUS_INSUFFICIENT",
        background_years=0,
        notes=("THE FIRST CHART WHOSE CORRECT ANSWER IS AN ABSTENTION. CORPUS_INSUFFICIENT was "
               "declared in the contract and offered by the tool with no chart that could ever "
               "score it right, so the status could only ever be measured as a mistake. It is "
               "also the control for the opposite error: a smoke run on 2026-08-02 returned "
               "CORPUS_INSUFFICIENT on SYNX06, where the answer was present in dictation "
               "shorthand — a retrieval failure presenting as a fact about the corpus. Neither "
               "direction is measurable without a chart on each side. THAT OBSERVATION IS WHY "
               "THE GAP WAS NOTICED; the trap itself is derived from the contract's own "
               "abstention block, and no card's behaviour shaped it."),
        expect={"dx_date": "CORPUS_INSUFFICIENT", "naive_answer": "EVIDENCE_INSUFFICIENT",
                "requires": "telling an absent record apart from a silent one"},
        informed_module_design=False,
        designed_from="contract_clause: abstention.CORPUS_INSUFFICIENT",
    ),
    Blueprint(
        pid="SYNY06", pattern="suspicious imaging weeks before the biopsy — the later date wins",
        trap="imaging_only_early",
        site_text="body of pancreas", site_code="C251",
        histology_text="adenocarcinoma", histology_code="8140", behavior="3",
        dx_date="20221115",
        candidate_stratum="competing",
        gold_candidates=('20221115', '20221008'),
        gold_rejections={'20221008': 'NONQUALIFYING_RADIOLOGY'}, index_date="20221115",
        dx_date_why=("The CT of 2022-10-08 reads 'HIGHLY SUSPICIOUS FOR MALIGNANCY' and "
                     "recommends tissue sampling. evidence_rules.does_not_count[2] excludes a "
                     "radiology report that merely describes a suspicious mass absent a "
                     "physician's diagnostic statement, and no such statement exists before "
                     "the biopsy. The pathology of 2022-11-15 is the first admissible witness."),
        tissue=True, recurrence_type="00", recurrence_date="",
        imaging=[],
        # Same reason as SYNY01: the earliest admissible witness must be the 2022-11-15
        # pathology, and sixteen background notes claiming biopsies of the same organ put
        # earlier tissue claims in front of it.
        background_types=("EKG", "Procedure-Note", "Fluoroscopy-Up-to-1-Hr"),
        notes=("THE INVERSE TRAP, and the only one in the corpus. SYNX01, SYNX02, SYNX05 and "
               "SYNY01 all reward reaching for the earliest date, so a card written from them "
               "learns 'earlier wins' and cannot be told apart from a card that learned the "
               "rule. Here the earliest candidate is inadmissible. A policy that only "
               "generalises in one direction fails exactly here and nowhere else."),
        expect={"dx_date": "20221115", "naive_answer": "20221008",
                "requires": "refusing an early candidate the evidence rules exclude"},
        informed_module_design=False,
        designed_from="contract_clause: evidence_rules.does_not_count[2] (imaging is not a diagnosis)",
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


# ======================================================================================
# ADVERSARIAL NOTE BUILDERS
#
# One per `Blueprint.trap`. Every one of them writes the answer into the chart in a place a
# straightforward pass does not reach, and leaves a plausible WRONG date sitting where it does.
# The wrong date is the point: it makes a failure a MISMATCH the scorer can count, rather than
# an abstention, which is a different failure with a different owner.
#
# House style is deliberate. These read like the notes beside them — same header, same
# dictation register — because a trap that announces itself typographically is not a trap, it
# is a label, and a model can learn the label without learning to look.
# ======================================================================================


def incidental_nodule_note(bp, name, sex, dob, dtype, d) -> str:
    """The scan that turns out, years later, to have been the tumour already.

    Reads as a negative study, because at the time it WAS one. Nothing here establishes a
    diagnosis; it only fixes a date that a later note reaches back to.
    """
    body = [
        "CLINICAL HISTORY:",
        "  Routine screening. No respiratory complaints.",
        "",
        "FINDINGS:",
        f"  An 8 mm nodule is noted in the {bp.site_text}. Margins are smooth.",
        "  No lymphadenopathy. No effusion.",
        "",
        "IMPRESSION:",
        "  8 mm nodule, likely benign. Attention on routine follow-up imaging.",
        "",
    ]
    return _hdr(bp, name, sex, dob, dtype, d) + "\n".join(body) + "\n"


def retrospective_note(bp, name, sex, dob, dtype, d, earlier: date) -> str:
    """The one sentence that moves the answer years earlier.

    STORE.390 decision_rule: "If a physician states that in retrospect the patient had cancer at
    an earlier date, use the earlier date." The sentence sits in the middle of an ordinary
    post-diagnosis oncology note, which is where such a remark actually gets made.
    """
    body = [
        "SUBJECTIVE:",
        "  Pt returns to review the biopsy result and discuss systemic therapy.",
        "  Tolerating symptoms; performance status preserved.",
        "",
        "ASSESSMENT AND PLAN:",
        f"  1. {bp.histology_text.capitalize()}, {bp.site_text}. Biopsy-confirmed.",
        f"     Reviewing the prior imaging with radiology, the {earlier.isoformat()} nodule in the",
        "     same location represents this same tumour in retrospect; the patient has had this",
        "     malignancy since at least that date. Interval growth is slow and consistent.",
        "  2. Staging complete. Begin systemic therapy.",
        "",
    ]
    return _hdr(bp, name, sex, dob, dtype, d) + "\n".join(body) + "\n"


def infusion_note(bp, name, sex, dob, d, cycle: int) -> str:
    """First-course treatment, administered before anything documents a diagnosis.

    STORE.390 decision_rule: "If first-course treatment began before a diagnosis was documented,
    use the treatment start date." The note records an administration, not an impression — a
    narrative diagnosis here would establish the date by the ordinary rule and there would be no
    trap left.
    """
    body = [
        "ADMINISTRATION RECORD:",
        f"  Cycle {cycle} of planned 4. Carboplatin AUC 5, pemetrexed 500 mg/m2 IV.",
        "  Pre-medications given per protocol. Infusion completed without reaction.",
        "",
        "OBSERVATIONS:",
        "  Vitals stable throughout. Port accessed and de-accessed without difficulty.",
        "  Next cycle in 21 days pending counts.",
        "",
    ]
    return _hdr(bp, name, sex, dob, "Procedure-Note", d) + "\n".join(body) + "\n"


def deferred_pathology_note(bp, name, sex, dob, d) -> str:
    """A report that defers its own conclusion. Establishes NOTHING and dates nothing."""
    body = [
        "SPECIMEN:",
        f"  A. {bp.site_text} — core biopsy",
        "",
        "MICROSCOPIC DESCRIPTION:",
        "  Sections show an atypical epithelial proliferation. The differential includes a",
        "  reactive process and a well-differentiated malignancy. Immunohistochemical stains",
        "  have been ordered and are PENDING at the time of this report.",
        "",
        "FINAL DIAGNOSIS:",
        "  A. DEFERRED pending immunohistochemical stains. SEE ADDENDUM.",
        "",
        "Electronically signed by Pathology.",
        "",
    ]
    return _hdr(bp, name, sex, dob, "Surgical-Pathology-Report", d) + "\n".join(body) + "\n"


def addendum_note(bp, name, sex, dob, d, original: date) -> str:
    """Where the deferred report was actually settled. Different date, different document type.

    Filed under `Surgical-Pathology-Document` rather than `-Report`, which is how an addendum
    reaches a chart and is also why sweeping the type that carried the preliminary does not find
    it. This is the P05 8046 error given somewhere further to go.
    """
    body = [
        f"ADDENDUM to surgical pathology of {original.isoformat()}.",
        "",
        "IMMUNOHISTOCHEMISTRY:",
        "  TTF-1 positive. Napsin A positive. p40 negative.",
        "",
        "ADDENDUM DIAGNOSIS:",
        f"  A. {bp.site_text.upper()}, CORE BIOPSY — {bp.histology_text.upper()}.",
        "     The staining pattern supports the above. The deferred diagnosis is now final.",
        "",
        "Electronically signed by Pathology.",
        "",
    ]
    return _hdr(bp, name, sex, dob, "Surgical-Pathology-Document", d) + "\n".join(body) + "\n"


def buried_impression_note(bp, name, sex, dob, dtype, d, rng) -> str:
    """A clinical diagnosis made in passing, in a note about something else entirely.

    STORE.390 takes the FIRST date, "whether clinically or histologically established", and a
    physician's impression of cancer counts. Here that impression is one clause inside a
    diabetes follow-up, months before any pathology — a specialty nobody searches for an
    oncology diagnosis, in a note whose type prior says it can establish nothing.
    """
    body = [
        "SUBJECTIVE:",
        "  Here for routine diabetes follow-up. Home glucose log reviewed; ranges 110-160.",
        "  No hypoglycaemic episodes. Adherent to metformin.",
        "",
        "OBJECTIVE:",
        f"  A1c {rng.choice(['7.1', '7.4', '6.9'])}%. Weight down 4 kg since last visit.",
        "  Feet examined, monofilament intact bilaterally.",
        "",
        "ASSESSMENT AND PLAN:",
        "  1. Type 2 diabetes, adequately controlled. Continue metformin.",
        "  2. Weight loss. CT abdomen from last week reviewed with radiology today; the",
        f"     {bp.site_text} lesion is malignant in my assessment and explains the weight loss.",
        "     Referred to oncology; they will arrange tissue sampling.",
        "  3. Follow up 3 months or sooner as oncology directs.",
        "",
    ]
    return _hdr(bp, name, sex, dob, dtype, d) + "\n".join(body) + "\n"


def shorthand_note(bp, name, sex, dob, dtype, d) -> str:
    """The diagnosis recorded only in dictation shorthand.

    No full word appears: not "adenocarcinoma", not "carcinoma", not "malignant", not
    "diagnosis". A run that searches the contract's own vocabulary finds nothing and concludes
    the chart is silent. The `controller-reactive` card's advice to try a shorter stem is exactly what
    reaches this — "adeno" hits, "adenocarcinoma" does not — so the chart tests that advice
    rather than merely repeating it.
    """
    body = [
        "SUBJECTIVE:",
        "  Pt in for results. Accompanied by spouse.",
        "",
        "ASSESSMENT AND PLAN:",
        "  1. Bx +ve. Path c/w adenoCA, RUL. D/w pt and spouse at length today.",
        "     Staging w/u ordered: PET, brain MRI. To med onc next wk.",
        "  2. Sx control prn. Rx per below.",
        "",
    ]
    return _hdr(bp, name, sex, dob, dtype, d) + "\n".join(body) + "\n"


def outside_transfer_note(bp, name, sex, dob, dtype, d, month_text: str) -> str:
    """Names an outside biopsy and its month. Carries no diagnosis of its own.

    This is what a chart looks like when the abstractor knew more than the record does. The note
    is a handover: it reports what the patient says was done elsewhere, which is hearsay about a
    document, not a physician's diagnostic statement and not a pathology report.
    """
    body = [
        "SUBJECTIVE:",
        "  New patient, transferred care. Reports a lung biopsy performed at an outside",
        f"  facility in {month_text}. Records requested; not yet received.",
        "  Brought a medication list only.",
        "",
        "ASSESSMENT AND PLAN:",
        "  1. History per patient of a lung lesion biopsied elsewhere. Outside pathology is",
        "     NOT available for review and no report is in this record. Cannot verify the",
        "     result independently today.",
        "  2. Records request re-sent. Restage here if they do not arrive.",
        "",
    ]
    return _hdr(bp, name, sex, dob, dtype, d) + "\n".join(body) + "\n"


def positive_cytology_note(bp, name, sex, dob, d) -> str:
    """Cytology reading POSITIVE, not "suspicious for".

    The distinction is the whole of SYNK03. STORE.390 counts "a pathology or cytology report
    whose interpretation establishes the diagnosis" and separately discounts "cytology carrying
    only an ambiguous term". POSITIVE FOR MALIGNANT CELLS is not an ambiguous term, so the first
    clause reaches it — and the conflict_rules, which are all written around the ambiguous case,
    say nothing about what happens next.
    """
    body = [
        "SPECIMEN:",
        f"  Fine needle aspirate, {bp.site_text}",
        "",
        "INTERPRETATION:",
        "  Cellular aspirate. Malignant epithelial cells are present, forming glandular",
        "  groups. No benign explanation is identified.",
        "",
        "IMPRESSION:",
        "  POSITIVE FOR MALIGNANT CELLS.",
        "",
    ]
    return _hdr(bp, name, sex, dob, "Surgical-Pathology-Document", d) + "\n".join(body) + "\n"


# ======================================================================================
# HELD-OUT NOTE BUILDERS (SYNY01-Y06)
#
# The rule that makes these held out, and it is checkable by a reader rather than promised:
# EVERY TRAP BELOW IS DERIVED FROM A CLAUSE THE CONTRACT STATES AND NO OTHER CHART EXERCISES.
# The clause is named in the blueprint's `designed_from`. None of them was derived from a run
# result, a card's failure mode, or an arm's score — which is the whole difference between
# this set and SYNX01-06, and the only reason a number computed over these six means anything
# about a card that was written before they existed.
#
# House style is the same as the adversarial builders above, for the same reason: a trap that
# announces itself typographically is a label, and a model can learn a label without learning
# to look.
# ======================================================================================


def death_summary_note(bp, name, sex, dob, d) -> str:
    """decision_rule[4]: death-certificate-only, so the date of death IS the date of diagnosis.

    Nothing ante-mortem names a cancer anywhere in this chart. The physician completing the
    summary is stating the diagnosis for the first time, which `counts_as_evidence[2]` admits,
    and `decision_rule[4]` then fixes the date at the death rather than at the discovery.
    """
    body = [
        "CIRCUMSTANCES:",
        f"  Patient expired {d.isoformat()} at home under hospice care. Pronounced by "
        "attending.",
        "",
        "SUMMARY:",
        "  Progressive decline over the preceding eight weeks with anorexia, weight loss and",
        "  jaundice. No tissue was obtained and no oncology referral was completed.",
        "",
        "CAUSE OF DEATH:",
        "  I. a. Hepatic failure",
        f"     b. Metastatic {bp.histology_text or 'carcinoma'} of the {bp.site_text}",
        "  No autopsy requested.",
        "",
        "COMPLETED BY:",
        "  Attending physician of record.",
        "",
    ]
    return _hdr(bp, name, sex, dob, "Death-Summary", d) + "\n".join(body) + "\n"


def year_only_note(bp, name, sex, dob, dtype, d, years_ago: int) -> str:
    """decision_rule[5]: the year cannot be identified, so it is APPROXIMATED.

    Deliberately says "roughly" and gives no month. On its own this is one soft anchor; the
    chart carries a second, hard one (adjuvant therapy in the pharmacy feed) so the year is
    determined and the case is scorable rather than a matter of taste. Finding the second is
    the retrieval work.
    """
    body = [
        "REASON FOR VISIT:",
        "  Establishing care. Transferred from out of state.",
        "",
        "HISTORY:",
        f"  {bp.histology_text.capitalize() if bp.histology_text else 'Carcinoma'} of the "
        f"{bp.site_text}, treated roughly {years_ago} years ago at an outside hospital.",
        "  The patient cannot recall the month and no outside records have been received.",
        "  Treatment was completed and there has been no further therapy since.",
        "",
        "ASSESSMENT:",
        "  Long-term survivor. Continue surveillance.",
        "",
    ]
    return _hdr(bp, name, sex, dob, dtype, d) + "\n".join(body) + "\n"


def adjuvant_pharmacy_note(bp, name, sex, dob, d, agent: str) -> str:
    """The hard anchor for the approximated year: adjuvant therapy has a start date.

    A pharmacy claim is not a diagnostic statement and cannot establish the diagnosis. What it
    can do is bound the year, which is exactly what an approximation needs and exactly the sort
    of source a run looking only at clinical note types never opens.
    """
    body = ["FILLED PRESCRIPTIONS (external claims feed):", ""]
    for k in range(3):
        dd = d + timedelta(days=21 * k)
        body.append(f"  {dd.isoformat()}   {agent:32s} qty 84   ADJUVANT CYCLE {k + 1}")
    body.append("")
    return _hdr(bp, name, sex, dob, RX_TYPE, d) + "\n".join(body) + "\n"


def three_way_note(bp, name, sex, dob, dtype, d, stated: date) -> str:
    """conflict_rules[4]: sources disagree, so take the earliest that satisfies the rules AND
    cite every one of them.

    This note asserts a THIRD date in prose. It is a physician's diagnostic statement, so it is
    admissible; it is also later than the cytology and disagrees with the pathology. A run that
    picks correctly and cites one source has still not met the proof obligation.
    """
    body = [
        "INTERVAL HISTORY:",
        f"  Referred following the {bp.site_text} work-up.",
        "",
        "ASSESSMENT:",
        f"  {bp.histology_text.capitalize() if bp.histology_text else 'Carcinoma'} of the "
        f"{bp.site_text}.",
        f"  Per the referring service the diagnosis was made on {stated.isoformat()}.",
        "",
        "PLAN:",
        "  Staging complete. Proceed to multidisciplinary review.",
        "",
    ]
    return _hdr(bp, name, sex, dob, dtype, d) + "\n".join(body) + "\n"


def seasonal_note(bp, name, sex, dob, dtype, d, season: str, year: int) -> str:
    """date_imputation: a seasonal phrase is the only thing the record offers.

    The contract's third boundary case is exactly this shape and no chart has ever produced
    one. The answer needs the season table AND both imputation flags; the year is stated
    outright, so `year_imputed` must stay false — which is the distinction one boolean could
    not make.
    """
    body = [
        "REASON FOR REFERRAL:",
        "  Transfer of care. Requesting surveillance.",
        "",
        "HISTORY OF PRESENT ILLNESS:",
        f"  The patient reports that {bp.histology_text or 'carcinoma'} of the "
        f"{bp.site_text} was diagnosed in the {season} of {year}",
        "  at another institution. Resection followed; no adjuvant therapy was given.",
        "  Outside pathology has been requested and is not yet available.",
        "",
        "ASSESSMENT:",
        "  Establish surveillance. Obtain outside records.",
        "",
    ]
    return _hdr(bp, name, sex, dob, dtype, d) + "\n".join(body) + "\n"


def history_only_note(bp, name, sex, dob, dtype, d, rng) -> str:
    """The record begins after the diagnosis and never says when it was.

    No date, no season, no interval, no treatment start. The correct answer is
    CORPUS_INSUFFICIENT — the documents that would carry the date are not in this corpus — and
    it is a DIFFERENT report from EVIDENCE_INSUFFICIENT, which says the chart was searched and
    the documents it does hold do not establish one. Until this chart existed the contract
    declared the status, the tool offered it, and nothing could ever score it as right.
    """
    body = [
        "INTERVAL HISTORY:",
        f"  Surveillance visit. History of {bp.histology_text or 'carcinoma'} of the "
        f"{bp.site_text},",
        "  managed in full at another health system before transfer. No outside records are",
        "  present in this chart.",
        "",
        "EXAM:",
        "  Well appearing. Weight stable. No palpable adenopathy.",
        "",
        "ASSESSMENT:",
        f"  No evidence of disease at {rng.choice(['this', 'today\'s'])} visit.",
        "",
        "PLAN:",
        "  Continue interval surveillance.",
        "",
    ]
    return _hdr(bp, name, sex, dob, dtype, d) + "\n".join(body) + "\n"


def suspicious_imaging_note(bp, name, sex, dob, dtype, d) -> str:
    """does_not_count[2]: radiology describing a suspicious mass, with no physician statement.

    THE INVERSE TRAP, and it is the reason this chart is in the set. SYNX01, SYNX02 and SYNX05
    all reward reaching for the EARLIEST date, so a card written from them learns "earlier
    wins". Here the earliest candidate is inadmissible and the later one is the answer. A rule
    that generalises has to survive being run backwards.
    """
    body = [
        "CLINICAL HISTORY:",
        "  Abdominal pain, weight loss.",
        "",
        "FINDINGS:",
        f"  A {3 + (d.day % 4)}.1 cm mass is identified in the {bp.site_text}, with irregular",
        "  margins and heterogeneous enhancement. Regional nodes measure up to 11 mm.",
        "",
        "IMPRESSION:",
        f"  Mass in the {bp.site_text}, HIGHLY SUSPICIOUS FOR MALIGNANCY.",
        "  Tissue sampling is recommended for definitive diagnosis.",
        "",
    ]
    return _hdr(bp, name, sex, dob, dtype, d) + "\n".join(body) + "\n"


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


def _emit_trap(bp: Blueprint, emit, name, sex, dob, dx: date, rng) -> None:
    """Lay out the diagnostic event for an adversarial chart.

    One branch per `Blueprint.trap`. `dx` here is the INDEX date — where the workup happens —
    which for `retrospective` is two years after the answer.

    An unknown trap raises. A chart that silently fell through to no diagnostic event at all
    would still generate, still ship a ground truth naming a date, and be unanswerable from its
    own contents — a corpus bug that reads from the outside exactly like an agent failure.
    """
    onc = "Onc-Med-MD-OP-Progress-Note"

    if bp.trap == "retrospective":
        earlier = _d(bp.dx_date)
        # The scan that was already the tumour. Emitted at its own date, years before the
        # index workup, so it sits among the background notes rather than beside the biopsy.
        emit("Chest-CT-WWO-Contr", earlier,
             incidental_nodule_note(bp, name, sex, dob, "Chest-CT-WWO-Contr", earlier))
        emit("Surgical-Pathology-Report", dx,
             pathology_note(bp, name, sex, dob, "Surgical-Pathology-Report", dx))
        look_back = dx + timedelta(days=7)
        emit(onc, look_back, retrospective_note(bp, name, sex, dob, onc, look_back, earlier))

    elif bp.trap == "treatment_first":
        start = _d(bp.dx_date)
        for cycle, offset in enumerate((0, 21), start=1):
            cd = start + timedelta(days=offset)
            emit("Procedure-Note", cd, infusion_note(bp, name, sex, dob, cd, cycle))
        emit("Surgical-Pathology-Report", dx,
             pathology_note(bp, name, sex, dob, "Surgical-Pathology-Report", dx))
        emit(onc, dx + timedelta(days=3),
             progress_note(bp, name, sex, dob, onc, dx + timedelta(days=3), rng, kind="initial"))

    elif bp.trap == "cytology_no_impression":
        # Byte-identical in shape to SYN0001's cytology. The ONLY difference between the two
        # charts is that no clinical impression is emitted on this date — which is what flips
        # the conflict rule, and it is an absence, so nothing to search for can reveal it.
        cyto = dx - timedelta(days=23)
        emit("Surgical-Pathology-Document", cyto, cytology_note(bp, name, sex, dob, cyto))
        emit("Surgical-Pathology-Report", dx,
             pathology_note(bp, name, sex, dob, "Surgical-Pathology-Report", dx))
        emit(onc, dx + timedelta(days=2),
             progress_note(bp, name, sex, dob, onc, dx + timedelta(days=2), rng, kind="initial"))

    elif bp.trap == "deferred_addendum":
        emit("Surgical-Pathology-Report", dx, deferred_pathology_note(bp, name, sex, dob, dx))
        add = _d(bp.dx_date)
        emit("Surgical-Pathology-Document", add,
             addendum_note(bp, name, sex, dob, add, dx))
        emit(onc, add + timedelta(days=4),
             progress_note(bp, name, sex, dob, onc, add + timedelta(days=4), rng, kind="initial"))

    elif bp.trap == "buried_late":
        buried = _d(bp.dx_date)
        endo = "Endo-Diab-MD-OP-Progress-Note"
        # The CT the buried impression says it reviewed. Descriptive only: on its own it is a
        # radiology report of a suspicious mass, which the spec explicitly says does not count.
        emit("Abd-Pelvis-CT-W-Contr", buried - timedelta(days=6),
             imaging_note(bp, name, sex, dob, "Abd-Pelvis-CT-W-Contr",
                          buried - timedelta(days=6), rng, mention_mets=False))
        emit(endo, buried, buried_impression_note(bp, name, sex, dob, endo, buried, rng))
        emit("Surgical-Pathology-Report", dx,
             pathology_note(bp, name, sex, dob, "Surgical-Pathology-Report", dx))
        emit(onc, dx + timedelta(days=2),
             progress_note(bp, name, sex, dob, onc, dx + timedelta(days=2), rng, kind="initial"))

    elif bp.trap == "search_resistant":
        # `tissue=False` on this blueprint, so no pathology_note is written anywhere: the
        # shorthand note is the ONLY place the diagnosis exists in the chart.
        #
        # The imaging is written here rather than by the standard index-imaging loop because
        # `imaging_note` says "neoplasm is favored" — one word, and a search of the contract's
        # own vocabulary lands six days from the answer. An incidental-nodule study is the same
        # clinical shape with nothing to match on: the nodule was seen, and months later a
        # biopsy nobody filed came back.
        emit("Chest-CT-W-Contr", dx - timedelta(days=118),
             incidental_nodule_note(bp, name, sex, dob, "Chest-CT-W-Contr",
                                    dx - timedelta(days=118)))
        pulm = "Pulm-MD-OP-Progress-Note"
        emit(pulm, dx, shorthand_note(bp, name, sex, dob, pulm, dx))

    elif bp.trap == "outside_evidence":
        # `tissue=False`: the establishing report is at the other hospital, which is the point.
        # The chart gets a transfer note that NAMES the outside biopsy without reproducing it.
        emit(onc, dx, outside_transfer_note(bp, name, sex, dob, onc, dx, "March"))
        emit(onc, dx + timedelta(days=40),
             progress_note(bp, name, sex, dob, onc, dx + timedelta(days=40), rng,
                           kind="disease_free"))

    elif bp.trap == "key_typo":
        # Ordinary chart, ordinary layout. Everything that makes this case is in the KEY, which
        # names a date on which nothing was written — decidable without reading clinical text,
        # and the reason this is the dispute an evaluation must catch if it catches any.
        emit("Surgical-Pathology-Report", dx,
             pathology_note(bp, name, sex, dob, "Surgical-Pathology-Report", dx))
        emit(onc, dx + timedelta(days=2),
             progress_note(bp, name, sex, dob, onc, dx + timedelta(days=2), rng, kind="initial"))

    elif bp.trap == "genuinely_ambiguous":
        cyto = dx - timedelta(days=35)
        emit("Surgical-Pathology-Document", cyto, positive_cytology_note(bp, name, sex, dob, cyto))
        emit("Surgical-Pathology-Report", dx,
             pathology_note(bp, name, sex, dob, "Surgical-Pathology-Report", dx))
        emit(onc, dx + timedelta(days=2),
             progress_note(bp, name, sex, dob, onc, dx + timedelta(days=2), rng, kind="initial"))

    # ---- held-out traps (SYNY01-Y06) ------------------------------------------------
    elif bp.trap == "death_certificate_only":
        # `tissue=False`, and NOTHING ante-mortem names a cancer. The decline is written as
        # decline: weight loss, jaundice, hospice. The first and only diagnostic statement in
        # the chart is on the death summary, and decision_rule[4] puts the date there.
        for k, off in enumerate((-84, -49, -21)):
            hd = dx + timedelta(days=off)
            emit("GI-Gen-MD-OP-Progress-Note", hd,
                 progress_note(bp, name, sex, dob, "GI-Gen-MD-OP-Progress-Note", hd, rng,
                               kind="interval"))
        emit("Death-Summary", dx, death_summary_note(bp, name, sex, dob, dx))

    elif bp.trap == "year_only_approximate":
        # TWO anchors and neither is a date. The soft one is prose ("roughly four years ago")
        # in a note whose own date does the arithmetic; the hard one is an adjuvant course in
        # the pharmacy feed, which cannot establish a diagnosis but can bound a year. The
        # answer is an APPROXIMATED year with month and day unknown -- 20159999 shaped, which
        # is the notation two E4 runs reached for and wrote as 20999999 instead.
        anchor = date(dx.year - 4, 8, 17)
        emit(RX_TYPE, anchor, adjuvant_pharmacy_note(bp, name, sex, dob, anchor,
                                                     "CAPECITABINE 500MG TAB"))
        gi = "GI-Gen-MD-OP-Progress-Note"
        emit(gi, dx, year_only_note(bp, name, sex, dob, gi, dx, 4))
        emit(gi, dx + timedelta(days=190),
             progress_note(bp, name, sex, dob, gi, dx + timedelta(days=190), rng,
                           kind="disease_free"))

    elif bp.trap == "three_sources_disagree":
        # Three admissible sources, three dates, and the earliest is the answer. The point is
        # not the arithmetic -- it is that `conflict_rules[4]` demands EVERY conflicting source
        # be cited, and a run that picks right and cites one has not met the obligation.
        cyto = _d(bp.dx_date)
        emit("Surgical-Pathology-Document", cyto, cytology_note(bp, name, sex, dob, cyto))
        emit("Onc-Med-MD-OP-Progress-Note", cyto,
             progress_note(bp, name, sex, dob, "Onc-Med-MD-OP-Progress-Note", cyto, rng,
                           kind="initial"))
        emit("Surgical-Pathology-Report", dx,
             pathology_note(bp, name, sex, dob, "Surgical-Pathology-Report", dx))
        stated = dx + timedelta(days=9)
        emit("Onc-Med-MD-OP-Progress-Note", stated + timedelta(days=5),
             three_way_note(bp, name, sex, dob, "Onc-Med-MD-OP-Progress-Note",
                            stated + timedelta(days=5), stated))

    elif bp.trap == "seasonal_phrase":
        # `tissue=False`: the resection was elsewhere and the outside pathology never arrived.
        # A season and a year is everything the record offers, which is the contract's own
        # third boundary case and has never been generated.
        gi = "GI-Gen-MD-OP-Progress-Note"
        # `_d` cannot parse this chart's own answer, and that is the answer being correct:
        # `20191099` is a partial date, month known and day unknown. Take the year off the
        # string rather than round-tripping through a `date` the notation cannot become.
        emit(gi, dx, seasonal_note(bp, name, sex, dob, gi, dx, "fall", int(bp.dx_date[:4])))
        emit(gi, dx + timedelta(days=180),
             progress_note(bp, name, sex, dob, gi, dx + timedelta(days=180), rng,
                           kind="disease_free"))

    elif bp.trap == "record_starts_after":
        # `tissue=False`, no imaging, no interval, no season, no treatment start. Every note is
        # surveillance and every one of them refers to the diagnosis only as history. The
        # correct answer is CORPUS_INSUFFICIENT and there is nothing to find that would change
        # it -- which is what makes it a test of the ABSTENTION rather than of retrieval.
        onc = "Onc-Med-MD-OP-Progress-Note"
        for k in range(9):
            hd = dx + timedelta(days=119 * k)
            emit(onc, hd, history_only_note(bp, name, sex, dob, onc, hd, rng))

    elif bp.trap == "imaging_only_early":
        # THE INVERSE TRAP. Suspicious imaging with no physician statement is inadmissible by
        # `does_not_count[2]`, so the answer is the LATER biopsy. Every other adversarial chart
        # rewards reaching earlier; a rule that only generalises in one direction fails here.
        early = _d(bp.index_date) - timedelta(days=38)
        emit("Abd-Pelvis-CT-W-Contr", early,
             suspicious_imaging_note(bp, name, sex, dob, "Abd-Pelvis-CT-W-Contr", early))
        emit("Surgical-Pathology-Report", dx,
             pathology_note(bp, name, sex, dob, "Surgical-Pathology-Report", dx))
        emit(onc, dx + timedelta(days=3),
             progress_note(bp, name, sex, dob, onc, dx + timedelta(days=3), rng, kind="initial"))

    else:
        raise ValueError(
            f"{bp.pid}: unknown trap {bp.trap!r}. Add a branch here, or the chart generates "
            f"with no diagnostic event and a ground truth nothing in it supports.")


def build_patient(bp: Blueprint, out_root: Path) -> dict:
    # THE ORIGINAL ARITHMETIC IS PRESERVED FOR NUMERIC IDS, AND THAT IS THE WHOLE POINT.
    # `int(bp.pid[3:]) * 7919` assumed every id ends in digits; it raised on the first
    # adversarial chart. Replacing it outright with a digest also "worked" — and silently
    # regenerated all twelve original charts under new seeds, taking SYN0001 from 321 documents
    # to 310. Every committed pilot number was measured on the old bytes, so that is not a
    # refactor, it is quietly invalidating the baseline while the corpus still looks like the
    # corpus. Numeric ids keep their arithmetic; anything else gets a digest.
    #
    # The digest is sha256 and not `hash()`: `hash()` is salted per process unless
    # PYTHONHASHSEED is pinned, and this module's docstring promises byte-identical
    # regeneration.
    tail = bp.pid[3:]
    seed = (int(tail) * 7919 if tail.isdigit()
            else int(hashlib.sha256(bp.pid.encode("utf-8")).hexdigest()[:12], 16))
    rng = random.Random(seed)
    name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
    sex = rng.choice(["M", "F"])
    # THE LAYOUT DATE IS NOT ALWAYS THE ANSWER. `dx` anchors the decade of background notes,
    # the index imaging and the follow-up arc; `bp.dx_date` is the ground truth. They differ
    # exactly when the answer precedes the workup that found it (`retrospective`), and using
    # one for the other would file ten years of routine care after the event it precedes.
    dx = _d(bp.index_date or bp.dx_date)
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
    start = dx - timedelta(days=365 * bp.background_years)
    day = start
    while day < dx - timedelta(days=10):
        if rng.random() < 0.80:
            emit(RX_TYPE, day, rx_note(bp, name, sex, dob, day, rng))
        if rng.random() < 0.55:
            t = rng.choice(PROGRESS_SPECIALTIES)
            d2 = day + timedelta(days=rng.randint(0, 20))
            emit(t, d2, progress_note(bp, name, sex, dob, t, d2, rng, kind="interval"))
        if rng.random() < 0.35:
            t = rng.choice(list(bp.background_types) or OTHER_TYPES)
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
    if bp.trap:
        _emit_trap(bp, emit, name, sex, dob, dx, rng)
    elif bp.pid == "SYN0001":
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
    if bp.followup == "terminal":
        pass                                    # the index date is the last day of the record
    elif bp.followup in ("interior", "truncated"):
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
    while bp.followup != "terminal" and pd_ <= last_day:
        yrs = (pd_ - dx).days / 365.25
        in_hole = bp.followup == "interior" and bp.gap_years[0] <= yrs < bp.gap_years[1]
        past_end = bp.followup == "truncated" and yrs > bp.truncate_after_years
        if not (in_hole or past_end):
            emit(RX_TYPE, pd_, rx_note(bp, name, sex, dob, pd_, rng))
        pd_ += timedelta(days=rng.randint(26, 40))

    obs_end = max(d for _, d in written)
    # A chart may not CLAIM to be held out without naming where its trap came from. The claim
    # is the thing a headline number rests on, and an unbacked claim is worse than the default.
    if not bp.informed_module_design and not bp.designed_from.strip():
        raise ValueError(
            f"{bp.pid}: informed_module_design=False with no `designed_from`. A chart that is "
            f"held out has to say what clause of the contract its trap came from, because that "
            f"is the sentence a reader checks against the contract. Without it the claim is "
            f"unfalsifiable and the chart is treated as informed.")
    bad = [c for c in bp.gold_rejections.values() if c not in REJECTION_CODES]
    if bad:
        raise ValueError(f"{bp.pid}: rejection code(s) {bad} not in REJECTION_CODES.")
    if bp.gold_candidates and bp.dx_date and bp.dx_date not in bp.gold_candidates:
        raise ValueError(f"{bp.pid}: the gold ANSWER {bp.dx_date} is not in gold_candidates; a "
                         f"candidate set without the winner in it cannot score anything.")
    gt = {
        "patient_id": bp.pid,
        "evidence_pattern": bp.pattern,
        # THE CANDIDATE-LEVEL GOLD. Empty on a chart nobody has stated it for, which the
        # analyser reads as "exclude from the candidate metrics" rather than as zero.
        "candidate_stratum": bp.candidate_stratum,
        "gold_candidates": list(bp.gold_candidates),
        "gold_rejections": dict(bp.gold_rejections),
        "gold_answerability": bp.gold_answerability,
        "designer_notes": bp.notes,
        # WHETHER THIS CHART MAY BE SCORED AS A HEADLINE NUMBER. See the Blueprint field: the
        # SYNX charts were designed by watching runs fail and the cards were written from the
        # same failures, so a card's score on them is a score on its own development set.
        # `tools/analyze_arms.py` refuses to fold the two populations together.
        "informed_module_design": bp.informed_module_design,
        "designed_from": bp.designed_from,
        "n_documents": len(written),
        "date_range": [min(d for _, d in written).isoformat(), max(d for _, d in written).isoformat()],
        "ground_truth": {
            "STORE.390.date_of_initial_diagnosis": {
                # `value` is null for an abstention rather than "" — an empty string is a value
                # the field's format would reject, and a reader cannot tell it from a value
                # nobody filled in.
                "value": bp.dx_date or None,
                "status": bp.dx_status,
                # The three imputation flags, present only when the answer carries one. A chart
                # whose date is fully read from the record declares nothing, which is the same
                # as declaring all three false and is shorter to read.
                **({"flags": bp.dx_flags} if bp.dx_flags else {}),
                "why": bp.dx_date_why},
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
    if bp.dispute:
        # SEPARATE FROM `ground_truth` ON PURPOSE. `ground_truth` is what a deployment has —
        # the registry's value, wrong or right — and `acr eval score` reads it, so an agent
        # that reads the chart correctly against a bad key scores MISMATCH exactly as it would
        # in production. This block is the designer's record of what is actually going on, and
        # it scores the EVALUATION, never the agent. Anything that fed it to a run would be
        # handing over the answer to the question the run exists to ask.
        gt["key_dispute"] = dict(bp.dispute)
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
