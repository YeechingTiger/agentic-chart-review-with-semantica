# Related work — the landscape this project sits in

Compiled 2026-08-07 from a seven-thread literature sweep. This file is a **bibliography with
provenance**, not an argument. The argument that draws on it belongs elsewhere.

## How to read this file

**No full text was read.** The session that compiled this had every academic host blocked by an
egress proxy — arxiv.org, pubmed.ncbi.nlm.nih.gov, pmc.ncbi.nlm.nih.gov, medrxiv.org,
api.semanticscholar.org, api.openalex.org, api.crossref.org, seer.cancer.gov, naaccr.org,
journals.plos.org, huggingface.co, ebi.ac.uk. Web search was the only channel.

Three consequences, and each is load-bearing:

* **Every number below is `SNIPPET`** — read from a search-result summary, never from a paper body.
  A number here is a lead to verify, not a fact to cite.
* **`VERIFIED` means the title was seen at two or more independent hosts.** That establishes the work
  exists and is not a fabricated mashup. It does not establish anything about its contents.
* **`UNVERIFIED` means one host only.** Under the compiling skill's rule — *a reference you cannot
  confirm exists does not go in the report* — these were kept only where the single host is
  authoritative (ACL Anthology, a publisher of record) or where the item is important enough that
  its absence would mislead. Each is marked.

Claims of the form *"nobody has published X"* appear in §R. They are **negative results from roughly
200 searches**, not proofs. They are the most valuable and the most fragile findings here.

Before any of this reaches a manuscript, read §S — five documents whose contents change what can be
claimed, and §R — entries that will become two bibliography rows by accident if nobody is watching.

---

## A. Rule-based and dictionary pipelines

The paradigm this project's baseline arm has to beat, and the one the "why not traditional NLP"
question is really about.

| Work | Status | Note |
|---|---|---|
| Savova et al., **cTAKES**: "Mayo clinical Text Analysis and Knowledge Extraction System: architecture, component evaluation and applications," *JAMIA* 17(5):507, 2010. [OUP](https://academic.oup.com/jamia/article-abstract/17/5/507/830823) · [code](https://github.com/apache/ctakes) · [component guide](https://cwiki.apache.org/confluence/display/CTAKES/cTAKES+4.0+Component+Use+Guide) | VERIFIED (3 hosts) | SNIPPET: NER **F = 0.715 exact span**, 0.824 overlapping, against component accuracies 0.936–0.949. **The end-to-end number is far below the component numbers people quote.** |
| Aronson & Lang, **MetaMap** overview. [PubMed](https://pubmed.ncbi.nlm.nih.gov/20442139/) · Demner-Fushman, Rogers & Aronson, **MetaMap Lite**, *JAMIA* 24(4):841, 2017. [OUP](https://academic.oup.com/jamia/article-abstract/24/4/841/2961848) · [tool](https://metamap.nlm.nih.gov/) | VERIFIED | Pure concept→CUI mapping. No relations, no patient, no time. SNIPPET: "comparable to or exceeding" MetaMap and cTAKES; **no per-corpus F1 surfaced — do not cite a number.** |
| Soysal et al., **CLAMP**: "a toolkit for efficiently building customized clinical NLP pipelines," *JAMIA* 25(3):331, 2018. [OUP](https://academic.oup.com/jamia/article/25/3/331/4657212) · [tool](https://clamp.uth.edu/) | VERIFIED | Default NER is a CRF trained on **i2b2 2010**. Customization = annotate in the GUI and retrain. **No numeric F reported in any snippet — do not cite a number.** |
| Chapman et al., **NegEx**: "A Simple Algorithm for Identifying Negated Findings and Diseases in Discharge Summaries," *JBI* 2001. [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1532046401910299) | VERIFIED | 35 trigger phrases, 5-token window. SNIPPET: specificity 94.5%, PPV 84.5%, **sensitivity 77.8% — below the 88.3% baseline.** It bought precision with recall. |
| Harkema, Dowling, Thornblade & Chapman, **ConText**: "an algorithm for determining negation, experiencer, and temporal status from clinical reports," *JBI* 2009. [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1532046409000744) · [ACL 2007 precursor](https://aclanthology.org/W07-1011.pdf) | VERIFIED | SNIPPET: negation P/R 97.0/97.0; temporality-historical 67.4/74.2; **experiencer P 50.0 / R 100**. Patient-vs-family attribution at coin-flip precision in the originating evaluation. |
| Denny et al., **SecTag**: "Evaluation of a method to identify and categorize section headers in clinical documents," *JAMIA* 16(6):806, 2009. [OUP](https://academic.oup.com/jamia/article-abstract/16/6/806/735645) | VERIFIED | Terminology of **1,129 canonical section labels / 6,773 variants** — built for one institution. SNIPPET: labeled sections R/P 99.0/95.6; **implied sections 96.6/86.8, correct boundaries for only 85.9%.** |
| Friedman et al., **MedLEE**: "A broad-coverage natural language processing system," AMIA 2000. [PubMed](https://pubmed.ncbi.nlm.nih.gov/11079887/) · ["Towards a comprehensive medical language processing system"](https://pubmed.ncbi.nlm.nih.gov/9357695/) · [patent US6055494A](https://patents.google.com/patent/US6055494A/en) | VERIFIED | ~350 hand-written grammar rules, ~1,700 words + ~1,400 phrases hand-built lexicon, extended by hand per report type. SNIPPET: **a UMLS-derived lexicon degraded performance** relative to the manual one — the knowledge cannot be bought off the shelf. SNIPPET: 96% chest radiography vs 93% discharge summaries vs 85% vital-sign extraction, attributed to "smaller variability of potential findings in radiography reports." |

**The architectural fact, which is what actually matters.** cTAKES/CLAMP/MetaMap process one UIMA
CAS — one document — through a pre-declared annotator DAG; documents are pushed in by a collection
reader. There is no retrieval step. The system cannot follow "see path report 3/14," cannot fetch a
note it did not know it needed, cannot issue a second query from what it just read. **The document
set is fixed at input time.** The evidence that this is structural rather than incidental is that
cross-document synthesis had to be built as a separate system (DeepPhe, §D) whose mechanism is
coreference plus hand-written summarization rules.

---

## B. Supervised and transformer clinical NLP

| Work | Status | Note |
|---|---|---|
| Alsentzer et al., "Publicly Available Clinical BERT Embeddings," Clinical NLP Workshop 2019. [ACL](https://aclanthology.org/W19-1909/) · [code](https://github.com/EmilyAlsentzer/clinicalBERT) | VERIFIED | SNIPPET: domain pretraining improved **3 of 5** clinical tasks. 512-wordpiece input cap. |
| Lee et al., **BioBERT**, *Bioinformatics* 36(4):1234, 2020. [OUP](https://academic.oup.com/bioinformatics/article/36/4/1234/5566506) · [code](https://github.com/dmis-lab/biobert) | VERIFIED | SNIPPET: **+0.62% F1 NER**, +2.80% RE, +12.24% MRR QA over prior SOTA. Domain pretraining buys single digits on NER and nothing on the labeling requirement. |
| Gu et al., **PubMedBERT** + **BLURB**, *ACM Trans. Computing for Healthcare* 2021. [arXiv](https://arxiv.org/abs/2007.15779) · [ACM](https://dl.acm.org/doi/fullHtml/10.1145/3458754) | VERIFIED | BLURB = 5 NER + 1 PICO + 3 RE + 1 similarity + 1 doc-class + 2 QA. **13 benchmark tasks = 13 separately annotated corpora.** |
| Yang et al., **GatorTron**, *npj Digital Medicine* 2022. [Nature](https://www.nature.com/articles/s41746-022-00742-2) · [arXiv](https://arxiv.org/abs/2203.03540) | VERIFIED | SNIPPET: pretrained on **>90B words / 290M notes / 126 departments**; 345M–8.9B params. Concept extraction F1 **0.8996** (2010 i2b2), **0.8091** (2012 i2b2), **0.9000** (2018 n2c2). **Two orders of magnitude more pretraining does not remove per-task annotation.** |
| Peng et al., **GatorTronGPT**, *npj Digital Medicine* 2023. [Nature](https://www.nature.com/articles/s41746-023-00958-w) | VERIFIED | 277B words, 20B params. Framed as generation + synthetic data, not as an agent. |
| Li et al., **Clinical-Longformer / Clinical-BigBird**, 2022. [arXiv](https://arxiv.org/abs/2201.11838) | UNVERIFIED (arXiv + HF model card) | **512 → 4,096 tokens.** That is roughly one long discharge summary. The ceiling on context for the encoder family, and it is fixed at architecture time. |
| Rasmy et al., **Med-BERT**, *npj Digital Medicine* 2021. [Nature](https://www.nature.com/articles/s41746-021-00455-y) · [code](https://github.com/ZhiGroup/Med-BERT) | VERIFIED | **Structured codes, not text.** Include only for contrast — it is often miscited alongside ClinicalBERT. See §O for its label-efficiency claim and the rebuttal. |

---

## C. Shared tasks and benchmarks

**The i2b2 / n2c2 series** — [portal](https://n2c2.dbmi.hms.harvard.edu/) · [datasets](https://www.i2b2.org/NLP/DataSets/).
Repeatedly described as the field's de facto task taxonomy: the tasks *are* the taxonomy.

| Year | Task | Scale (SNIPPET) |
|---|---|---|
| 2006 | Smoking status | 928 records, 5 classes, 2 pulmonologists |
| 2008 | Obesity + comorbidities | Present/Absent/Questionable/Unmentioned |
| 2009 | Medication extraction | later reused by Steinkamp (§O) |
| 2010 | Concepts / assertions / relations | 394 training docs originally; **public release now 170** |
| 2011 | Coreference | see §H |
| 2012 | Temporal | see §H |
| 2014 | De-identification + heart disease risk factors | — |
| 2018 T1 | **Cohort selection** | 288 diabetic patients, 2–5 notes each, **13 criteria**, 47 teams; best micro-F1 **0.91, from a rule-based system** |
| 2018 T2 | ADE + medication | 505 MIMIC-III discharge summaries; best F1 **0.9418** concept, **0.9630** relation, **0.8905 end-to-end** |
| 2019 | Family history | synthetic narratives |
| 2022 T2 | **SDOH** | SHAC, 4,405 social history sections; **0.901 in-domain → 0.774 cross-institution → 0.889 with target data** |

Overview papers: [2010 i2b2/VA](https://academic.oup.com/jamia/article-abstract/18/5/552/830538) ·
[2018 Track 2](https://academic.oup.com/jamia/article-abstract/27/1/3/5581277) ·
[2018 Track 1 cohort selection](https://pubmed.ncbi.nlm.nih.gov/31562516/) ·
[2022 SDOH](https://arxiv.org/pdf/2301.05571).

**Two structural observations about this series, and they are the point.**

1. **Every task takes a pre-selected span or section as input.** The 2022 SDOH task literally hands
   the system the social history section. No task in the series requires or permits a system to
   decide which part of the chart to read. **The benchmark family never tested retrieval, because
   the input was always pre-scoped by the task designers.**
2. **n2c2 2018 Track 1 is binary — met / not met, no "unknown."** Anything evaluated on it is
   structurally incapable of measuring abstention. See §F and §R.

**Clinical QA benchmarks, sorted by whether the answer's location is given.**

| Benchmark | Location given? | |
|---|---|---|
| MedQA | Yes — no chart at all | [overview](https://www.emergentmind.com/topics/medqa-and-medmcqa) |
| emrQA | Yes — passage supplied; questions generated from i2b2 logical-form templates, so **the query set is enumerable by construction** | [site](https://emrqa.github.io/) |
| RadQA | Yes — single MIMIC-III radiology report. 3,074 questions | [S2](https://www.semanticscholar.org/paper/bd0d9d7b373f18ace4dea46a7038a3a0269ac947) |
| emrKBQA | Yes — logical forms over structured EHR; the enumerable extreme | [S2](https://www.semanticscholar.org/paper/c1aa45a570f1d445a70988c6ace89e1b19405cd2) |
| CliCR, MedNLI, DiSCQ | Yes | seen only inside other papers — UNVERIFIED |
| EHRNoteQA | Patient-specific, single-turn | [arXiv 2402.16040](https://arxiv.org/abs/2402.16040) |
| EHRXQA | Multi-modal | [arXiv 2310.18652](https://arxiv.org/abs/2310.18652) |
| **EHRNote-ChatQA** | **No** — multi-note grounding + per-turn evidence identification | [arXiv 2606.15735](https://arxiv.org/html/2606.15735v1), UNVERIFIED |
| **FHIR-AgentBench** | **No** — single-turn vs multi-turn is the studied variable | [arXiv 2509.19319](https://arxiv.org/abs/2509.19319) · [PMLR v297](https://proceedings.mlr.press/v297/lee26a.html), VERIFIED |
| **MedAgentBench** | **No** — agent must navigate | [NEJM AI](https://ai.nejm.org/doi/full/10.1056/AIdbp2500144), VERIFIED |

`DiSCQ` deserves a note: it pairs each question with **the trigger text that prompted it** —
"reading X made me want to ask Y." That is the bridge structure (§I) encoded in a clinical dataset,
without anyone naming it as an axis.

Broader benchmark taxonomies, both of which sort on a different axis than ours:
**MedHELM** ([Nature Medicine](https://www.nature.com/articles/s41591-025-04151-2) · [site](https://medhelm.org/)) —
5 categories / 22 subcategories / 121 tasks, sorted by *clinical workflow purpose*; SNIPPET scores:
note generation 0.73–0.85, **clinical decision support 0.56–0.72, administration 0.53–0.63**.
**DR.BENCH** ([arXiv 2209.14901](https://arxiv.org/abs/2209.14901)) — 6 tasks / 10 datasets, sorted by
*cognitive step*.

---

## D. Deployed cancer-registry NLP

Our exact application domain.

| Work | Status | Note |
|---|---|---|
| Savova et al., **DeepPhe**, *Cancer Research* 77(21):e115, 2017. [AACR](https://aacrjournals.org/cancerres/article/77/21/e115/662609/) · [site](https://deepphe.github.io/mission/) | VERIFIED | cTAKES mentions → coreference aggregation → **hand-written summarization rules**. Self-description: combines "details from multiple documents to form longitudinal summaries, distinguishing it from earlier NLP approaches that focused on individual documents." Research system on a curated UPMC breast corpus; **reports stage.** |
| **DeepPhe-CR**: "NLP Software Services for Cancer Registrar Case Abstraction," *JCO CCI* 2023;7:e2300156, DOI 10.1200/CCI.23.00156. [ASCO](https://ascopubs.org/doi/10.1200/CCI.23.00156) · [PMC10752457](https://pmc.ncbi.nlm.nih.gov/articles/PMC10752457/) | VERIFIED | SNIPPET, reproduced across four queries: **topography, histology, behavior, laterality, grade at 0.79–1.00 F1**, across breast, prostate, lung, colorectal, ovary, pediatric brain, on **two population-based registries**. **Stage, date of diagnosis, first recurrence, class of case are not in the variable set.** |
| Alawad et al., **MT-CNN**: "Automatic extraction of cancer registry reportable information from free-text pathology reports using multitask convolutional neural networks," *JAMIA* 27(1):89, 2020. [OUP](https://academic.oup.com/jamia/article/27/1/89/5618621) · [code](https://github.com/CBIIT/NCI-DOE-Collab-Pilot3-Multitask-Convolutional-Neural-Network) | VERIFIED | ~95,000 pathology reports, Louisiana Tumor Registry. Five fields: site, laterality, behavior, histology, grade. SNIPPET primary site **micro F 0.944 / macro F 0.592** — rare-class collapse in a high-cardinality ICD-O-3 label space. |
| Hsu, Hanson, Coyle, Stevens, Tourassi & Penberthy, "Machine learning and deep learning tools for the automated capture of cancer surveillance data," *JNCI Monographs* 2024;(65):145. [OUP](https://academic.oup.com/jncimono/article/2024/65/145/7727693) | VERIFIED | SEER 50th-anniversary program review. |
| Chen, Negoita, Schwartz, Hsu et al., "Toward real-time reporting of cancer incidence," *JNCI Monographs* 2024;(65):123. [OUP](https://academic.oup.com/jncimono/article/2024/65/123/7727697) | VERIFIED | **The most structurally revealing artifact in the sweep.** SNIPPET: e-path covers **>90%** of new-case tumors; current standards **22 months** to submission / **27.5 months** to reporting; real-time target **2 months**. The 2-month consolidated tumor case covers "time and place of diagnosis, sociodemographics, tumor characteristics." **Stage and first course of treatment appear to be excluded — because at 2 months the treatment has not finished and the source document does not yet exist.** ⚠ The exclusion is a snippet inference; confirm against the element list (§S). |
| "Fully Automated Abstraction of Longitudinal Breast Oncology Records with Off-The-Shelf LLMs," 2026. [PMC13042122](https://pmc.ncbi.nlm.nih.gov/articles/PMC13042122/) | UNVERIFIED-ish (PMC + medRxiv) | **Fixed retrieval pipeline**, four off-the-shelf LLMs, no fine-tuning. Targets exactly the cross-document set: diagnosis and recurrence dates, clinical stage, biomarker subtype, systemic therapies with timing/intent/discontinuation reason. SNIPPET: all four **beat research coordinators** on systemic therapy abstraction; **exact therapy-line reconstruction ran 9 points below a second oncologist**, with inter-LLM disagreement comparable to inter-oncologist. **Flat extraction beats humans; temporal sequence reconstruction does not.** |
| Carrell et al., "Using NLP to Improve Efficiency of Manual Chart Abstraction in Research: The Case of Breast Cancer Recurrence," *Am J Epidemiol* 179(6):749, 2014. [OUP](https://academic.oup.com/aje/article/179/6/749/109613) | VERIFIED | SNIPPET, **attribution unconfirmed**: 92% of recurrences identified, dates within 30 days for 88%. Framed as improving efficiency of manual abstraction, not replacing it. |
| "Automated Extraction of Cancer Registry Data from Pathology Reports: Comparing LLM-Based and Ontology-Driven NLP Platforms" (Brim vs DeepPhe, Johns Hopkins), medRxiv 2026-03. [medRxiv](https://www.medrxiv.org/content/10.64898/2026.03.20.26348915v1.full) | UNVERIFIED | 330 pancreatic + 34 breast path reports. SNIPPET: Brim mean 96.7% pancreatic, **T stage 96.4%**; DeepPhe comparable on N stage but **T stage 83.6% pancreatic / 70.6% breast**. ⚠ breast n=34. **This result puts T stage on the easy side of the divide** — it is often stated in the path report. |
| HiSAN metastasis detection, PMID 40655537. [medRxiv](https://www.medrxiv.org/content/10.1101/2024.12.12.24318789v1.full) | UNVERIFIED | 55,000 annotated path reports, five sites, **binary Metastasis Positive/Negative** — not a recurrence date, and it does not settle recurrence-vs-new-primary. |
| NAACCR: [AI resources for central registries](https://narrative.naaccr.org/artificial-intelligence-resources-for-central-registries/) · [education](https://education.naaccr.org/products/artificial-intelligence-and-natural-language-processing-in-the-cancer-registry-field) | — | **Useful negative finding: NAACCR publishes data-item standards and AI *education*, but no accuracy threshold at which automation is acceptable.** That absence explains why the 95–98% figure below can only be informal. |

### The automation threshold — read this before quoting it

From DeepPhe-CR, reproduced by exact-phrase query:

> "although the performance of the DeepPhe-CR NLP components meets the set goal of 0.75 F1 for a
> computer-assisted abstraction, the goal for full automation is in the 95-98% F1 range,
> **as informally set forward by SEER and others**."

And the companion sentence: "An initial goal with the SEER program at the NCI was extraction of
attributes at F1 scores of 0.75 or better to demonstrate initial feasibility for computer-assisted
abstraction."

Three cautions.

* **"Informally" is load-bearing.** This is not a SEER standard, a NAACCR standard, or a documented
  acceptance criterion — it is the authors reporting a community norm. Write "a threshold the
  DeepPhe-CR authors describe as informally set forward by SEER and others."
* **A near-miss:** one search rendered it as "95%-97%." Exact-phrase search reproduced **95-98%**
  with the "informally" clause. Treat as verified-pending-full-text; confirm the digits (§S).
* **Do not conflate with SEER's real 98%,** which is a **case-ascertainment completeness** standard —
  did the registry find all the cases. Two different 98%s measuring two different things, and the
  coincidence makes the conflation easy.

The sentence that follows is the authors' own argument for selective automation, and it is the
strongest in-domain endorsement of a triage framing:

> "For a subset of the documents, that high level F1 is already achieved. Methods for identifying
> these documents with high confidence need to be developed thus introducing complete automation for
> a portion of the incoming data."

### Which registry variables are documented as hard

**No paper states the dichotomy as a claim.** The sweep found no sentence saying "variables requiring
cross-document synthesis are harder than variables readable from a single pathology report." Do not
let a draft manufacture one.

What the evidence supports is different and arguably stronger: **the dichotomy is visible in what
deployed systems choose to attempt.** Across independent groups the automated variable set is
identical and closed, and so is the excluded set.

| Class A — automated, single pathology report | Class B — not attempted, degraded, or structurally deferred |
|---|---|
| topography, histology, behavior, laterality, grade | summary / clinical stage; date of initial diagnosis; first recurrence; first course of treatment; **class of case (no NLP work found at all)** |

Four mechanisms, in descending order of defensibility:

1. **The information is in no single document.** DeepPhe-CR's multi-document capability is
   *aggregation* of the same five path-derived attributes across documents — reconciliation, not
   derivation of a variable that exists nowhere singly.
2. **The information does not exist yet at pipeline time.** The 2-month vs 22-month gap. Not a
   model-capability claim; no amount of scaling addresses it.
3. **The judgment is contested among humans.** Therapy-line reconstruction sits within
   inter-oncologist disagreement (the LLM longitudinal study above). Where humans disagree, F1
   against a single-annotator gold standard measures the wrong thing.
4. **The literature has not attempted them.** The oncology NLP survey (§Q) names "disease
   progression" as underrepresented across 156 articles.

**Refined claim, and the one that survives the Brim T-stage result:** not "stage is hard" but
**"variables requiring integration across documents *and* across time are hard."**

---

## E. LLM and agentic clinical abstraction

The novelty surface. Highest fabrication risk in the sweep — treat every unverified entry as a lead.

| Work | Status | Note |
|---|---|---|
| **ACIE** — "Configurable Clinical Information Extraction with Agentic RAG," [arXiv 2606.19602](https://arxiv.org/abs/2606.19602), June 2026, University Medicine Essen | **UNVERIFIED — arXiv and arXiv-derived aggregators only** | **The closest published work to this project.** SNIPPET: on-premise agentic RAG over complete patient contexts of "hundreds of heterogeneous documents and thousands of structured data points"; explicitly **replaces retrieve-then-generate with retrieval as a dynamic, content-driven reasoning task**; names temporal reasoning, cross-document dependencies and missing metadata as targets; grounds every answer in source passages. Evaluated inside a real lymphoma registry study: **96.5% clinician acceptance across 7,326 judgments** (range 80–99%). **No non-agentic baseline found. The 96.5% is clinician acceptance of a cited extraction — it conflates "found the evidence" with "got the answer" rather than separating them.** |
| **Multi-agent cancer registry IE**, BioNLP 2026. [ACL Anthology](https://aclanthology.org/2026.bionlp-1.43/) | Single host, but authoritative | **A published negative result on our exact task.** Per the anthology, *"the baseline achieved slightly higher average weighted F1-scores overall."* The authors fall back on modularity, traceability and interpretability rather than accuracy. **Cite it.** |
| **FHIR-AgentBench**, [arXiv 2509.19319](https://arxiv.org/abs/2509.19319) · [PMLR v297](https://proceedings.mlr.press/v297/lee26a.html) · [Verily](https://verily.com/perspectives/Introducing-FHIR-AgentBench) | VERIFIED (4 hosts) | 2,931 clinical questions over HL7 FHIR. SNIPPET: best configuration (multi-turn + retriever + code, o4-mini) **50% answer correctness, 68% retrieval recall, 35% retrieval precision.** Precision far below correctness is direct evidence of answers landing without the right evidence underneath. Also SNIPPET, attribution uncertain: **iterative-refinement agents reach 71% retrieval recall vs 58% single-turn.** Reports consistent failures in multi-hop reasoning and reference resolution. |
| **EHRNote-ChatQA**, [arXiv 2606.15735](https://arxiv.org/html/2606.15735v1) | UNVERIFIED | Built for exactly the split we need: **every content question is paired with an evidence-grounding question.** SNIPPET: 967 patient-level multi-turn samples over 1–5 notes, 16,072 expert-verified pairs (8,036 + 8,036), 11 experts, 22 LLMs. Headline: **"LLMs struggle more with evidence grounding than content answering,"** and multi-turn errors compound. **Not the same paper as EHRNoteQA (§C).** |
| **RaR** — multi-step retrieval for radiology QA, *npj Digital Medicine*. [Nature](https://www.nature.com/articles/s41746-025-02250-5) · [arXiv 2508.00743](https://arxiv.org/abs/2508.00743) | VERIFIED (3 hosts) | SNIPPET: **75% vs 69%** single-step RAG vs 67% zero-shot. Critically: **"Gains were largest in mid-sized and small models (Mistral Large 72%→81%), while very large models showed minimal change."** ⚠ The loop may buy little on frontier models. Must appear in any limitations section. |
| **CLINES**, medRxiv 2025.12.01. [medRxiv](https://www.medrxiv.org/content/10.64898/2025.12.01.25341355v1) | UNVERIFIED | Modular **fixed-pipeline** agent (chunk → extract → attribute → UMLS normalize → date resolve → aggregate). Multi-stage, **not adaptive retrieval**. SNIPPET: beat rule/lexicon systems, transformer encoders and single-prompt LLMs on MIMIC-III, 4CE, CORAL; **+0.21 to +0.38 F1 over the strongest single-prompt LLM**. Largest agentic-over-single-pass margin found. |
| **EHRAgent**, [arXiv 2401.07128](https://arxiv.org/pdf/2401.07128) | UNVERIFIED (PDF URL only) | Code-generating agent over **structured** EHR tables, not narrative chart review. |
| **AgentEHR**, [arXiv 2601.13918](https://arxiv.org/pdf/2601.13918) | UNVERIFIED — title only | Nothing known beyond the title. Could be a competitor; check. |
| **OphthoACR**, [PMC12517364](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12517364/) | UNVERIFIED | Fine-tuned LLM in a pipeline; no agentic loop claimed. |
| **Brim Analytics** (commercial) | — | "LLM-based system that guides and orchestrates abstraction," links every data point to source text. Orchestration ≠ published adaptive retrieval. See §D for the DeepPhe head-to-head. |
| "AI Agents in Clinical Medicine" systematic review, [medRxiv 2025.08.22.25334232](https://www.medrxiv.org/content/10.1101/2025.08.22.25334232v1.full) | UNVERIFIED | 20 peer-reviewed studies, 2024–2025. SNIPPET: **"No study provides a comparison of multi-agent to single-agent systems."** Heterogeneity precluded meta-analysis. |

**Abstention in clinical LLMs.** [ClinDet-Bench](https://www.researchgate.net/publication/408321868_ClinDet-Bench_Beyond_Abstention_Evaluating_Judgment_Determinability_of_LLMs_in_Clinical_Decision-Making),
"When silence is safer," AbstentionBench. Consistent finding: **models rarely refuse when they
should, and refusal rates are lowest in medical tasks.** No chart-abstraction system that abstains
was found — ACIE routes to a human verifier, which is not the model declining.
Also "Knowing When to Abstain: Medical LLMs Under Clinical Uncertainty"
([arXiv 2601.12471](https://arxiv.org/pdf/2601.12471), UNVERIFIED): SNIPPET **abstention precision
71.43% / recall 13.16%**, and abstention rates swing from >50% to 25% **purely on prompt framing**.
If confirmed, any abstention metric must report prompt sensitivity.

**Cost.** No clinical paper reports tokens, latency or dollars per abstracted field for an agentic
loop versus a single pass. The only figures found are non-clinical: ~2.6× token overhead for 5.9×
recall@1 on BRIGHT, and **two retrieval iterations capture 95% of the gains of five**, with query
drift hurting simple questions.

---

## F. Clinical trial matching and cohort identification

The closest neighbouring task, and where the decompose-then-combine pattern already has names.

| Work | Status | Note |
|---|---|---|
| **TrialGPT**: "Matching patients to clinical trials with large language models," *Nature Communications* 2024. [Nature](https://www.nature.com/articles/s41467-024-53081-z) · [NCBI](https://www.ncbi.nlm.nih.gov/research/trialgpt/) · [code](https://github.com/ncbi-nlp/TrialGPT) | VERIFIED (6 hosts) | Three modules: Retrieval → **criterion-level** Matching → **trial-level** Ranking. Per-criterion output: explanation + **list of relevant sentence IDs** + eligibility label. Inclusion label set: `included` / `not included` / **`not enough information`** / `not applicable`. SNIPPET: retrieval >90% recall at <6% of collection; matching **87.3%** over 1,015 pairs; **sentence-ID localization 90.1% P / 87.9% R / 88.6% F1**; screening time −42.6%. |
| Wornow et al., "Zero-Shot Clinical Trial Patient Matching with LLMs," *NEJM AI* 2024. [NEJM AI](https://ai.nejm.org/doi/abs/10.1056/AIcs2400360) · [arXiv 2402.05125](https://arxiv.org/abs/2402.05125) · [code](https://github.com/som-shahlab/clinical_trial_patient_matching) | VERIFIED (5 hosts) | SNIPPET: SOTA on n2c2 2018; manual chart review takes **up to 1 hour/patient**; clinicians judged LLM justifications coherent for **97% of correct decisions and 75% of incorrect ones**. |
| Wong et al., "Scaling Clinical Trial Matching Using LLMs: A Case Study in Oncology," MLHC 2023. [PMLR](https://proceedings.mlr.press/v219/wong23a.html) · [code](https://github.com/microsoft/CTM-LLM) | VERIFIED (4 hosts) | SNIPPET: GPT-4 can "structure elaborate eligibility criteria" and extract **"nested AND/OR/NOT"**. The clearest statement that the combine step is boolean structure recovered from text. |
| "Real-world validation of a multimodal LLM-powered pipeline for high-accuracy clinical trial patient matching," *Communications Medicine* 2025. [Nature](https://www.nature.com/articles/s43856-025-01256-0) | VERIFIED (3 hosts) | Per-criterion labels **Eligible / Not Eligible / Insufficient information**. SNIPPET: **93% benchmark → 87% real-world**, degradation attributed to "difficulties in replicating human decision-making when medical records lack sufficient information." |
| Ghosh, Schneider, Reinicke & Eickhoff, "A Survey on LLM-Assisted Clinical Trial Recruitment," IJCNLP 2025. [ACL Anthology](https://aclanthology.org/2025.ijcnlp-long.35/) | VERIFIED (3 hosts) ⚠ see §R | **The canonical name for our decomposition.** SNIPPET: "two levels of matching: **trial-level direct matching** … and **criterion-level matching with result aggregation**"; criterion-level labels aggregated by "set-based reasoning mechanisms." |
| "Enhancing Patient-Trial Matching With LLMs: A Scoping Review," *JCO CCI* 2025. [ASCO](https://ascopubs.org/doi/10.1200/CCI-25-00071) | VERIFIED | SNIPPET: criterion-level LLM classification in **12 studies**, typically **Eligible / Not Eligible / Unknown**. Names **"decomposed criteria matching"**: criteria "decomposed into multiple simplified, **independent units**," each unit-patient pair evaluated, "unit-level results **aggregated** into a final decision using predefined algorithms or LLMs." |
| Morrison et al., "A systematic review of trial-matching pipelines using large language models" (UCSF). [arXiv 2509.19327](https://arxiv.org/abs/2509.19327) | VERIFIED (2 hosts) | 126 screened → **31 included**, 2020–2025. GPT-4 consistently best where directly compared; "variability in datasets and evaluation metrics limited cross-study comparability." |
| **αNeSy-CTM** — "Neurosymbolic Clinical Trial Matching via LLM-Driven Abduction and Logical Verification." [arXiv 2606.20895](https://arxiv.org/pdf/2606.20895) | UNVERIFIED ⚠ see §R | **The closest problem statement in the corpus, and it takes the opposite stance.** SNIPPET: "LLMs struggle with the **deterministic verification** required for complex eligibility criteria, while purely symbolic methods provide formal rigor but **fail when facing incomplete patient records**." Its remedy is **abduction — inferring the missing fact** — where ours is declaring the silence. Position against it explicitly. |
| Kang, Zhang & Weng, **EliIE**, *JAMIA* 2017. [OUP](https://academic.oup.com/jamia/article-abstract/24/6/1062/3098256) · [code](https://github.com/Tian312/EliIE) | VERIFIED (5 hosts) | SNIPPET: NER F1 0.79, relation 0.89, negation 0.94 — **end-to-end query formalization 0.71**. **The combine step is where the error concentrates.** |
| **Criteria2Query**, *JAMIA* 26(4):294. [OUP](https://academic.oup.com/jamia/article/26/4/294/5308980) · [code](https://github.com/OHDSI/Criteria2Query) · **3.0** with GPT-4, *JBI* 2024 [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1532046424000674) | VERIFIED / 3.0 UNVERIFIED | Splits "from document to sentences"; human-in-the-loop; output is an **ATLAS cohort definition**. |
| "Clinical Trial Eligibility Criteria Decomposition and Parsing with Large Language Models," SHTI 2025. [IOS Press](https://ebooks.iospress.nl/doi/10.3233/SHTI250928) | VERIFIED (2 hosts) | Names the **"Decomposition and Parsing (DP) workflow"** over **"study traits"** ("the smallest meaningful units"), and a task called **"trait computability determination"** — deciding whether a trait can be evaluated at all. The criterion-side analogue of our abstention. |
| **MatchMiner-AI** [arXiv 2412.17228](https://arxiv.org/abs/2412.17228); parent **MatchMiner** [npj Precision Oncology](https://www.nature.com/articles/s41698-022-00312-5) | VERIFIED | Open-source cancer trial matching. |
| **SatIR** — constraint-satisfaction framing of the combine step. [arXiv 2604.08849](https://arxiv.org/pdf/2604.08849) | UNVERIFIED | — |
| **PRISM** [npj Digital Medicine](https://www.nature.com/articles/s41746-024-01274-7) · **LLM-Match** [arXiv 2503.13281](https://arxiv.org/pdf/2503.13281) · **CriteriaMapper** [Sci Rep](https://www.nature.com/articles/s41598-024-77447-x) · **EC2Seq2Sql** [PMC12900307](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12900307/) | UNVERIFIED (1 host each) | Leads. |
| "Transforming oncology clinical trial matching through neuro-symbolic, multi-agent AI and an oncology-specific knowledge graph: prospective evaluation in 3804 patients." [PMC13091143](https://pmc.ncbi.nlm.nih.gov/articles/PMC13091143/) | UNVERIFIED | Large prospective evaluation; worth opening. |

### Names for decompose-then-combine — do not claim this as novel

| Name | Source |
|---|---|
| **criterion-level matching with result aggregation** (vs *trial-level direct matching*) | Ghosh et al., IJCNLP 2025 — **the canonical name** |
| **decomposed criteria matching** | JCO CCI scoping review 2025 |
| **Decomposition and Parsing (DP) workflow** over **study traits** | SHTI 2025 |
| criterion-level prediction → trial-level scoring | TrialGPT |
| **nested AND/OR/NOT** matching logic | Wong et al., MLHC 2023 |
| **DNF question decomposition**; implicit AND over inclusions, implicit NOT-AND over exclusions | described in the JCO CCI review |
| **query formalization** | EliIE — the pre-LLM name |
| **cohort definition** = entry event + inclusion rules + exit criteria, compiled to SQL | The Book of OHDSI, §G |
| neurosymbolic / abductive verification; constraint satisfaction | αNeSy-CTM; SatIR |

---

## G. Computable phenotyping — how a definition is authored and validated

| Work | Status | Note |
|---|---|---|
| **PheKB**: "a catalog and workflow for creating electronic phenotype algorithms for transportability." [VUMC](https://www.vumc.org/cpm/publication/phekb-catalog-and-workflow-creating-electronic-phenotype-algorithms-transportability) | VERIFIED | SNIPPET: algorithms "manually built by researchers with advanced knowledge of the specific disease" and **"require validation through manual chart review by experts before being deposited."** |
| eMERGE validation, *JAMIA* 20(e1):e147. [OUP](https://academic.oup.com/jamia/article/20/e1/e147/2909179) | VERIFIED | SNIPPET: "multisite validation improves phenotype algorithm accuracy"; development and validation work best as an **iterative process**. |
| "Evaluation of the portability of computable phenotypes with NLP in the eMERGE network," *Sci Rep* 2023. [Nature](https://www.nature.com/articles/s41598-023-27481-y) | VERIFIED | SNIPPET: adding NLP improved or maintained precision/recall for all but one of six algorithms, **but "with NLP, development and validation took longer."** Variation driven by clinical document heterogeneity; portability requires **local customizations**. |
| "A case study evaluating the portability of an executable computable phenotype algorithm across multiple institutions and EHR environments," *JAMIA* 25(11):1540. [OUP](https://academic.oup.com/jamia/article/25/11/1540/5075388) | VERIFIED | SNIPPET (BPH): PPV 100% at 2 of 3 sites, >80% at the third; all 4 PhEMA-implementation sites reported PPV ≥90%. |
| "LLMs facilitate the generation of EHR phenotyping algorithms," *JAMIA* 31(9):1994. [OUP](https://academic.oup.com/jamia/article/31/9/1994/7645319) | VERIFIED (3 hosts) | LLMs authoring the **definition**, not evaluating the **record**. Directly adjacent. |
| **PhEMA** (FHIR + CQL). [JAMIA Open](https://academic.oup.com/jamiaopen/article/7/2/ooae034/7668532) · Workbench [JAMIA](https://academic.oup.com/jamia/article-abstract/29/9/1449/6633603) | VERIFIED | Phenotype logic expressed in Clinical Quality Language, "designed to be **human and machine readable**." |
| **Phenoflow**. [PMC8378606](https://pmc.ncbi.nlm.nih.gov/articles/PMC8378606/) | VERIFIED | Definitions "not tightly coupled to a single CDM." |
| **The Book of OHDSI**, Ch. 10 "Defining Cohorts." [book](https://ohdsi.github.io/TheBookOfOhdsi/Cohorts.html) · [Phenotype Library](https://ohdsi.github.io/PhenotypeLibrary/) | Canonical | Cohort definition = **entry event + inclusion criteria + exit criteria**, stored as Circe JSON, compiled to SQL. **The attrition chart is a per-rule accounting of how many patients each conjunct removes — a native audit trail for the combine step.** |
| **PheValuator** — probabilistic gold standard replacing chart review. [PMC10746303](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10746303/) | VERIFIED | An explicit alternative to human chart review for phenotype evaluation. |
| **SDAVV**: "A strategy for validation of variables derived from large-scale EHR data," *JBI*. [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1532046421002082) | VERIFIED | Sample-size formulae from pre-specified PPV/NPV lower bounds; four steps including **a stopping rule for the iterative develop/validate cycle**. The mature form of a pre-declared kill condition. |
| "An online tool for correcting verification bias when validating electronic phenotyping algorithms." [medRxiv](https://www.medrxiv.org/content/10.1101/2023.11.22.23298913.full.pdf) | UNVERIFIED | Verification bias arises because you chart-review the algorithm-positives preferentially. Named and corrected-for. |
| Label-free phenotyping: **PheNorm** [package](https://celehs.github.io/PheNorm/) · **MAP** [bioRxiv](https://www.biorxiv.org/content/10.1101/587436.full.pdf) · **sureLDA** · **LATTE** [arXiv 2305.11407](https://arxiv.org/pdf/2305.11407) | VERIFIED / mixed | **These exist because per-variable annotation does not scale, and they say so.** PheNorm: "the time intensiveness of annotation and feature curation **severely limits** the ability to achieve high-throughput phenotyping." MAP: manual labeling makes supervised approaches **"infeasible for highly multi-phenotype applications … requiring de novo labeling of hundreds to thousands of phenotypes."** |

---

## H. Temporal reasoning, timelines, coreference

| Work | Status | Note |
|---|---|---|
| "Review of Temporal Reasoning in the Clinical Domain for Timeline Extraction: Where we are and where we need to be," *JBI*. [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1532046421001131) | VERIFIED | **The closest thing to a hierarchical clinical task taxonomy.** Decomposes timeline extraction into six subtasks: temporal expression recognition/normalization → event identification → temporal relation identification → **event coreference resolution** → event ordering → visualization. Names the open needs as event coreference, causal reasoning and **cross-document processing**. |
| "Evaluating temporal relations in clinical text: 2012 i2b2 Challenge," *JAMIA*. [PubMed](https://pubmed.ncbi.nlm.nih.gov/23564629/) · end-to-end system [OUP](https://academic.oup.com/jamia/article/20/5/849/727891) | VERIFIED | 310 discharge summaries. SNIPPET cascade: **EVENT F 0.9166 → TLINK on gold events 0.6849 → end-to-end 0.5924.** |
| **Clinical TempEval** SemEval-2015/2016/2017. [2017](https://aclanthology.org/S17-2093.pdf) · [2015](https://www.derczynski.com/papers/clinical-tempeval-semeval.pdf) | VERIFIED | SNIPPET 2017 (colon → brain cancer domain transfer): time expressions >0.55, events >0.70, **temporal relations >0.40**; **~20-point drop from 2016.** |
| **THYME** annotation guidelines. [PMC5657277](https://pmc.ncbi.nlm.nih.gov/articles/PMC5657277/) | VERIFIED | Document sections, historical mentions, event anchoring. |
| "Temporal Relation Extraction in Clinical Texts: A Systematic Review," *ACM CSUR*. [ACM](https://dl.acm.org/doi/fullHtml/10.1145/3462475) | VERIFIED | SNIPPET: **inter-annotator agreement is lower for temporal relations than for events or temporal expressions** — the ceiling itself is lower. |
| **i2b2 2011 coreference**. [task](https://n2c2.dbmi.hms.harvard.edu/challenge/2011-coreference) · [overview](https://pubmed.ncbi.nlm.nih.gov/22366294/) · [guidelines](https://www.i2b2.org/NLP/Coreference/assets/CoreferenceGuidelines.pdf) | VERIFIED | SNIPPET: one system F 0.847. **ODIE corpus: overall F 79.4%, max 82.0% (persons), min 13.1% (diagnostic reagents).** Difficulty is wildly category-dependent; failures attributed to cases "required domain knowledge." |
| "A Cross-document Coreference Dataset for Longitudinal Tracking across Radiology Reports," LREC 2022. [ACL](https://aclanthology.org/2022.lrec-1.393/) | VERIFIED | 5,872 mentions, 638 MIMIC-III radiology reports, 60 patients. "One of the first attempts focusing on CDCR in the clinical domain." |
| Wright-Bettner, Palmer et al., "Cross-document coreference: An approach to capturing coreference without context." [S2](https://www.semanticscholar.org/paper/e59690adc1aa167ac03602f178a2bb707db9b063) | UNVERIFIED (1 host) | Schema built "to further automatic extraction of timelines." SNIPPET: annotation relying **less on context-guided intuition and more on schematic rules** produced more consistent cross-document relations. **An argument that cross-document linking must be rule-enumerable rather than judgment-driven.** |
| General CDCR framing. [Streamlining CDCR](https://arxiv.org/pdf/2009.11032) · [Realistic evaluation principles](https://arxiv.org/pdf/2106.04192) · [Contrastive representation learning for CDCR](https://arxiv.org/pdf/2205.11438) | UNVERIFIED (arXiv) | SNIPPET, and it is the citable statement: CDCR "introduces additional unique challenges, most notably that **lexical similarity is often not a good indicator** … as documents are authored independently." |
| "What's in a Summary? Laying the Groundwork for Advances in Hospital-Course Summarization." [arXiv 2105.00816](https://arxiv.org/pdf/2105.00816) | VERIFIED (2 hosts) | SNIPPET: "an incredibly challenging multi-document summarization task"; **109,000 hospitalizations / 2M source notes**; requires linking problems to symptoms, procedures, medications and observations under temporal and problem-specific constraints. |
| Biomedical entity linking overview. [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S153204642200257X) | VERIFIED | Three named challenges: **ambiguity, variation, absence.** Notes that entity linking = normalization = encoding. |

**Terminology, so nobody invents a word.** Within-document = **event coreference resolution**;
across documents = **cross-document coreference resolution (CDCR)**; mapping a mention to a KB
concept = **entity linking / normalization / encoding**. These are three different things.
**"Deduplication" is not used as a task name in clinical NLP.**

---

## I. Multi-hop, bridge structure, compositionality — the borrowed frame

Our working axis — *can the query set be enumerated in advance, or does a query contain a value that
must first be read from the record* — **has no name in clinical NLP.** These are the nearest
constructs, and they come from open-domain QA.

| Work | Status | Note |
|---|---|---|
| **Bridge / comparison question types** — HotpotQA and 2WikiMultiHopQA taxonomies. [overview](https://www.emergentmind.com/topics/multi-hop-question-answering-mhqa) | **UNVERIFIED — secondary/encyclopedic sources only; primary papers not reached** | A **bridge** question chains through an explicit intermediate entity: the answer to hop 1 is the query term for hop 2. **That is our axis, exactly.** The contrast classes — *comparison*, *intersection* — are our enumerable side: all entities known up front, only their attributes need combining. |
| Press et al., "Measuring and Narrowing the Compositionality Gap in Language Models," *Findings of EMNLP* 2023. [arXiv 2210.03350](https://arxiv.org/abs/2210.03350) · [ACL](https://aclanthology.org/2023.findings-emnlp.378/) · [self-ask](https://github.com/ofirpress/self-ask) | VERIFIED (3 hosts) | **The measurement.** The compositionality gap is the fraction of compositional questions answered incorrectly **among those whose sub-questions the model answers correctly** — it isolates second-hop failure while holding first-hop competence fixed. SNIPPET: **the gap does not shrink with scale.** Read alongside RaR (§E): generic multi-step gains vanish on frontier models, but bridge failure does not. |
| "Beyond Static Retrieval: Opportunities and Pitfalls of Iterative Retrieval in GraphRAG." [arXiv 2509.25530](https://arxiv.org/pdf/2509.25530) | UNVERIFIED | SNIPPET, the operational statement: *"if the first retrieval call fails to surface the key bridge entity, the generator never has a chance to see it."* |
| "Retrieval Augmented Generation (RAG) and Beyond" — the four-level query taxonomy. [arXiv 2409.14924](https://arxiv.org/abs/2409.14924) | VERIFIED (4 hosts) | **L1 explicit fact** → **L2 implicit fact** (requires connecting multiple data points) → **L3 interpretable rationale** → **L4 hidden rationale**. Our boundary sits at L1→L2. But their ladder cuts on *reasoning depth*, not on *whether the retrieval target is knowable a priori*. The most-cited recent taxonomy our axis must be positioned against. |
| Chamberlin, Bedrick, Cohen, Wang et al., "A Query Taxonomy Describes Performance of Patient-Level Retrieval from Electronic Health Record Data." [medRxiv 19012294](https://www.medrxiv.org/content/10.1101/19012294v1) | UNVERIFIED (preprint, 2 hosts) | **The only clinical taxonomy built to predict retrieval difficulty from query structure.** SNIPPET: 59 topic characteristics; **no strong association with individual characteristics**, strong associations with six derived complexity composites. Cohort discovery, not abstraction. **Cite as the ancestor, not as support.** |
| "A Taxonomy for Contextual Information in Electronic Health Records." [PMC3799164](https://pmc.ncbi.nlm.nih.gov/articles/PMC3799164/) | UNVERIFIED | Six attribute themes including **"retrieval effort"** — nearest clinical vocabulary, but it measures clinician burden, not query enumerability. |
| Iterative retrieval / query reformulation systems. [Memory-augmented sequential paragraph retrieval](https://arxiv.org/pdf/2102.03741) · [Memory-aware uncertainty-guided retrieval](https://arxiv.org/pdf/2503.23095) | UNVERIFIED | Consistently defined as: retrieve for sub-questions, **extract key entities from the results to form subsequent queries**. Note the naming asymmetry — the enumerable side has no name because it is the unmarked default ("static retrieval," "single-turn"). |

---

## J. QA practice: registry, trial, hospital

Everything in this section measures **value agreement**. Nothing measures the evidence path.

### Cancer registry re-abstraction

| Work | Status | Note |
|---|---|---|
| German RR et al., "Quality of cancer registry data: findings from CDC-NPCR's Breast and Prostate Cancer Data Quality and Patterns of Care Study." [PubMed](https://pubmed.ncbi.nlm.nih.gov/22096878/) · [design paper](https://pubmed.ncbi.nlm.nih.gov/15801489/) | VERIFIED | **The best-documented US reabstraction study.** SNIPPET: **9,103 breast + 8,995 prostate** cases, independently re-abstracted as gold standard. **Only 53% (8/15) of cancer-site × treatment combinations reached κ ≥ 0.60 *and* percent agreement, sensitivity and PPV ≥ 80%.** Read the inverse: **~47% of treatment variables failed a fairly lenient joint threshold.** |
| "The National Cancer Database Conforms to the Standardized Framework for Registry and Data Quality," *Ann Surg Oncol* 2024. [Springer](https://link.springer.com/article/10.1245/s10434-024-15393-8) | VERIFIED (2 hosts) | SNIPPET: 6,828,507 cases, 73.7% of US cancer cases; histologic verification 99.1%; **"validity criteria for re-abstracting, recording, and reliability procedures across hospitals demonstrated 94.2% compliance."** ⚠ That 94.2% is compliance with **having a procedure**, not agreement produced by it — a structural standard, not a measurement of process quality. |
| NAACCR Volume III, "Standards for Completeness, Quality, Analysis, and Management of Data." [NAACCR](https://www.naaccr.org/standards-for-completeness-quality-analysis-and-management-of-data/) | **UNREAD — blocked** | SNIPPET describes "central registry **structural requirements, process standards and outcome measures**." NAACCR explicitly uses structure/process/outcome vocabulary. **The single highest-value unread document in the sweep** — see §S. |
| SEER QI. [process](https://seer.cancer.gov/qi/process.html) · [tools](https://seer.cancer.gov/qi/tools/) | **UNREAD — blocked** | SNIPPET/UNVERIFIED: casefinding, recoding and reliability studies in even-numbered years; per-registry Data Quality Profile. SNIPPET, **unattributable, do not cite**: "the majority of data items had a coding quality greater than 90 percent, with exceptions found in **cause of death, follow-up source, and SEER Summary Stage**." The identity of the exceptions is the interesting part and it is unverified. |
| Other registry validation: [Indian PBCRs 10% reabstraction, 94.9–97.4% agreement](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8408406/) · [reliability of registry records](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1968748/) · [clinician vs registrar data](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1968711/) · [SEER chemotherapy ascertainment](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5924509/) · [SEER breast receptor vs central lab](https://pmc.ncbi.nlm.nih.gov/articles/PMC3782852/) · [Swedish NPCR κ 0.87–1.0](https://www.sciencedirect.com/science/article/pii/S0959804914010612) | mixed | The [clinician vs registrar](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1968711/) study gives major-disagreement rates: DOB 0%, gender 0%, **date of incidence 5%, primary site 6%**, histologic type 2%, behavior 2%. ⚠ Older, non-US registry — do not lean hard. |

**Useful contrast:** registry-vs-registry agreement is high (94–97%) while treatment-variable
agreement against a gold-standard reabstraction is much weaker (~half failing κ≥0.60).
**Aggregate agreement hides variable-level failure.**

**Still missing:** a cancer-registrar reabstraction study with per-variable agreement separating
stage / date / treatment from site / histology. German et al. is the closest substitute.

### Inter-rater reliability, and which variables fail

| Study | SNIPPET |
|---|---|
| [Transition of care after childhood cancer, PLOS One](https://pmc.ncbi.nlm.nih.gov/articles/PMC4441480/) | Cohen's κ **0.70–0.83**, agreement 86–100% |
| [National acute stroke register IRR](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4617717/) | **κ < 0.40 for stroke onset time, stroke team consultation, time of initial brain imaging, discharge destination.** Worst: **mobilized out of bed within 24 h κ = 0.04**, heart rate monitoring κ = 0.17, swallowing test performed κ = 0.19 |
| [Non-physician abstraction, orthopedics](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3684546/) | "excellent reliability for information with low technical complexity, moderate-to-good for greater complexity" |
| [Intra- and inter-rater, community asthma](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2396663/) · [AIS/trauma coding](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10229665/) · [probabilistic correction for abstractor disagreement](https://www.researchgate.net/publication/6415954) | — |

**The generalization across all of these:** agreement is high for demographic, numeric and
single-location facts and poor for judgment variables and for **variables whose answer requires
locating a *time* or an *event* scattered across the record** — onset time, "was X performed within
24 h," consult occurred, imaging time. **The stroke variables at κ ≈ 0.04–0.19 are almost certainly
search failures rather than judgment failures, and no author in this set frames them that way.**

### Source data verification, and hospital quality abstraction

| Work | Note |
|---|---|
| SDV scoping review [PMC11639101](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11639101/) · ["Evaluating the evaluators"](https://www.sciencedirect.com/science/article/pii/S1551714422000908) · [SDV variability](https://www.sciencedirect.com/science/article/pii/S1532046418300935) | SNIPPET: SDV **"confirms that data collected from source records … remain the same throughout"**; ~**46%** of on-site monitoring time; **25–40%** of trial cost; audit proportion 3–100%; **"no evidence of its effectiveness and significant evidence of its lack of effectiveness."** ⚠ **SDV presupposes the source has already been identified. It is structurally incapable of detecting an unopened decisive document.** |
| [TJC chart-abstracted measures](https://www.jointcommission.org/measurement/specification-manuals/chart-abstracted-measures/) · [CMS Hospital IQR FY2026 guide](https://www.qualityreportingcenter.com/globalassets/2024/06/iqr/2.-hospital-iqr-fy-2026-program-guide_vfinal508_c.pdf) | CMS validates **up to 8 chart-abstracted cases per quarter per hospital**. The two operative metrics — **DEAR** (Data Element Agreement Rate) and **CAAR** (Category Assignment Agreement Rate), DEAR ≥80% acceptable — are **pure outcome-agreement metrics**. ⚠ DEAR/CAAR thresholds sourced to a vendor page, UNVERIFIED. |
| HEDIS over-read | SNIPPET: NCQA requires 95%+ abstraction accuracy; over-readers must **document the error and the specific change made**. ⚠ Payer/industry sources only. No published taxonomy of the error categories used. |
| [ACS NSQIP IRR audits](https://www.carta.healthcare/wp-content/uploads/2025/12/NSQIP-IRR-Audits-to-Improve-Abstraction-Quality.pdf) | SNIPPET: IRR on 5% randomized cases; workstation error rate 0.35% → 0.16% after education. ⚠ Vendor source. |
| [Registry Partners, CathPCI data integrity](https://www.registrypartners.com/project/ensuring-data-quality-and-integrity-in-cathpci-abstraction/) | **The one genuine process-error sighting.** Describes an error from *"coding based on an abstractor viewing cath images rather than proper documentation"* — right value possibly, wrong evidence source. ⚠ Not peer-reviewed. **Trade-practice evidence that practitioners recognise the phenomenon without a name or a metric for it.** |

### The registry text-documentation rule

[California Cancer Registry manual](https://www.ccrcal.org/wp-content/uploads/V1_2017_Online_Manual/Part_I_Introduction/I_1_6_2_Reporting_Methods.htm) ·
Iowa SHRI "Texting 101" (blocked) · [Missouri manual](https://cancerregistry.missouri.edu/wp-content/uploads/2023/12/MCR_Manual_August_2023-1.pdf) ·
[CDC Abstract Plus](https://www.cdc.gov/national-program-cancer-registries/registry-plus/abstract-plus.html)

SNIPPET: **"Coded fields must be supported by text documentation on the abstract."** Stated purposes:
support all coded fields, assure facility data quality, allow review for QC audits, and **"reduce the
need to return to the original medical record to verify information."**

**This is the nearest analogue in registry practice to a provenance requirement, and its stated
design goal is the opposite of ours** — it exists so auditors *do not have to go back to the source
record*. It audits the presence and consistency of abstractor-written justification. It cannot check
whether a document the abstractor never opened would have changed the code.

### EHR audit logs — the substrate that exists and is unused for this

[JAMIA, using audit logs for research](https://academic.oup.com/jamia/article-abstract/30/1/167/6730781) ·
[Huilgol et al., audit logs in cancer care, *Cancer Medicine* 2022](https://onlinelibrary.wiley.com/doi/full/10.1002/cam4.4690) ·
[macrostructure of EHR work](https://pmc.ncbi.nlm.nih.gov/articles/PMC9933072/)

SNIPPET: audit logs "unobtrusively capture a trail of clinician activities," enabling "fine-grained
physician behaviors." Also SNIPPET: **49% of ICU clinicians reported reviewing charts "haphazardly,"**
searching back three or more years.

**The capability to answer "did they open the note" demonstrably exists. No study was found that
pairs audit-log document-open events with the ground-truth decisive document for a given abstracted
variable.**

---

## K. Chart-review methodology

| Work | Status | Note |
|---|---|---|
| Gilbert, Lowenstein, Koziol-McLain, Barta & Steiner, "Chart reviews in emergency medicine research: Where are the methods?" *Ann Emerg Med* 1996. [PubMed](https://pubmed.ncbi.nlm.nih.gov/8599488/) · [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0196064496702640) | VERIFIED (3 hosts) | SNIPPET: **986** articles, **244 (25%)** used chart review. Reporting rates: abstractor training **18%**, standardized forms **11%**, periodic monitoring **4%**, blinding **3%**, IRR mentioned **5%**, **tested statistically 0.4%**. Proposes **eight** criteria. ⚠ **The verbatim eight-item list was not retrieved — see §S.** |
| Worster, Bledsoe et al., "Reassessing the methods of medical record review studies in emergency medicine research," *Ann Emerg Med* 2005. [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0196064404017135) | VERIFIED | Emphasizes describing **whether abstractors were monitored, how frequently, and by whom.** |
| Kaji, Schriger & Green, "Looking through the retrospectoscope: reducing bias in emergency medicine chart review studies," *Ann Emerg Med* 64(3):292, 2014. [PubMed](https://pubmed.ncbi.nlm.nih.gov/24746846/) · [PDF](https://www.annemergmed.com/pb/assets/raw/Health%20Advance/journals/ymem/kaji-1399398337020.pdf) | VERIFIED | Models **10 potential layers of bias** (Figure 1). **The most likely place an evidence-retrieval concept would live if it existed in EM methodology — check it (§S).** |
| Vassar & Holzmann, "The retrospective chart review: important methodological considerations," *J Educ Eval Health Prof* 10:12, 2013. [PubMed](https://pubmed.ncbi.nlm.nih.gov/24324853/) | VERIFIED | **Ten** common methodological mistakes; contents UNVERIFIED. |
| Also: [Bauman, JACCP 2019](https://accpjournals.onlinelibrary.wiley.com/doi/10.1002/jac5.1064) · ["Conducting Retrospective Studies, Audits and Chart Reviews"](https://doi.org/10.3390/ecm3010011) · [Structured Chart Review](https://www.researchgate.net/publication/338214970) | mixed | — |

**The assessment that matters.** The canonical recommendations are training, explicit variable
definitions, standardized forms, blinding, periodic monitoring and IRR testing. **Every one is either
an input control or an output check. None inspects the abstractor's traversal of the record.**
"Periodic monitoring," the closest, means re-checking abstracted *values* in every description found.

---

## L. Error taxonomies, and "right answer, wrong process"

| Work | Status | Note |
|---|---|---|
| **Kundel's search / recognition / decision taxonomy** (radiology). [PMC6603246](https://pmc.ncbi.nlm.nih.gov/articles/PMC6603246/) · [Frontiers](https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2019.00213/full) · [eye-tracking review](https://link.springer.com/article/10.1186/s41235-019-0159-2) | VERIFIED | **The only medical error taxonomy found that formally separates "never looked at it" from "looked and misjudged."** Classifies false negatives by dwell time: **search error** (never fixated) / **recognition error** (fixated but below the ~0.48 s recognition threshold) / **decision error** (fixated, features extracted, dismissed). SNIPPET distribution **25% / 25% / 50%.** Uses eye tracking on **images** — never applied to documents or chart abstraction. **The document-level analogues are: never opened / opened but not read closely / read and misinterpreted.** |
| "Guessing right — whether and how medical students give incorrect reasons for their correct diagnoses." [PubMed](https://pubmed.ncbi.nlm.nih.gov/31844657/) · [PMC6905369](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6905369/) | VERIFIED (2 hosts) | **The measured base rate for "right answer, wrong process."** SNIPPET: of **correct** diagnoses, 92% had correct reasoning, **7% incorrect reasoning**, 1% guessed — about **every 14th correct diagnosis rests on a false explanation**. Causes: lack of pathophysiological knowledge 50%, lack of diagnostic skills 30%. Conclusion: **"To assess diagnostic competence, both the diagnosis result and the diagnostic process should be recorded."** |
| Nahm, *Data Accuracy in Medical Record Abstraction* (dissertation). [TMC](https://digitalcommons.library.tmc.edu/uthshis_dissertations/15/) | **UNREAD** | See §S. |
| Zozus et al., "Factors Affecting Accuracy of Data Abstracted from Medical Records," *PLOS One* 2015. [PLOS](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0138649) | **UNREAD — blocked** | SNIPPET: content validity via a four-round two-group Delphi with expert abstractors; factors organized into a **control-theory-based MRA-QC framework**. ⚠ **This is a taxonomy of *factors affecting accuracy*, not of *error types*.** |
| MRA error rates: ["Measuring and controlling MRA error rates"](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9380367/) · ["Comparing MRA error rates to pooled rates," *BMC Med Res Methodol*](https://link.springer.com/article/10.1186/s12874-024-02424-x) · [training as intervention](https://pubmed.ncbi.nlm.nih.gov/30741251/) | VERIFIED | SNIPPET: literature-pooled MRA error rates **70–5,019 errors per 10,000 fields**; studies using a formal MRA-QC framework achieved **1.04%–2.57%**, i.e. 4.00–5.53 percentage points below the observed literature rate. |
| Systematic-review data extraction (same cognitive task, different corpus): [Buscemi, single vs double extraction](https://pubmed.ncbi.nlm.nih.gov/16765272/) · [Mathes et al., *BMC Med Res Methodol* 2017](https://link.springer.com/article/10.1186/s12874-017-0431-4) | VERIFIED | SNIPPET: single extraction generated **21.7% more errors** (p=.019) but was 36.1% faster. Error rates ~**15–30%, up to 45–50%**; discrepancies concentrated in sample sizes, design, **start/end dates**, selection criteria, secondary outcomes. |
| [AHRQ PSNet, "Getting the Diagnosis Both Right and Wrong"](https://psnet.ahrq.gov/web-mm/getting-diagnosis-both-right-and-wrong) · [NASEM, *Improving Diagnosis in Health Care*](https://www.ncbi.nlm.nih.gov/books/NBK338593/) · [Key Features assessment](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3068728/) | — | Process-oriented clinical assessment precedents. |

---

## M. Evidence attribution and process supervision (the NLP machinery)

The formal apparatus for "did the cited evidence actually drive the answer" exists. It has not been
transplanted to chart review.

| Work | Status | Note |
|---|---|---|
| DeYoung, Jain, Rajani, Lehman, Xiong, Socher & Wallace, **ERASER**, ACL 2020. [ACL](https://aclanthology.org/2020.acl-main.408/) · [arXiv 1911.03429](https://arxiv.org/abs/1911.03429) | VERIFIED | **The formal ancestor.** SNIPPET: **sufficiency** = extent to which the extracted rationale was actually used; **comprehensiveness** = how much the prediction changes when the rationale is **removed**. **Comprehensiveness is literally "does removing this evidence change the answer."** |
| "Correctness is not Faithfulness in RAG Attributions." [arXiv 2412.18004](https://arxiv.org/pdf/2412.18004) | UNVERIFIED | **The name for our failure mode: post-rationalization** — citing documents the model did not actually rely on. SNIPPET: **up to 57% of citations lack faithfulness.** |
| Lightman et al., "Let's Verify Step by Step." [arXiv 2305.20050](https://arxiv.org/pdf/2305.20050) · [OpenReview](https://openreview.net/forum?id=v8L0pN6EOi) | VERIFIED | Process vs outcome supervision. SNIPPET: process-supervised model solved 78% of MATH; **PRM800K = 800,000 step-level human labels** — the cost of process supervision when it is done by hand. |
| ["Do We Need to Verify Step by Step?"](https://arxiv.org/html/2502.10581v1) · [automated process supervision](https://arxiv.org/pdf/2406.06592) · [OVM](https://arxiv.org/pdf/2311.09724) · [FindTheFlaws](https://arxiv.org/pdf/2503.22989) | UNVERIFIED | — |
| "The Clever Hans Mirage: A Comprehensive Survey on Spurious Correlations in ML." [arXiv](https://arxiv.org/html/2402.12715) · [PMC12827554](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12827554/) | mixed | SNIPPET: across 13 datasets **including clinical discharge summaries**, performance overestimated by **up to 20% on average** due to shortcut learning of hidden acquisition biases. |
| ["Right for the Wrong Reasons: Epistemic Regret Minimization…"](https://arxiv.org/html/2602.11675v2) | UNVERIFIED | SNIPPET: "most current evaluation paradigms fixate on whether the final answer is correct while overlooking the reasoning path." |
| **ABSeeker** — "Training Long-Horizon Search Agents via Answer-Backtracked Credit Assignment," arXiv 2608.05102. [code](https://github.com/PolarSeeker/ABSeeker) | code read directly; paper blocked | Answer-Backtracked Clue Recovery → Clue-Anchored Step Scoring → ABC-SFT / ABC-GRPO. **The clue set is consumed as a training signal** — `msg["step_weight"]` in SFT, per-turn rewards in RL. Its incompleteness is benign because it can only under-credit and because an unbiased outcome signal sits underneath. See `docs/` for the analysis of why that property does not transfer to an audit setting. |

**Assertion detection** is the older, coarser version of the unknown-vs-negative distinction:
["Beyond Negation Detection: Comprehensive Assertion Detection Models for Clinical NLP"](https://arxiv.org/abs/2503.17425).
Label space: present / absent / possible / hypothetical / conditional / associated-with-someone-else.
**It has no "not mentioned" — assertion detection presupposes the concept was mentioned.** That is
precisely the gap an abstention distinction fills.

**Informative missingness** — the statistical framing of the same problem:
["Informative Missingness: What can we learn from patterns in missing laboratory data in the EHR?" *JBI*](https://www.sciencedirect.com/science/article/pii/S1532046423000278) (VERIFIED, 3 hosts) ·
[strategies for handling missing data in EHR-derived data](https://pmc.ncbi.nlm.nih.gov/articles/PMC4371484/).
An industry post states it most crisply — for an absent diagnosis code, "evidence of absence and
absence of evidence present in **exactly the same way** … the missingness indicator itself is
missing" — ⚠ **not peer-reviewed; use for framing, cite the JBI paper for authority.**

---

## N. Portability and domain shift

| Work | Status | SNIPPET |
|---|---|---|
| Wu et al., **"Negation's Not Solved: Generalizability Versus Optimizability in Clinical Natural Language Processing,"** *PLOS ONE* 9(11):e112774, 2014. [PLOS](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0112774) · [DTIC PDF](https://apps.dtic.mil/sti/pdfs/AD1107234.pdf) | VERIFIED | **The strongest portability citation in the sweep.** YTEX NegEx: **F1 95.3% on the NegEx Test Set → 62.3% on SHARP data** without modification. Single-OOD-corpus training gave 59.3–95.4% F. Authors: *"an optimizable solution does not equal a generalizable solution"*; the only reliable fix is manual in-domain annotation or manual rule editing. **A ~33-point collapse on the most-solved-looking subtask in clinical NLP.** |
| Sohn et al., "Clinical documentation variations and NLP system portability: a case study in asthma birth cohorts across institutions," *JAMIA* 2018. [PubMed](https://www.ncbi.nlm.nih.gov/pubmed/29202185) | VERIFIED | **F = 0.937 at Mayo → 0.813 at Sanford Children's → 0.908 after refinement**, n=298 each. Word-level similarity **0.669**, semantic similarity **0.971**. **The gap was wording, not content.** |
| n2c2 2022 SDOH Subtask B (built-in domain-shift experiment). [shared task](https://arxiv.org/pdf/2301.05571) · [prompt-tuning follow-up](https://pmc.ncbi.nlm.nih.gov/articles/PMC12150740/) | VERIFIED | Top team **0.901 → 0.774 → 0.889**; drops of **7–17%** across top teams. Decoder-only LLMs with prompt tuning generalized better cross-domain than fine-tuned encoders. ⚠ Measured on a **pre-scoped section** — the easiest possible conditions. |
| "A cross-institutional evaluation on breast cancer phenotyping NLP algorithms on EHR," *CSBJ* 2023. [PMC10480628](https://pmc.ncbi.nlm.nih.gov/articles/PMC10480628/) · [arXiv 2303.08448](https://arxiv.org/abs/2303.08448) | VERIFIED | UMN ↔ Mayo, same annotation guideline. CancerBERT fine-tuned cross-site **0.925 vs 0.932** locally trained. **Pearson correlation between phenotype similarity and performance drop: −0.678 (CRF), −0.345 (BiLSTM-CRF), −0.712 (CancerBERT).** ⚠ **Cuts both ways** — transformers port better than CRFs, but transfer degrades precisely as target concepts become less similar, i.e. on novel and long-tail variables. |
| ["Generalizability and portability of NLP to extract individual social risk factors," *Int J Med Inform* 2023](https://pubmed.ncbi.nlm.nih.gov/37302362/) | VERIFIED | ~**6%** accuracy drop on external geographic evaluation. The honest steel-man for rule-based portability — but even the positive result is conditioned on "institution-specific modification of rules." |
| ["A study of deep learning methods for de-identification of clinical notes in cross-institute settings," *BMC MIDM* 2019](https://link.springer.com/article/10.1186/s12911-019-0935-4) | VERIFIED | "directly adopting challenge winning models for local clinical corpora … could lead to dramatic performance drop when training and test data are from different institutes." |
| ["The portability paradox of foundation models for clinical decision support," *npj Digital Medicine*](https://www.nature.com/articles/s41746-026-02615-4) | UNVERIFIED (2026, 1 host) | SNIPPET: "Limited positive signal during adaptation may bias the FM toward spurious or institution-specific patterns that appear predictive internally but vanish elsewhere." Verify DOI and authorship. |

---

## O. Annotation cost

The evidence for the marginal-cost argument.

| Claim | Source | SNIPPET |
|---|---|---|
| **A new concept family ≈ 1,000 annotated notes with multi-layer annotation** | Steinkamp et al., "Task definition, annotated dataset, and supervised NLP models for symptom extraction from unstructured clinical notes," *JBI*. [PubMed](https://pubmed.ncbi.nlm.nih.gov/31838210/) | **1,009 de-identified discharge summaries** annotated for NER + coreference + normalization — to support **one** concept family, "symptom" |
| **Porting an already-trained model to a new site costs hundreds of local notes, saturating ~700** | "Customize Deep Learning-based De-Identification Systems Using Local Clinical Notes — A Study of Sample Size." [medRxiv](https://www.medrxiv.org/content/10.1101/2020.08.09.20171231v1.full) | 1,100 notes across 39 note types; tested 100/300/500/700/900; **beyond ~700 notes improvement became marginal** |
| **The last few points of quality are where cost goes convex** | "Is the Juice Worth the Squeeze? Costs and Benefits of Multiple Human Annotators for Clinical Text De-identification." [PubMed](https://pubmed.ncbi.nlm.nih.gov/27405787/) | Median cost per PII instance **$0.71** for a single annotator vs **$377** for instances found only by a *fourth* annotator. A second annotator reached 0.99 recall at reasonable cost; benefits diminished beyond two |
| **Best-in-class annotation-cost reduction shaves ~20%, not 80%** | "Utilizing active learning strategies in machine-assisted annotation for clinical NER," *JAMIA* 31(11):2632, 2024. [OUP](https://academic.oup.com/jamia/article-abstract/31/11/2632/7724491) | At 99% target effectiveness, CNBSE needed **20.4% fewer edits** (22.5% for high-difficulty entities). Note the metric is *edits*, i.e. a shaving of a human-in-the-loop budget, not its elimination. Pre-annotation studies report **2.89–29.1%** per-entity savings ([IEEE](https://ieeexplore.ieee.org/document/6366101)) |
| **Long-tail attributes stay starved after a deliberate active-learning campaign** | "Leveraging deep active learning to identify low-resource mobility functioning information in public clinical notes." [arXiv 2311.15946](https://arxiv.org/abs/2311.15946) | Final dataset **4,265 sentences / 11,784 entities**: 5,511 Action, 5,328 Mobility, **306 Assistance, 639 Quantification**. IAA 0.72 exact / 0.91 partial. **Rare attributes are rare in the source text, so cost-per-positive-example rises without bound exactly where coverage is most needed** |
| **The phenotyping field states the scaling problem as settled** | PheNorm; MAP (§G) | "severely limits"; **"infeasible for highly multi-phenotype applications … requiring de novo labeling of hundreds to thousands of phenotypes"** |
| **Two orders of magnitude more pretraining does not eliminate per-variable labels** | GatorTron (§B) | >90B words / 290M notes, still fine-tunes per task, F1 0.90 / 0.81 / 0.90 |
| Time-of-annotation factors | ["Clinical text annotation – what factors are associated with the cost of time?"](https://pmc.ncbi.nlm.nih.gov/articles/PMC6371268/) | 8 time-associated factors; framed as input to active-learning cost models |
| Sample size for fine-tuning | Majdik, Graham et al., *JMIR AI* 2024, DOI 10.2196/52095. [JMIR](https://ai.jmir.org/2024/1/e52095) | 490 annotated documents, 2,500 stratified subsamples. "Relatively modest sample sizes can be used"; **training-set entity density should approximate production entity density.** ⚠ The documents are conflict-of-interest disclosures, **not clinical notes** — cite for methodology, not as a clinical result |

**The counter-claim to engage.** Med-BERT reports that pretraining let **300–500 samples** match a
**10× larger** set without pretraining (AUC gains 1.21–6.14%). The UMN↔Mayo study concludes BERT
models port "requiring minimal effort as only a small amount of annotated data is needed."
**The rebuttal is in those same sources:** Med-BERT's labels are structured outcomes, nearly free,
not chart-abstracted variables; and the breast-cancer study's own correlation coefficients
(−0.678 / −0.345 / −0.712) say transfer degrades precisely as target concepts become less similar.

**Head-to-head, encoder vs LLM — the split is itself the finding.**

| Study | Result |
|---|---|
| ["Improving large language models for clinical NER via prompt engineering," *JAMIA* 31(9):1812, 2024](https://academic.oup.com/jamia/article/31/9/1812/7590607) | Relaxed F1: GPT-3.5 0.794 / GPT-4 **0.861** vs **BioClinicalBERT 0.901** (MTSamples); 0.676 / 0.736 vs **0.802** (VAERS). **Fine-tuned encoder wins on both** |
| ["Zero-shot inference with LLMs vs supervised modeling in breast cancer pathology classification"](https://pmc.ncbi.nlm.nih.gov/articles/PMC10889046/) | Across 13 classification tasks, GPT-4 "significantly better than or as well as the best supervised model," average macro F1 **0.83** |
| ["Zero-shot extraction of seizure outcomes from clinical notes using GPTs"](https://link.springer.com/article/10.1007/s41666-025-00198-5) | **"In sparse contexts,"** best GPT **76%** vs fine-tuned BERT **67%** |
| Agrawal, Hegselmann, Lang, Kim & Sontag, "Large language models are few-shot clinical information extractors," EMNLP 2022. [ACL](https://aclanthology.org/2022.emnlp-main.130/) | The canonical opening citation for the pivot; names the two roadblocks as **"dataset shift from the general domain and a lack of public clinical corpora and annotations."** |
| ["Efficiency at scale: diminutive language models in clinical tasks"](https://www.sciencedirect.com/science/article/pii/S0933365724002446) | The honest steelman: "if the use of LLMs is prohibitive, simpler supervised models with large annotated datasets can provide comparable results" |

**Pattern:** span-level NER in dense context → fine-tuned encoders still win; document-level
classification and sparse contexts → LLMs win or tie. **Neither camp benchmarks cross-document
variables.**

---

## P. Cross-custodian records and patient matching

The clinical term of art for our cross-boundary case is **outside medical records (OMR)**.

| Work | Status | Note |
|---|---|---|
| GAO-19-197, "Health Information Technology: Approaches and Challenges to Electronically Matching Patients' Records across Providers." [GAO](https://www.gao.gov/products/gao-19-197) | Authoritative | The policy citation for identity resolution across custodians. |
| ["Salience of Medical Concepts of Inside Clinical Texts and Outside Medical Records for Referred Cardiovascular Patients"](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8982748/) | UNVERIFIED (1 host) | SNIPPET: OMRs arrive as faxes from external providers, "machine-illegible"; referred patients "can have a massive amount of OMRs"; "expert manual review to identify recent and relevant information demands significant resources … with significant chances for errors or lapses." |
| [Pew, hospital/clinic executives on record exchange](https://www.pew.org/en/research-and-analysis/issue-briefs/2019/05/hospital-and-clinic-executives-see-rising-demand-for-accurate-exchange-of-patient-records) · [unmatched records in an HIE](https://pmc.ncbi.nlm.nih.gov/articles/PMC4941843/) · [EHR linkage module across neighboring centers](https://pubmed.ncbi.nlm.nih.gov/33147645/) · [data-adaptive Fellegi-Sunter](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9562057/) · [probabilistic linkage accuracy across sociodemographic categories, *JMIR Form Res* 2026](https://formative.jmir.org/2026/1/e78622) · [AHRQ, managing patient identity across data sources](https://www.ncbi.nlm.nih.gov/books/NBK208618/) | mixed | ⚠ SNIPPET figures circulating in this cluster — "up to half of records may not be correctly linked," "~33% of denied claims linked to patient identification errors, >US$6bn annually" — **could not be attributed to a specific page. Do not use as-is.** Probabilistic matching accuracy is **not uniform across demographics.** |

**The distinction that decides whether cross-system work is on our axis at all.** A health system with
a master patient index and an enumerable list of subsystems is "big but enumerable" — a fixed
retrieval policy suffices, and "query every system with the local MRN" is a constructible control
arm. The genuinely non-enumerable cases are those where **identity or system membership is itself
chart-derived**: outside records and HIE, pre-migration historical records, scanned documents with no
structured identity.

---

## Q. Surveys and systematic reviews

| Work | Status | Note |
|---|---|---|
| Wang et al., "Clinical information extraction applications: A literature review," *JBI* 77:34, 2018. [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1532046417302563) | VERIFIED (5 hosts) | 263 studies, 61 using ML. ⚠ **The snippet gave methods counts, not the task taxonomy. Do not cite this paper for a task list without opening it.** The "88 unique diseases / portability limited" attribution came from a **secondary paraphrase** — verify. |
| "A survey of NLP methods for oncology in the past decade with a focus on cancer registry applications," *Artificial Intelligence Review* 2025, DOI 10.1007/s10462-025-11316-5. [Springer](https://link.springer.com/article/10.1007/s10462-025-11316-5) | VERIFIED-ish; **authorship UNVERIFIED** | 156 articles, 2014–2024, categorized by method, **document type**, cancer site, research aim. Methods: rule-based 70, ML 66, traditional DL 70, transformers 29. Stated gaps: pediatric cancers, melanoma, lymphoma underrepresented; **"disease progression"** and trial matching underrepresented as aims. **The document-type breakdown is likely the single best table for our central claim.** |
| Dahl, Bøgsted, Sagi & Vesteghem, "Performance of NLP for Information Extraction From EHRs Within Cancer: Systematic Review," *JMIR Med Inform* 2025;1:e68707. [JMIR](https://medinform.jmir.org/2025/1/e68707) | VERIFIED | 33 articles. **Bidirectional transformers outperformed every other category** (average F1 advantage 0.0439–0.2335). |
| "Using NLP to extract information from clinical text in EMRs for populating clinical registries: a systematic review," *JAMIA* 33(2):484. [OUP](https://academic.oup.com/jamia/article/33/2/484/8287208) | VERIFIED | The closest published analogue to automated chart review. SNIPPET: NER-then-normalization F1 **71.1 (lab tests), 89.3 (drugs)**; "the diversity of clinical text and extracted data elements posed challenges to the generalizability"; "the performance of the NLP methods **varied significantly**." |
| ["Automatic Classification of Cancer Pathology Reports: A Systematic Review"](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8860734/) · ["Clinical NER and RE using NLP of medical free text: A systematic review"](https://www.sciencedirect.com/science/article/pii/S1386505623001405) · ["Clinical concept extraction: A methodology review"](https://www.sciencedirect.com/science/article/pii/S1532046420301544) · ["Coreference resolution: A review"](https://www.sciencedirect.com/science/article/pii/S153204641100133X) · ["Clinical Information Retrieval: A Literature Review"](https://link.springer.com/article/10.1007/s41666-024-00159-4) | mixed | — |
| NASEM, *Enabling 21st Century Applications for Cancer Surveillance Through Enhanced Registries and Beyond*. [NCBI Bookshelf NBK614527](https://www.ncbi.nlm.nih.gov/books/NBK614527/) | **UNREAD** | Likely contains policy-level statements on automation limits. |

---

## R. Integrity flags — resolve before any of this reaches a bibliography

Each of these becomes two rows by accident if nobody is watching.

1. **DeepPhe-CR has two PubMed IDs and two PMC IDs.** PMID 38113411 / PMC10752457 is the *JCO CCI*
   published version (DOI 10.1200/CCI.23.00156); PMID 37205575 / PMC10187451 / medRxiv
   2023.05.05.23289524 is the preprint. **Same work in two states. Cite the JCO CCI version.**
2. **arXiv 2506.15301 appears under two titles.** The HTML v1 is *"Cohort Discovery: A Survey on
   LLM-Assisted Clinical Trial Recruitment"*; the abs/PDF and the ACL Anthology version are
   *"A Survey on LLM-Assisted Clinical Trial Recruitment."* **Cite the ACL Anthology version as the
   version of record.**
3. **arXiv 2604.05190 was returned under two different titles** — *"Retrieval-Augmented LLMs for
   Evidence Localization in Clinical Trial Recruitment from Longitudinal EHR Narratives"* and
   *"Improving Clinical Trial Recruitment using Clinical Narratives and Large Language Models."*
   **Do not cite either title until the PDF is opened.**
4. **medRxiv 2025.10.21.25338475 returned three different titles** across v1/v2/full-PDF under one
   DOI, including *"Comprehensive Structured Abstraction of Pathology Reports Is Now Feasible Using
   Local Large Language Models."* Same warning.
5. **αNeSy-CTM: the observed URL was arXiv 2606.20895; accompanying prose said 2606.20897.**
   One is wrong.
6. **GatorTron's medRxiv v2 title differs from v1.** Cite the *npj Digital Medicine* version.
7. **EHRNoteQA (arXiv 2402.16040) and EHRNote-ChatQA (arXiv 2606.15735) are different papers.**
   The names are confusingly similar.
8. **Two different March-2026 medRxiv items** on LLM registry abstraction — the longitudinal breast
   oncology study and the Brim/DeepPhe comparison. **Do not merge.**
9. **Unattributed figures — do not use as-is:** the ~6% external-validation drop in the portability
   cluster; the pathology T/N/M staging figures 0.88/0.90/0.24; the annotation-time figures in the
   dictionary-pre-annotation cluster; the learning-curve plateau figure; the HIE "half of records"
   and "$6bn" numbers; FHIR-AgentBench's 58%→71%; the Carrell 92%/88%.
10. **Unconfirmed leads, no primary URL:** TrialMatchAI; OncoLLM as a standalone entity;
    **CLAMP being unmaintained and replaced by an LLM system called "Kiwi"** — the last would be
    strong narrative material and needs a primary source before use.

### Negative results — stated narrowly, and what would falsify them

Each is a negative result from roughly 200 searches, not a proof.

* **No published system follows a pointer from one document to another within a patient record.**
  Nothing in the clinical literature names this. ACIE claims "cross-document dependencies"
  generically; FHIR-AgentBench reports multi-hop failures as a finding. *Falsified by:* ACIE's full
  text, or AgentEHR.
* **No prior art measures whether a reviewer consulted the document that would have changed the
  answer.** Registry, hospital and trial QA all measure value agreement; the unit is always the
  field, never the evidence path. *Falsified by:* NAACCR Volume III's process standards, the Nahm
  dissertation, or Kaji's ten bias layers.
* **No published error taxonomy for chart abstraction separates missed-document from
  misinterpretation from rule-misapplication.** The one medical taxonomy that makes this cut is
  Kundel's, and it is about pixels and eye fixations. *Falsified by:* the Nahm dissertation.
* **Abstention exists as a label but not as a measurement.** TrialGPT and others carry
  `not enough information`; no per-class performance for it was found, and the field's main
  cohort-selection benchmark is binary. *Falsified by:* the TrialGPT Nature Communications
  supplement.
* **The enumerable-vs-bridge axis has no name in clinical NLP.** *Falsified by:* anything in the
  clinical IR literature not reached in ~30 targeted queries.

---

## S. Read these five before publishing

Every one is blocked from the compiling environment. Each changes what can be claimed.

1. **ACIE, arXiv 2606.19602.** Two questions only: does it contain a single-pass ablation, and does
   its 96.5% separate "found the evidence" from "got the answer"? Both answers change the
   positioning section.
2. **DeepPhe-CR full text, *JCO CCI*.** Confirm the digits **95-98**, the presence of the word
   **"informally,"** and the per-variable F1 table behind the 0.79–1.00 range.
3. ***JNCI Monographs* 2024;(65):123.** Confirm the consolidated-tumor-case element list excludes
   stage and first course of treatment. This is the cleanest structural argument available and it
   currently rests on a snippet inference.
4. **NAACCR Volume III process standards.** The most likely place a partial precedent for
   process-quality measurement is hiding.
5. **Gilbert et al. 1996 (the eight criteria verbatim) and Kaji et al. 2014 (Figure 1, the ten bias
   layers).** Confirm neither contains an evidence-retrieval layer.

Also worth obtaining: the *AI Review* oncology survey's document-type × research-aim table; the Nahm
dissertation's error categories; and **a cancer-registrar reabstraction study with per-variable
agreement** — the one hole this sweep could not close.
