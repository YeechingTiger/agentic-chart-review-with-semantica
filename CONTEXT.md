# Agentic Chart Review

An agent reads a patient's record against a formal extraction contract and returns one answer, or a
justified refusal to answer. The framework is general — cancer-registry abstraction is the first use
case, not the boundary — so nothing here names a disease, an organ or a coding system.

## Language

### The question being asked

**Task Contract**:
The frozen statement of one extraction question: its fields, its decision rules, its evidence rules,
its conflict rules, and what must be true before an answer stands.
_Avoid_: spec (used in code and filenames; fine there, imprecise in discussion), schema, template

**Field**:
One output the Task Contract asks for. A contract may declare several, each with its own standing.

**Decision Rule**:
A clause of the Task Contract that selects among admissible answers. Numbered, and a run may cite
the number.

**Conflict Rule**:
A Decision Rule of the form *if two sources disagree in this specific way, take this one*. Every one
of them names a **Discriminating Fact**.

**Discriminating Fact**:
The thing a Conflict Rule turns on — the fact whose presence or absence selects one competing answer
over another. Example: *whether a physician's clinical impression accompanies an ambiguous cytology
on its own date*.

### What a document is worth

**Standing**:
What one document is worth for one Field, judged against the contract's evidence rules. Three
values, and the middle one must never be folded into the first.
- **can establish** — this document satisfies the evidence rules, rendered here by whoever was
  entitled to render it
- **merely mentions** — it bears on the question (restates, carries forward, refers to, plans
  around, argues against) but cannot settle it
- **neither** — it does not bear on the question
_Avoid_: relevance, admissibility (used in code; "standing" is the domain word)

**Answer-Bearing Note**:
A document whose Standing for a Field is *can establish*. The denominator for anything called a
yield.

### What a run produces

**Answer**:
A value for each Field, or a declared abstention. Which abstention matters: "no document in this
chart establishes it" and "this chart contains no document from the relevant period" are different
claims and the contract declares both.

**Complete Answer**:
An answer where the run went and looked for the thing that would have changed it — that is, checked
the **Discriminating Fact** of every Conflict Rule it was subject to.
Completeness is independent of correctness: an answer can be right and incomplete (arrived at by a
shortcut that happens to work on this chart), and that is the case worth catching, because the same
shortcut is wrong on the mirror chart.
_Avoid_: thorough, exhaustive, high-confidence, well-covered

**Coverage**:
The separate and much stronger claim that enough of the record was reviewed for an ABSENCE to stand.
Not the same as completeness: coverage is about how much was read, completeness is about whether the
one thing that mattered was looked for.
_Avoid_: using "coverage" loosely to mean completeness

**Warrant**:
The cited evidence and rule references a run offers for its answer. A warrant can be articulate and
false — a run may state that a Discriminating Fact is absent having never searched for it.

### What varies between runs

**Arm**:
One configuration of the system, held fixed while the patient varies. Two runs are the same arm when
their recorded configuration hashes alike.

**Retrieval Prior**:
A measurement, taken over a development set, of where a Field's answer tends to live: which document
types can establish it and at what rate, and which terms surface those documents at what cost. It is
data offered as reference, never a rule and never a filter.
_Avoid_: experience, knowledge, keyword list (each names a part of it and reads as the whole)

**Method Card**:
Prose guidance assembled into the prompt, which the agent may depart from. It changes what the model
is told, never what the runtime enforces.
_Avoid_: policy, tactic, controller, skill — see Flagged ambiguities

## Relationships

- A **Task Contract** declares many **Conflict Rules**; each names one **Discriminating Fact**
- A document has one **Standing** per **Field**, not one per document
- A **Complete Answer** has checked every **Discriminating Fact** it was subject to; a correct
  answer need not have
- **Coverage** is a claim about the record; **completeness** is a claim about the reasoning
- A **Retrieval Prior** is measured over **Answer-Bearing Notes**, so its denominator is a Standing
  judgement, not a document count

## Example dialogue

> **Dev:** The run answered `20230427` and cited the cytology conflict rule. Gold is `20230412`. Is
> that a retrieval failure?
>
> **Domain expert:** No. Both documents it needed to compare were in front of it. It said "no
> physician clinical impression is documented at that date" — but one is, in an oncology note it
> never opened. It didn't fail to find the answer; it failed to look for the thing that would have
> changed the answer.
>
> **Dev:** But on the mirror chart the same run answered correctly.
>
> **Domain expert:** With the same shortcut. Take the tissue date. That is right when nothing
> supports the earlier cytology and wrong when something does, and the run has no way to tell those
> apart because it never checks. The correct answer on that chart is not evidence of anything.

## Flagged ambiguities

- **"policy" vs "tactic"** — the code splits Method Cards into a `policy` slot (one per run, said to
  be the shape of the whole traversal) and a `tactic` slot (many, each with a precondition).
  Measured against 75 recorded runs across three policy arms, the traversal shape does not vary:
  every arm does one search-then-read round, median. And the cards' own descriptions cross the line
  in both directions — `policy-information-gain` describes choosing the next step "rather than what
  shape the whole traversal should have", while `tactic-coverage-pool` and
  `tactic-orient-from-summary` both describe traversal shapes. **Unresolved**: the slot names are
  retained for now, but the distinction they claim is not one this domain has been shown to have.

- **"experience" / "prior" / "keyword list" / "note-type list"** — four names in use for one
  artifact and its parts. Resolved: the artifact is a **Retrieval Prior**; the keyword half and the
  document-type half are two measurements inside it, not separate things.

- **"complete"** — was being used for at least three properties: correct, exhaustively read, and
  well-justified. Resolved: **Complete Answer** as defined above; "exhaustively read" is
  **Coverage**; "well-justified" is **Warrant**, which can be false.
