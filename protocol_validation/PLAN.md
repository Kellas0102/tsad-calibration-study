# Protocol validation extension: local plan before outcome extraction

Date: 2026-11-12. Status: developer-led, retrospective extension; not preregistered,
not independent expert validation, and not a systematic literature review.
The 25-page/14-page candidate remains untouched until new evidence is reconciled.

## A. Controlled challenge study

Freeze `diagnostics.claim_assessment` and `diagnose_claim.diagnose` from the latest
candidate before evaluation. Preserve hashes and existing examples/tests as
development material. New challenge records are separate files created after the
freeze; they are NOT statistically independent of the developers or unknown to the
authoring assistant. Do not call them an external test set.

The claim assessor contains four implemented contradiction predicates: PA represented
as strict F1; test-label oracle promoted to deployment calibration; future context
promoted to causal-online evidence; known artifact-hash mismatch. Construct evidence
bundles with measured hashes, split/time metadata, score arrays and metric labels.
Derive gate facts from those bundles, never feed expected verdicts to the assessor.
Freeze expected semantic labels separately before running. Report within-rule
contradictions, valid controls and incomplete bundles separately, including every
miss, unjustified definitive verdict and ambiguous result. Combinations of missing
and contradictory evidence are explicit stress tests, not silently discarded.

Add scope probes for pseudoreplication and non-significance promoted to equivalence:
these are NOT implemented contradiction predicates. Report them as coverage gaps,
not successes, and do not retrofit the frozen engine to improve evaluation outcomes.
Any repair belongs to a subsequent version with a transparent development history.
Literal document parsing or general natural-language reasoning is not evaluated.

The evidence-card interface consumes already-adjudicated gate statuses. It is tested
for aggregation/fail-closed behavior only; successful status propagation is not
independent detection of scientific faults. Keep this distinct from computed-bundle
checks and from mathematical score properties.

## B. Fixed public-case applicability panel

Purposive ten-report panel, selected from the candidate's already-cited literature
before detailed field extraction: four detector/method reports (TimesNet, MEMTO,
CATCH, OmniAnomaly), three evaluation reports (Kim rigorous evaluation, Huet
affiliation, PATE), three benchmark/toolkit reports (TSB-UAD, TimeEval, TAB).
No replacement based on whether the protocol reaches a favorable/critical verdict.
Unavailable reports remain in the retrieval denominator. This is not a representative
sample of TSAD, nor novel evidence independent of protocol development literature.

Read a pinned full report and relevant appendices, with a lawful primary/repository
source preferred. Record URL, retrieval time, file hash and PDF-page locator.
For each report extract: (1) metric/adjustment semantics; (2) threshold-selection
source or threshold integration; (3) temporal/event/aggregation scope;
(4) experimental replication/sampling unit; (5) code/artifact access pointer;
(6) operational-use evidence relevant to the quoted claim. Distinguish positive
explicit evidence, a bounded-search unresolved field, and not-applicable with reason.
Do not equate unavailable information with a false statement or failed study.

Each case contains a short source quotation and locator, a scope-bounded paraphrase,
applicable protocol gates and unresolved minimum evidence. Assess chosen statements,
not paper quality, author intent, or the truth of all empirical findings. Extraction
is AI-assisted single-reviewer and remains pending accountable author verification.
No inter-rater reliability, accuracy, clinical/production benefit or expert consensus.

Report retrieval denominators, field-level availability and non-applicability counts,
representative scope distinctions, and negative cases. If all complete-artifact gates
are unavailable in document-only evidence, retain that as a usability limitation
rather than engineering favorable acceptance scores.

## Outputs and integration gate

Preserve sources locally for audit, but do not redistribute full papers or extracted
full text without license review. A distributable evidence table uses short quotations,
locators, hashes, access links and limitations, not copied full reports.
Outputs: frozen-engine manifest; challenge bundles and oracle; results/confusion tables;
ten source records and extraction table; limitations and a concise proposed manuscript
addition. Only completed, checked outputs may be incorporated into a new candidate.
