# Revision analysis contract — frozen before the new result tables

Date: 2026-11-07. Stage: retrospective reanalysis and author working draft, not a preregistered study. Existing results and the first-round audit have already been seen. No confirmatory or prospective validation is claimed.

## Objective

Replace universal F1 ranking prohibitions with a claim-relative evidence protocol. Separate a scoring-function claim, a calibrated-pipeline claim, an event-retrieval claim, and a deployment-utility claim. A logical/metric check is not an editorial acceptance rule.

## Immutable inputs

1. Legacy `paper_pool_roster.csv`: 354 records, six detectors, 12 dataset/entity units. This is the common-model descriptive panel, not 354 independent datasets.
2. Incoming recent archive: 30 records, three protocol-adapted detectors, SMD machine-1-1 and the flattened SKAB stream. Kept separate from the legacy panel.
3. Legacy 313-row criterion cohort: retained as an audit/sensitivity cohort, not pooled with the common-model panel as independent observations.
4. Run metadata and score snapshots are read-only. Preserve the recorded scores_path directory identity and bind each selected file to SHA256; never select conflicting snapshots by basename alone.

## Analysis population and restrictions

- Pair model records only within the same panel, dataset/entity, and seed identifier. Verify identical test labels and lengths, and record whether validation labels also align. Shared seed labels alone do not prove a randomized paired design; all uncertainty statements are conditional on the archived construction.
- No merging recent and legacy model results into a single league table.
- HAI cropped data and MSL/SMAP selected channels remain those original scopes.
- SKAB event summaries, if computed, are flattened-stream diagnostics, never trial-level operational events. Training/test independence is unresolved without the raw file manifest; deployment and independent-test claims are withheld.
- No model retraining or raw-data loader execution in this revision stage.

## Questions and estimands

RQ1: At a fixed unit and frozen operating point, what distinct event coverage and in-event recall can point-wise F1 conceal?

RQ2: Does the sign of a within-unit model contrast change across a declared common family of validation-calibration policies? The target is policy sensitivity, not an oracle-selected deployed winner.

RQ3: How much of a same-unit observed F1 contrast is algebraically allocated to coverage, within-covered-event recall, and normal-point false positives? Signed symmetric allocation must exactly sum to the observed contrast. It is descriptive, not causal.

RQ4: Do the executable checks satisfy their mathematical invariances and detect deliberately constructed protocol faults while leaving known-valid counterexamples unflagged? This is a specification/property test, not evidence of independent expert agreement or clinical/industrial effectiveness.

## Calibration family (fixed for this reanalysis)

K=4 policies, computed separately for every detector/run using validation scores only:

1. `val_f1`: exact F1 maximization on the archived synthetic-labelled validation split; score >= threshold; choose highest threshold among tied maxima.
2. `normal_005`: validation-normal upper order statistic targeting 0.5% normal exceedance.
3. `normal_010`: same, target 1%.
4. `normal_050`: same, target 5%.

For the latter three, m validation normal points are sorted, q is order statistic ceil((1-alpha)m), and test positives satisfy score > q. This conservative tie policy bounds validation-normal exceedance, not future/test FPR. It is invariant to a common strictly increasing transformation of each run's validation and test scores. It is not an estimated probability of anomaly.

The originally archived threshold (>=) is an additional provenance/check column and is not a fifth optimized policy. Test-label oracle F1 is obtained from all distinct test-score cutoffs and marked retrospective throughout.

## Metrics and edges

Raw TP/FP/FN/TN; strict F1; ROC-AUC and tie-block AP; prevalence; SegR; anomalous-point coverage c; r_in conditional on covered anomalous mass; normal-point FPR; rho=FP/TP when TP>0.

If no true event exists, event recall and coverage are unavailable. If no event is touched, c=0 and conditional r_in is undefined; an explicit zero-extension is used only inside the algebra. Never present an undefined conditional recall as a measured zero.

True and predicted events are maximal half-open contiguous positive intervals within known boundaries. Event F1 uses maximum-cardinality one-to-one overlap matching, delta=0; repeat alarms are penalized. It is one declared event convention, not the unique correct metric. Official affiliation implementation is used only after source inspection and reference-case tests. No legacy proxy retains the affiliation label.

## Attribution

At each same-unit, same-seed model contrast, use all six orders of replacement of (c,r_in,FPR), with p fixed because the unit and labels are identical. Average the signed increments per channel (three-variable Shapley allocation). Assert closure to observed F1 difference. Report absolute allocations separately from signed effects; normalized absolute shares are compositional descriptions, not a percent of a causal or net gap. Hybrid channel values are algebraic, not guaranteed feasible threshold interventions.

## Statistical reporting

- Lead with within-unit seed differences, sign counts and SD. Seed resampling does not resample real fault events or sites.
- Report percentile bootstrap intervals (B=10,000, seed fixed by deterministic contrast identifier); these are descriptive and can under-cover with few seeds. Do not infer significance from them.
- Exact two-sided sign tests address sign consistency conditional on independent pairs, omit exact-zero differences, and Holm-adjust the full model-pair × K policy family separately per panel/unit. No seed-count universal prohibition and no t-halfwidth named minimum detectable effect.
- An interval simultaneously covering all K policy means is constructed by a conservative union of bounded-variable Hoeffding intervals, assuming independent seed pairs and differences in [-1,1]. State that this is a sufficient, often conservative certificate, not necessary evidence. Use alpha=.05, practical margin delta=0 as an explicit mathematical reference only, not an industrial utility threshold.
- Calibration sign reversal is descriptive when min_k mean difference <0<max_k mean difference. Do not call it statistically established reversal without simultaneous support.
- Dataset/entity units receive equal weight in any cross-unit descriptive summaries; also report each unit. Never treat seeds, model pairs, and policy points as independent datasets.

## Required diagnostics before prose

Snapshot and cohort integrity; frozen-threshold and validation re-selection agreement; identity and attribution closure; positive monotone rescaling invariance; conservative validation-budget compliance; exact oracle upper-envelope check; explicit ties/undefined states; event metric reference fixtures; all models and all K policy outcomes retained.

## Withheld until human/extra evidence

Official-implementation fidelity, SKAB raw-data independence, corrected source-trial boundaries, HAI timestamp alignment, causal/online delay, independent-rater agreement, optimal detector choice, real cost ratios, updated Zenodo publication and submission readiness. No wording may imply these have been demonstrated.

## Transparent implementation amendment after first run

The first reanalysis found one run whose affine transform 17*s+5 coalesced distinct near-zero scores in float64, changing predictions. This is retained as a numerical failure boundary, not silently discarded. The metamorphic audit now records whether each proposed transformation is strictly order-preserving on the observed unique validation+test scores. Invariance is asserted only for admissible transformations, and every finite-precision collapse remains listed separately. The data and policy family are unchanged. This amendment is post hoc and is not described as preregistered.

After the zero-crossing envelope results were inspected, a post hoc sensitivity panel was added for symmetric reference magnitudes 0.005, 0.01 and 0.02 F1. A material descriptive reversal requires a mean below -delta and above +delta among the same four policies. All margins are reported, no policy or run is removed, and none of these values is claimed to be an operator-elicited utility threshold. The original zero-margin result remains visible.

## Final-round secondary analyses (retrospective; outcomes not yet computed)

1. Recompute descriptive reversal counts using only the three validation-normal policies. Retain the original four-policy analysis and all units. This separates sensitivity within a normal-budget family from changing between synthetic-positive and known-normal calibration objectives.
2. For all thirty HAI legacy runs, retain all four policies and six deterministic offline gap-closing widths g={0,1,5,10,30,60} points. Fill only internal negative gaps of length <=g between predicted positive intervals. This is retrospective interval consolidation, not causal cooldown or an operational recommendation. Report point F1, one-to-one event precision/recall/F1 and test normal-point FPR. Quantify unmatched overlapping predictions separately from predictions with no true-event overlap; do not attribute every event false positive to duplicated alarms.

The purpose is to check finite-policy dependence and alarm-counting interpretation. These are transparent post hoc sensitivity analyses on existing scores, not prospective tests or additional model runs. No width is selected as an optimized test policy, and all outcomes are retained.
