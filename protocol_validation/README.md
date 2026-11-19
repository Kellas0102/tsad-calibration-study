# Developer-led protocol-validation research extension — 2026-11-12

Read `RESEARCH_REPORT_ZH.md`, `PLAN.md`, and the proposed manuscript addition.
This directory does not replace the previous 25-page/14-page candidate.

Completed: 60 structured synthetic evidence challenges, with a frozen minimal rule
engine and separate oracle; ten purposively fixed public reports and 60 source-bound
field extractions. Negative outcomes are retained. This is neither independent expert
validation nor a systematic review or automated prose-understanding benchmark.

## Reproduce

```sh
python3 evaluate_challenges.py
python3 extract_public_cases.py
python3 validate_extension.py
```

Python, NumPy and pypdf are used. Challenge execution needs no network or models.
The engine was copied before challenge evaluation; freeze_manifest.json records its
source and hash. The original development examples/tests are retained separately in
frozen_engine/ and are not counted in the new 60 cases. Do not rerun the build script
over existing bundles: it refuses overwrite to preserve evaluation history.

Public quote verification additionally requires local_sources/ from this working
directory. Full PDFs and extracted full text are **not** redistributed in the small
research-output ZIP. sources.json and retrieval_followup.json preserve requested and
resolved URLs, failures, page counts and hashes; retrieval scripts explain acquisition.
Re-downloads may change bytes, especially unversioned sources, and must not silently
replace the analyzed version. Public tables use short quotations and page locators.

The generated validation report checks local consistency, not researcher agreement,
scientific-quality certification or author approval. All human verification remains
pending. The frozen rule engine's misses are findings, not reasons to fail the QA
script: QA checks that these known failures are faithfully retained.
