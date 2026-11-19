"""Source-bound, single-assistant extraction; no paper-quality scoring.

Each quoted string is checked against the named PDF page after NFKC/whitespace
and line-wrap hyphen normalization. This is a source locator check, not human verification.
normalization. Absent evidence is bounded to these reports, not their entire projects.
"""
from pathlib import Path
import collections
import csv
import hashlib
import json
import re
import unicodedata

ROOT = Path(__file__).resolve().parent
FIELDS = ('metric', 'calibration', 'temporal', 'replication', 'artifact_pointer', 'utility')


def norm(text):
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'(?<=\w)-\s+(?=\w)', '', text)
    return ' '.join(text.split())


def field(status, page, quote, observation):
    return dict(status=status, pdf_page=page, quote=quote, observation=observation)


# Extraction assembled from the named report sections; not an automated NLP result.
CASES = {
 'timesnet': {
  'claim_page': 8, 'claim_quote': 'TimesNet still achieves the best performance in anomaly detection',
  'scope': 'Table 5 benchmark F1 comparison; strict-vs-adjusted semantics and cutoff provenance need implementation evidence.',
  'metric': field('partial', 13, 'For anomaly detection, we adopt the F1-score', 'F1 named; PA/strict distinction unresolved in report-only evidence.'),
  'calibration': field('unresolved', None, '', 'Searched Sec.4.5, Appendix A and report-wide threshold/adjustment terms; operational cutoff source not located. Citation to preprocessing is not a fully specified cutoff.'),
  'temporal': field('explicit', 8, 'consecutive non-overlapping segments', 'Window segmentation stated; not proof of a causal deployment timeline.'),
  'replication': field('explicit', 12, 'ments are repeated three times', 'Three repetitions stated across experiments; not three independent sites. Quote starts after line-broken experi-.'),
  'artifact_pointer': field('explicit', 1, 'https://github.com/thuml/TimesNet', 'Repository pointer present; exact table-producing snapshots not authenticated here.'),
 },
 'memto': {
  'claim_page': 7, 'claim_quote': 'from 93.62% to 95.74%',
  'scope': 'Reported benchmark improvement is under explicitly adjusted metrics, not strict pointwise F1 or operator utility.',
  'metric': field('explicit', 7, 'Here point adjustment method is used', 'Positive disclosure of adjusted metric; no assertion that the paper mislabeled it as strict.'),
  'calibration': field('explicit', 14, 'both the training and validation data', 'Appendix A describes top-p% scores combined from training/validation. This does not establish identical semantics of later pinned source-code exports.'),
  'temporal': field('explicit', 7, 'a non-overlapped sliding window', 'Length100 and 80/20 training-validation split stated in surrounding paragraph.'),
  'replication': field('explicit', 15, 'mean and standard deviation over 10 runs', 'Table8 reports 10-run mean/SD and t-tests; pairing and sampling independence require further evidence for inferential promotion.'),
  'artifact_pointer': field('unresolved', None, '', 'Report-wide search for github, code availability and repository pointer did not identify an explicit model-code URL in this PDF. Proceedings supplemental package and known repository remain separate access evidence, not absence of code.'),
 },
 'catch': {
  'claim_page': 8, 'claim_quote': 'CATCH has the high- est AUC-ROC values on most datasets',
  'scope': 'AUC/affiliation benchmark comparisons have different semantics; no deployed threshold or alert-cost benefit follows from AUC alone.',
  'metric': field('explicit', 8, 'Affiliated-F1- score', 'Affiliation-F1 and ROC separately named; Appendix A.2 enumerates 16 metrics.'),
  'calibration': field('partial', 8, 'under various pre-selected thresholds', 'AUC integrates operating points, but exact label-based Aff-F cutoff selection not located in Sec4 or A.2-A.4/report-wide threshold search.'),
  'temporal': field('explicit', 8, 'point-granularity alignment', 'Sec3.4 and A.6 specify overlapping-patch score mapping; online causality not inferred from alignment.'),
  'replication': field('partial', 9, 'Average A-R (AUC-ROC) and Aff-F', 'Table2 aggregate means identifiable; training-seed repetitions/uncertainty not located using report-wide seed/repeat/deviation searches.'),
  'artifact_pointer': field('explicit', 17, 'https://github.com/decisionintelligence/CATCH', 'Pointer present; original table-specific model and score hashes not checked.'),
 },
 'omnianomaly': {
  'claim_page': 8, 'claim_quote': 'F1 obtained through POT vs.',
  'scope': 'Table3 distinguishes POT F1 from retrospective best F1; both are evaluated with disclosed point adjustment.',
  'metric': field('explicit', 6, 'We adopt the point-adjust way', 'Adjustment explicitly disclosed, not evidence of strict metric mislabeling.'),
  'calibration': field('explicit', 6, 'enumerate all possible anomaly thresholds', 'Sec5.1 separates best-F1 from method-specific thresholds; Sec4.4 describes training-score POT.'),
  'temporal': field('explicit', 5, 'consecutive observations preceding to it', 'Input for observation t includes preceding observations; full production causality would additionally require preprocessing audit.'),
  'replication': field('partial', 6, 'union of these three datasets', 'Aggregate dataset population stated; independent run repetitions/uncertainty not located in Sec5, Appendix B or seed/run searches.'),
  'artifact_pointer': field('explicit', 3, 'https://github.com/smallcowbaby/OmniAnomaly', 'Historical repository pointer documented; current rename/redirect not proof of original artifacts.'),
 },
 'rigorous': {
  'claim_page': 7, 'claim_quote': 'PA can severely overestimate',
  'scope': 'Counterexample and selected-benchmark evaluation claim; oracle cutoff is disclosed, not promoted here to deployable calibration.',
  'metric': field('explicit', 1, 'F1 score computed without', 'First page distinguishes PA F1 and unadjusted F1; surrounding text defines label expansion.'),
  'calibration': field('explicit', 6, 'All thresholds were obtained from those that yielded the best score.', 'Retrospective best-score threshold explicitly stated.'),
  'temporal': field('explicit', 7, 'window lengths ranging from', 'Sec5.4 evaluates window-length sensitivity; not a chronological-drift intervention.'),
  'replication': field('explicit', 6, 'ent seeds and reported the average values', 'Five different seeds stated in line-broken sentence for randomized Cases1/3; not all baselines uniformly independently rerun.'),
  'artifact_pointer': field('unresolved', None, '', 'Report-wide github/code availability searches did not locate a URL for the authors own baseline in the retrieved conference PDF; officially provided baseline codes are mentioned. Supplement not retrieved as a separate source in this panel.'),
 },
 'affiliation': {
  'claim_page': 6, 'claim_quote': 'classical and the affiliation metrics are not sensitive to the adversary algorithm',
  'scope': 'Robustness claim concerns the explicitly constructed adversary; not every adversarial strategy or a detector ranking in production.',
  'metric': field('explicit', 3, 'precision/recall pair', 'Affiliation distances/probabilities defined in Sec3; distinct from one-to-one event F1.'),
  'calibration': field('not_applicable', 3, 'Both prediction and ground truth are represented', 'Metric maps supplied event sets to scores; no detector cutoff learned for this constructed-adversary claim.'),
  'temporal': field('explicit', 3, 'set of disjoint anomalous events', 'Continuous intervals and local ground-truth zones explicitly defined.'),
  'replication': field('explicit', 6, 'select the four time series datasets', 'Four-case metric/adversary illustration with analytical derivation, not a repeated-training population study.'),
  'artifact_pointer': field('explicit', 11, 'available at [9]', 'Appendix D points to library; reference9 contains Figshare DOI. Pointer only, not original prediction snapshots.'),
 },
 'pate': {
  'claim_page': 5, 'claim_quote': 'distinguishes the scenarios',
  'scope': 'Synthetic detector-response discrimination under declared buffer/threshold integration; not a ground-truth accuracy assessment of all metrics.',
  'metric': field('explicit', 5, 'compute the area under the curve', 'Weighted PR area averaged over pre/post-buffer settings, not fixed-cutoff F1.'),
  'calibration': field('explicit', 5, 'we apply a range of thresholds', 'Threshold integration stated; no deployed cutoff implied. Binary PATE-F1 extension has a different target.'),
  'temporal': field('explicit', 5, 'pre-buffer', 'Sec2.3 and constructed scenarios distinguish early/delayed timing and coverage; buffer preference is not measured operator utility.'),
  'replication': field('explicit', 5, 'ten different detection scenarios', 'Ten constructed scenarios for selected claim; not ten independently sampled experts/datasets.'),
  'artifact_pointer': field('explicit', 10, 'https://github.com/Raminghorbanii/PATE', 'Appendix A source pointer; not proof of independent reproduction.'),
 },
 'tsbuad': {
  'claim_page': 1, 'claim_quote': 'reproducible, and frequently updated resource',
  'scope': 'Benchmark resource and documented evaluation framework, not an independent verification of all supplied detector scores.',
  'metric': field('explicit', 7, 'Precision, Recall, and F-score', 'Point/range measures, ROC area and precision@k distinguished in Sec4.4.'),
  'calibration': field('explicit', 7, 'vary the threshold from the highest', 'AUC integration explicit; other fixed-cutoff metrics still have separate parameter rules.'),
  'temporal': field('explicit', 7, 'fragmented prediction regions', 'Range metrics account for fragmentation; transformation/mixed-source datasets do not imply chronological drift.'),
  'replication': field('explicit', 8, 'evaluate pairs of methods over multiple datasets', 'Cross-dataset Wilcoxon/Friedman/Nemenyi design identified; perturbation SD is not seed SD.'),
  'artifact_pointer': field('explicit', 1, 'www.timeseries.org/TSB-UAD', 'Data/code access pointer, not exact run/score identity verification.'),
 },
 'timeeval': {
  'claim_page': 3, 'claim_quote': 'calculates the quality metrics, and stores all results',
  'scope': 'Toolkit orchestration/measurement capability; demonstration paper alone does not validate a detector-superiority or review-efficiency claim.',
  'metric': field('explicit', 3, 'AUC-ROC', 'ROC/PR and range-based PR area supported; runtime separate from predictive quality.'),
  'calibration': field('explicit', 3, 'threshold-agnostic evaluation measures', 'Integrated metrics do not specify an operational decision threshold.'),
  'temporal': field('explicit', 2, 'training and testing time series', 'Generator supports explicit train/test scenarios and anomalies; algorithm-specific causal preprocessing not inferred.'),
  'replication': field('not_applicable', None, '', 'Selected statement concerns toolkit workflow, not a repeated-trial effect estimate; separate comprehensive evaluation is a different report.'),
  'artifact_pointer': field('explicit', 1, 'https://github.com/hpi-information-systems/timeeval-gui', 'Artifact pointer explicit in publisher PDF; toolkit availability not frozen-result authenticity.'),
 },
 'tab': {
  'claim_page': 7, 'claim_quote': 'TAB can ensure a fair com- parison',
  'scope': 'Unified pipeline addresses several inconsistencies, but best-over-threshold retrospective results are not one prechosen operational policy.',
  'metric': field('explicit', 8, 'abandon drop-last and point adjustment', 'Explicit rejection of PA; Table5 distinguishes label/score metrics.'),
  'calibration': field('explicit', 8, 'metric calculations at all thresholds and report the best results', 'Finite percentage-threshold sweep reported. Do not conflate with common deployable validation-only calibration.'),
  'temporal': field('explicit', 8, 'non-overlapping window rolling approach', 'Output aggregation stated; common 50/50 and validation suffix policy described nearby.'),
  'replication': field('partial', 8, '29 multivariate datasets and 1,635 univariate time series', 'Dataset population clear; independent repeated training/uncertainty not located in Sec5/report-wide repeat/seed/deviation search.'),
  'artifact_pointer': field('explicit', 8, 'https://github.com/decisionintelligence/TAB', 'Pointer present; this case study does not independently reproduce all model tables.'),
 },
}


def main():
    records = json.loads((ROOT/'sources.json').read_text())
    records += json.loads((ROOT/'retrieval_followup.json').read_text())
    sources = {r['case_id']: r for r in records if r['status'] == 'retrieved'}
    assert set(CASES) == set(sources)
    rows, summaries = [], []
    for key, case in CASES.items():
        source = sources[key]
        assert hashlib.sha256((ROOT/source['local_pdf']).read_bytes()).hexdigest() == source['pdf_sha256']
        parts = re.split(r'=== PDF PAGE (\d+) ===', (ROOT/source['local_text']).read_text())
        pages = {int(parts[i]): norm(parts[i+1]) for i in range(1, len(parts), 2)}
        assert norm(case['claim_quote']) in pages[case['claim_page']], (key, 'claim quote')
        # No production-benefit field is inferred from benchmark or metric evidence.
        case['utility'] = field('not_applicable', None, '', 'Selected statement does not estimate deployment costs, workload or intervention benefit; G8 is not validated by this case.')
        word_count = len(case['claim_quote'].split())
        for name in FIELDS:
            record = case[name]
            if record['quote']:
                assert norm(record['quote']) in pages[record['pdf_page']], (key, name, record['quote'])
                word_count += len(record['quote'].split())
            rows.append(dict(case_id=key, group=source['group'], field=name, **record,
                             source_sha256=source['pdf_sha256'], human_verification='pending',
                             extractor='single AI assistant; no inter-rater study'))
        assert word_count <= 180, (key, word_count)
        summaries.append(dict(case_id=key, title=source['expected_title'], group=source['group'],
                              claim_page=case['claim_page'], claim_quote=case['claim_quote'],
                              claim_scope=case['scope'], report_url=source['resolved_url'],
                              source_sha256=source['pdf_sha256'], quoted_words=word_count,
                              whole_claim_verdict='not_certified; document-only scope diagnosis',
                              minimum_additional_evidence='Original run/score/label identities and claim-specific decision/statistical evidence; code pointer alone insufficient.',
                              human_verification='pending'))
    for name, data in [('public_field_extraction.csv', rows), ('public_case_summary.csv', summaries)]:
        with (ROOT/name).open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    counts = {name: dict(collections.Counter(r['status'] for r in rows if r['field'] == name)) for name in FIELDS}
    result = dict(selected_reports=10, retrieved_reports=len(sources), initial_failures=2,
                  same_report_retrieval_followups=2, field_records=len(rows), counts=counts,
                  design='purposive already-cited panel; single-AI extraction; not systematic or independent validation',
                  excerpt_checks='all positive quotations located in the named PDF page under NFKC/whitespace/line-wrap-hyphen normalization',
                  no_full_artifact_or_deployment_certificate=True, human_verification='pending')
    (ROOT/'public_case_counts.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
