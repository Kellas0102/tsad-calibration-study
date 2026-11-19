"""Evaluate frozen rules from evidence files; oracle is joined after prediction.

Adapter is new code; rule engine is frozen. This is structured-metadata testing,
not extraction accuracy from arbitrary scientific prose.
"""
from pathlib import Path
import collections
import csv
import hashlib
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'frozen_engine'))
from diagnostics import claim_assessment, calibrate, predict


def optional(folder, name):
    p = folder/(name+'.json')
    return json.loads(p.read_text()) if p.exists() else None


def predict_bundle(folder):
    claim = optional(folder, 'claim')
    metric = optional(folder, 'metric')
    cal = optional(folder, 'calibration')
    temporal = optional(folder, 'temporal')
    design = optional(folder, 'design')
    identity = optional(folder, 'identity')
    record = dict(target=claim['target'], claim_strict=claim['strict'],
                  causal_online=claim['causal_online'], unit='synthetic_timeline')
    if metric:
        record['metric'] = 'PA-F1' if metric['postprocessing'] == 'point_adjustment' else 'strict_F1'
    if cal:
        record['calibration'] = 'test_oracle' if cal['selection_data'] == 'test_labels' else 'validation'
    if temporal:
        record.update(timeline_causal=temporal['max_future_offset'] <= 0,
                      event_boundaries_known=temporal['event_boundaries_known'])
    if design: record['independence_verified'] = design['declared_independence']
    if identity:
        file = folder/identity['file']
        assert file.resolve().is_relative_to(folder.resolve())
        actual = hashlib.sha256(file.read_bytes()).hexdigest()
        mismatch = actual != identity['expected_sha256']
        record.update(snapshot_verified=not mismatch, snapshot_hash_mismatch=mismatch)
    outcome = claim_assessment(record)
    return dict(case_id=folder.name, predicted=outcome['status'], reasons=';'.join(outcome['reasons']))


def main():
    freeze = json.loads((ROOT/'freeze_manifest.json').read_text())
    for rec in freeze['files']:
        assert hashlib.sha256((ROOT/rec['frozen']).read_bytes()).hexdigest() == rec['sha256']
    manifest = json.loads((ROOT/'challenge_manifest.json').read_text())
    for rec in manifest['files']:
        assert hashlib.sha256((ROOT/rec['path']).read_bytes()).hexdigest() == rec['sha256']
    predictions = [predict_bundle(p) for p in sorted((ROOT/'challenge_bundles').iterdir()) if p.is_dir()]
    # Predict first. The oracle was not passed to the predictor or engine.
    oracle_file = ROOT/'challenge_oracle.json'
    assert hashlib.sha256(oracle_file.read_bytes()).hexdigest() == manifest['oracle_sha256']
    oracle = {r['case_id']: r for r in json.loads(oracle_file.read_text())}
    rows = []
    for p in predictions:
        o = oracle[p['case_id']]
        rows.append(dict(**p, group=o['group'], expected=o['expected'], matched=p['predicted'] == o['expected'],
                         rationale=o['rationale']))
    with (ROOT/'challenge_results.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    bygroup = {}
    for group in sorted({r['group'] for r in rows}):
        subset = [r for r in rows if r['group'] == group]
        bygroup[group] = dict(n=len(subset), matched=sum(r['matched'] for r in subset),
                             predictions=dict(collections.Counter(r['predicted'] for r in subset)))
    # Independent paired score check for the valid affine controls.
    invariant = 0
    for context in range(4):
        masks = []
        for suffix in ['strict', 'monotone']:
            with np.load(ROOT/'challenge_bundles'/f'C{context}-{suffix}'/'scores.npz') as z:
                t, c = calibrate(z['val_scores'], z['val_labels'], 'normal_050')
                masks.append(predict(z['test_scores'], t, c))
        assert np.array_equal(*masks)
        invariant += 1
    result = dict(cases=len(rows), groups=bygroup, affine_prediction_sets_identical=invariant,
                  sampling='same-developer synthetic structured-evidence fixtures; no independent sample accuracy',
                  engine_modified_after_freeze=False,
                  issue='missing-field short circuit can mask known contradictions; two scope predicates absent')
    (ROOT/'challenge_summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
