"""Build synthetic evidence bundles and a separate semantic oracle before evaluation.

Same-developer fixtures, not blinded external validation. Never imports the assessor.
"""
from pathlib import Path
import copy
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parent


def save(path, record): path.write_text(json.dumps(record, indent=2)+'\n')


def main():
    dest = ROOT/'challenge_bundles'
    assert not dest.exists(), 'preserve generated challenge set; refusing overwrite'
    dest.mkdir()
    truth = []
    for context in range(4):
        rng = np.random.default_rng(8100+context)
        val = rng.uniform(.05, .6, 80)
        test = rng.uniform(.05, .6, 160)
        labels = np.zeros(160, dtype=int); labels[20:33] = 1; labels[90:115] = 1
        test[labels.astype(bool)] += .5
        base = {
          'claim': {'target': 'pipeline', 'strict': True, 'causal_online': False,
                    'scope': 'descriptive on these supplied arrays'},
          'metric': {'reported_name': 'strict F1', 'postprocessing': 'none'},
          'calibration': {'selection_data': 'normal_validation', 'policy': 'normal_050'},
          'temporal': {'max_future_offset': 0, 'event_boundaries_known': True},
          'design': {'declared_independence': True, 'replication_unit': 'independent_entities'},
        }

        def add(tag, group, expected, reason, mutate=None, corrupt=False, transform=False):
            record = copy.deepcopy(base)
            if mutate: mutate(record)
            name = f'C{context}-{tag}'; folder = dest/name; folder.mkdir()
            vals, tests = (3*val+2, 3*test+2) if transform else (val, test)
            np.savez(folder/'scores.npz', val_scores=vals, val_labels=np.zeros(80, int),
                     test_scores=tests, test_labels=labels)
            digest = hashlib.sha256((folder/'scores.npz').read_bytes()).hexdigest()
            save(folder/'identity.json', {'file': 'scores.npz', 'expected_sha256': digest})
            if corrupt:
                changed = tests.copy(); changed[0] += .125
                np.savez(folder/'scores.npz', val_scores=vals, val_labels=np.zeros(80, int),
                         test_scores=changed, test_labels=labels)
            for kind, content in record.items(): save(folder/(kind+'.json'), content)
            truth.append(dict(case_id=name, group=group, expected=expected, rationale=reason,
                              context=context, engine_input='claim/evidence files only; this oracle is separate'))

        add('strict', 'valid_control', 'within_scope', 'Narrow strict score-array claim')
        add('pa_named', 'valid_control', 'within_scope', 'PA explicitly named; not claimed strict',
            lambda r: (r['claim'].update(strict=False), r['metric'].update(reported_name='PA-F1', postprocessing='point_adjustment')))
        add('oracle_offline', 'valid_control', 'within_scope', 'Test oracle explicitly limited to retrospective comparison',
            lambda r: r['calibration'].update(selection_data='test_labels'))
        add('offline_window', 'valid_control', 'within_scope', 'Future context disclosed for an offline claim',
            lambda r: r['temporal'].update(max_future_offset=12))
        add('deploy_fixed', 'valid_control', 'within_scope', 'Synthetic complete-bundle deployment gate control, not real utility validation',
            lambda r: r['claim'].update(target='deployment', causal_online=True))
        add('monotone', 'valid_control', 'within_scope', 'Exact positive affine score transform with recalibration', transform=True)

        add('pa_mislabeled', 'implemented_fault', 'contradicted', 'PA represented as strict pointwise F1',
            lambda r: r['metric'].update(postprocessing='point_adjustment'))
        add('oracle_deploy', 'implemented_fault', 'contradicted', 'Test-label selection promoted to deployment threshold',
            lambda r: (r['claim'].update(target='deployment'), r['calibration'].update(selection_data='test_labels')))
        add('future_online', 'implemented_fault', 'contradicted', 'Future observations contradict causal-online claim',
            lambda r: (r['claim'].update(causal_online=True), r['temporal'].update(max_future_offset=12)))
        add('artifact_changed', 'implemented_fault', 'contradicted', 'Actual file bytes disagree with archived expected hash', corrupt=True)

        add('missing_metric', 'missing_evidence', 'not_evaluable', 'Metric/postprocessing record absent',
            lambda r: r.pop('metric'))
        add('missing_calibration', 'missing_evidence', 'not_evaluable', 'Calibration source absent',
            lambda r: r.pop('calibration'))

        add('mismatch_plus_missing', 'mixed_fault_missing', 'contradicted',
            'Confirmed hash contradiction remains despite missing calibration metadata',
            lambda r: r.pop('calibration'), corrupt=True)
        add('pa_plus_missing', 'mixed_fault_missing', 'contradicted',
            'Known PA/strict contradiction remains despite missing calibration metadata',
            lambda r: (r['metric'].update(postprocessing='point_adjustment'), r.pop('calibration')))

        if context < 2:
            add('nonsignificant_equivalence', 'unimplemented_scope_probe', 'contradicted',
                'p>.05 alone cannot establish equivalence without a margin/test',
                lambda r: (r['claim'].update(equivalence_asserted=True), r['design'].update(p_value=.41, equivalence_margin=None)))
            add('seed_to_sites', 'unimplemented_scope_probe', 'contradicted',
                'Repeated seeds on one entity cannot substantiate across-site replication',
                lambda r: (r['claim'].update(target='generalization', scope='across_sites'),
                           r['design'].update(replication_unit='training_seeds', n_entities=1, n_seeds=5)))
    save(ROOT/'challenge_oracle.json', truth)
    manifest = []
    for path in sorted(dest.rglob('*')):
        if path.is_file(): manifest.append(dict(path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    save(ROOT/'challenge_manifest.json', {'cases': len(truth), 'files': manifest,
                                        'oracle_sha256': hashlib.sha256((ROOT/'challenge_oracle.json').read_bytes()).hexdigest()})
    print('Generated', len(truth), 'synthetic bundles; oracle frozen separately before evaluation')


if __name__ == '__main__': main()
