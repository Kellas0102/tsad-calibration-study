"""Independent exported-score audit. No detector imports, training, or inference.

Run from this receipt folder; --code can select the preserved diagnostics directory.
"""
from pathlib import Path
import argparse
import collections
import csv
import hashlib
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
POLICIES = ('val_f1', 'normal_005', 'normal_010', 'normal_050')


def coverage(n, w):
    t = np.arange(n)
    return np.minimum(n-w, t)+1-np.maximum(0, t-w+1)


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--code', type=Path, default=ROOT.parent/'DMKD_revision_20261110/code')
    args = parser.parse_args()
    sys.path.insert(0, str(args.code))
    from diagnostics import calibrate, predict, metrics, ranking
    source = ROOT/'corrected'
    verified = 0
    for line in (source/'PUBLIC_SHA256SUMS.txt').read_text(encoding='utf-8-sig').splitlines():
        if not line.strip(): continue
        digest, name = line.split(maxsplit=1)
        path = source/name.lstrip('*')
        assert path.resolve().is_relative_to(source.resolve())
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name
        verified += 1
    rows, audits, random_rows = [], [], []
    for folder in sorted((source/'corrected').iterdir()):
        if not folder.is_dir(): continue
        cfg = json.loads((folder/'config.json').read_text())
        saved = json.loads((folder/'metrics.json').read_text())
        split = cfg['split']; w = split['window']; cut = split['split_index_i80']
        model = cfg['model']
        assert split['train_segment'] == [0, cut]
        assert split['val_segment'] == [cut, split['raw_train_points']]
        assert split['scaler_fit_range'] == [0, cut]
        assert cut == int(.8*split['raw_train_points'])
        assert split['train_window_range'][1]+w == cut
        assert split['val_window_range'][0] == cut
        cov_n = 0; matrix_error = None; saved_errors = []; threshold_errors = []
        with np.load(folder/'scores.npz', allow_pickle=False) as z:
            y, vy, iy = z['test_labels'], z['val_labels'], z['val_injected_labels']
            assert len(y) == split['test_points']
            assert len(vy) == len(iy) == split['val_points']
            assert np.all(vy == 0)
            timesnet = args.code.parent/'received_heldout'/f"timesnet_heldout_{cfg['entity']}_seed{cfg['seed']}"/'scores.npz'
            with np.load(timesnet, allow_pickle=False) as reference:
                for key in ['test_labels', 'val_labels', 'val_injected_labels', 'split_index']:
                    np.testing.assert_array_equal(z[key], reference[key])
            for labels in [y, vy, iy]: assert np.isin(labels, [0, 1]).all()
            for key in z.files:
                if 'coverage' in key:
                    np.testing.assert_array_equal(z[key], coverage(len(z[key]), w))
                    cov_n += 1
            if model == 'MEMTO':
                W = z['window_scores_test']; starts = z['window_index_test']
                np.testing.assert_array_equal(starts, np.arange(len(y)-w+1))
                assert W.shape == (len(starts), w) and np.isfinite(W).all()
                # Independent diagonal scatter, not the returned row-loop implementation.
                acc = np.zeros(len(y)); cnt = np.zeros(len(y), dtype=int)
                for j in range(w):
                    acc[j:j+len(W)] += W[:, j]
                    cnt[j:j+len(W)] += 1
                reconstructed = acc/cnt
                np.testing.assert_allclose(reconstructed, z['test_scores'], rtol=2e-6, atol=2e-9)
                matrix_error = float(np.max(np.abs(reconstructed-z['test_scores'])))
                assert saved['memory_sha256_loaded'] == cfg['memory']['mem_final_sha256']
                rngs = [None]
            else:
                np.testing.assert_array_equal(z['rng_list'], np.arange(5))
                assert cfg['schedule_facts']['mask_step'] == saved['mask_step'] == 17
                assert cfg['schedule_facts']['epochs'] == saved['epochs'] == 3
                assert len(json.loads((folder/'epoch_log.json').read_text())) == 3
                rngs = range(5)
            f1_values, hashes = [], []
            for rng in rngs:
                suffix = '' if rng is None else f'_rng{rng}'
                s, vs, ins = [z[k+suffix] for k in ['test_scores', 'val_scores', 'val_injected_scores']]
                assert len(s) == len(y) and len(vs) == len(vy) and len(ins) == len(iy)
                for a in [s, vs, ins]: assert np.isfinite(a).all()
                rank = ranking(s, y)
                for policy in POLICIES:
                    cs, cy = (ins, iy) if policy == 'val_f1' else (vs, vy)
                    t, cmp = calibrate(cs, cy, policy)
                    m = metrics(y, predict(s, t, cmp))
                    record = dict(run_id=folder.name, model=model, entity=cfg['entity'],
                                  seed=cfg['seed'], rng=rng, policy=policy, threshold=t,
                                  comparison=cmp, **m, **rank)
                    rows.append(record)
                    if policy == 'val_f1':
                        original = saved if rng is None else saved['per_rng'][rng]
                        archived_t = original['threshold_exact_f1max'] if rng is None else original['threshold']
                        archived_f1 = original['f1_test']
                        f1_values.append(m['f1'])
                        saved_errors.append(abs(m['f1']-archived_f1))
                        threshold_errors.append(abs(t-archived_t))
                    else:
                        original = saved if rng is None else saved['per_rng'][rng]
                        alpha = {'normal_005': .005, 'normal_010': .01, 'normal_050': .05}[policy]
                        b = original['budget_strategies'][f'alpha_{alpha}']
                        assert t == b['threshold'] and cmp == b['comparison']
                        saved_errors.append(abs(m['f1']-b['f1_test']))
                if rng is not None:
                    digest = hashlib.sha256(np.ascontiguousarray(s).tobytes()).hexdigest()
                    assert digest[:32] == saved['per_rng'][rng]['score_sha256']
                    hashes.append(digest)
            assert max(saved_errors) < 6e-9
            assert max(threshold_errors) < 1e-12
            if model == 'CATCH':
                assert len(set(hashes)) == saved['distinct_score_hashes'] == 5
                random_rows.append(dict(run_id=folder.name, f1_sd=float(np.std(f1_values, ddof=1)),
                                        f1_range=float(np.ptp(f1_values)), distinct_test_hashes=5,
                                        rng0_replay_reported=saved['per_rng'][0]['replay_identical'],
                                        replay_arrays_exported=False))
            audits.append(dict(run_id=folder.name, model=model, coverage_arrays=cov_n,
                               test_matrix_max_abs_error=matrix_error,
                               max_saved_f1_error=max(saved_errors),
                               max_threshold_error=max(threshold_errors)))
    groups = collections.defaultdict(list)
    for r in rows:
        if r['rng'] in [None, 0]: groups[(r['model'], r['entity'], r['policy'])].append(r)
    summaries = []
    for (model, entity, policy), records in sorted(groups.items()):
        r = dict(model=model, entity=entity, policy=policy, n=len(records))
        for field in ['f1', 'event_f1', 'auc', 'fpr', 'c', 'rin']:
            values = [q[field] for q in records if q[field] is not None]
            r[field+'_mean'] = float(np.mean(values)) if values else None
            r[field+'_sd'] = float(np.std(values, ddof=1)) if len(values)>1 else None
        summaries.append(r)
    write_csv(ROOT/'verified_per_run.csv', rows)
    write_csv(ROOT/'verified_summary.csv', summaries)
    write_csv(ROOT/'verified_audit.csv', audits)
    write_csv(ROOT/'catch_rng_audit.csv', random_rows)
    result = dict(archive_checksum_entries_verified=verified, score_runs=len(audits),
                  coverage_arrays_verified=sum(r['coverage_arrays'] for r in audits),
                  policy_rng_records=len(rows), memto_test_matrices_reaggregated=10,
                  catch_rng_f1_sd_min=min(r['f1_sd'] for r in random_rows),
                  catch_rng_f1_sd_max=max(r['f1_sd'] for r in random_rows),
                  catch_rng_f1_max_range=max(r['f1_range'] for r in random_rows),
                  max_saved_f1_error=max(r['max_saved_f1_error'] for r in audits),
                  model_execution=False, replay_scope='rng0 test replay reported by runner; replay vectors absent',
                  acceptance='score-level consistency accepted; model-specific training/state limitations retained')
    (ROOT/'verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
