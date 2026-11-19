"""Independent score-only receipt checks; never imports the returned runner."""
from pathlib import Path
import collections
import csv
import json
import sys

import numpy as np

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'code'))
from diagnostics import calibrate,predict,metrics,ranking


def main():
    rows=[];audit=[]
    for folder in sorted(ROOT.glob('timesnet_heldout_*')):
        if not folder.is_dir():continue
        cfg=json.loads((folder/'config.json').read_text())
        saved=json.loads((folder/'metrics.json').read_text())
        split=cfg['split'];win=split['window'];cut=split['split_index_i80']
        assert split['train_segment']==[0,cut]
        assert split['val_segment'][0]==cut
        assert split['scaler_fit_range']==[0,cut]
        assert split['train_window_range'][1]+win==cut
        assert split['val_window_range'][0]==cut
        assert cfg['scaler']['n_fit_points']==cut
        with np.load(folder/'scores.npz',allow_pickle=False) as z:
            s,y=z['test_scores'],z['test_labels']
            vs,vy=z['val_injected_scores'],z['val_injected_labels']
            assert np.all(z['val_labels']==0)
            assert len(s)==len(y)==split['test_points']
            assert len(vs)==len(z['val_scores'])==split['val_points']
            for key in ['test_scores','val_scores','val_injected_scores']:
                assert np.isfinite(z[key]).all()
            for key in ['test_coverage','val_coverage','val_injected_coverage']:
                assert z[key].min()>=1
                n=len(z[key]);expected=np.minimum.reduce([np.arange(1,n+1),np.arange(n,0,-1),np.full(n,win)])
                np.testing.assert_array_equal(z[key],expected)
            threshold,cmp=calibrate(vs,vy,'val_f1')
            m=metrics(y,predict(s,threshold,cmp))
            err=abs(m['f1']-saved['f1_test'])
            assert err<6e-9
            assert abs(threshold-saved['thr_f1'])<1e-12
            for policy in ['val_f1','normal_005','normal_010','normal_050']:
                if policy=='val_f1':t,c=threshold,cmp;source='injected_validation'
                else:t,c=calibrate(z['val_scores'],z['val_labels'],policy);source='unmodified_normal_validation'
                q=metrics(y,predict(s,t,c))
                rows.append(dict(run_id=folder.name,unit=saved['entity'],seed=saved.get('seed',cfg['seed']),
                                 policy=policy,calibration_source=source,threshold=t,comparison=c,**q,**ranking(s,y)))
            # Exported budget was on ALL injected scores, not normal validation.
            b=cfg['thresholds']['order_stat_budget_gt']
            exported=metrics(y,predict(s,b['threshold'],'>'))
            assert abs(exported['f1']-saved['f1_test_budget_strategy'])<6e-9
            actual=int((vs>b['threshold']).sum())
            audit.append(dict(run_id=folder.name,f1_saved_error=err,
                              split_geometry_verified=True,coverage_geometry_verified=True,
                              saved_budget_calibration_source='all_injected_validation_scores',
                              saved_budget_nominal_k=b['k'],saved_budget_actual_exceedances=actual,
                              saved_budget_exact_k_claim_verified=actual==b['k']))
    for name,records in [('per_run.csv',rows),('audit.csv',audit)]:
        with (ROOT/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    groups=collections.defaultdict(list)
    for r in rows:groups[(r['unit'],r['policy'])].append(r)
    summary=[]
    for (unit,policy),rs in sorted(groups.items()):
        summary.append(dict(unit=unit,policy=policy,n=len(rs),f1_mean=float(np.mean([r['f1'] for r in rs])),
                            f1_sd=float(np.std([r['f1'] for r in rs],ddof=1)),
                            auc_mean=float(np.mean([r['auc'] for r in rs])),fpr_mean=float(np.mean([r['fpr'] for r in rs]))))
    result=dict(heldout_timesnet_runs=len(audit),summary=summary,
                max_saved_rounded_f1_error=max(r['f1_saved_error'] for r in audit),
                saved_budget_exact_k_failures=sum(not r['saved_budget_exact_k_claim_verified'] for r in audit),
                model_execution=False,proof_scope='exported_arrays_and_supplied_source_not_independent_retraining')
    (ROOT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
