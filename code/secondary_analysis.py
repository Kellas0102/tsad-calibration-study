"""Declared retrospective policy-family and alarm-consolidation sensitivity.

No model execution. Uses only frozen score arrays and existing contrast tables.
"""
from pathlib import Path
import collections
import csv
import json

import numpy as np
from diagnostics import calibrate,events,metrics,predict

ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'results'


def save(name,rows):
    with (R/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def consolidate(pred,gap):
    output=np.asarray(pred,dtype=bool).copy()
    intervals=events(output)
    for (_,b),(c,_) in zip(intervals,intervals[1:]):
        if c-b<=gap:output[b:c]=True
    return output


def alarm_detail(y,pred):
    gt=events(y);alarms=events(pred)
    overlapping=sum(any(max(a,c)<min(b,d) for c,d in gt) for a,b in alarms)
    m=metrics(y,pred)
    return dict(m,event_precision=m['event_tp']/len(alarms) if alarms else None,
                event_recall=m['event_tp']/len(gt) if gt else None,
                alarms_with_true_overlap=overlapping,
                unmatched_overlapping_alarms=overlapping-m['event_tp'],
                alarms_without_true_overlap=len(alarms)-overlapping)


def main():
    pairs=list(csv.DictReader((R/'paired_contrasts.csv').open()))
    results=[]
    for panel in ['legacy','recent']:
        for family,policies in [('all_four',{'val_f1','normal_005','normal_010','normal_050'}),
                                ('normal_only',{'normal_005','normal_010','normal_050'})]:
            group=collections.defaultdict(list)
            for row in pairs:
                if row['panel']==panel and row['policy'] in policies:
                    group[(row['dataset'],row['unit'],row['model_a'],row['model_b'])].append(float(row['difference_mean']))
            for delta in [0.,.005,.01,.02]:
                results.append(dict(panel=panel,family=family,K=len(policies),reference_margin=delta,
                                    model_pair_units=len(group),reversals=sum(min(v)<-delta and max(v)>delta for v in group.values()),
                                    status='post_hoc_descriptive_sensitivity'))
    save('policy_family_sensitivity.csv',results)
    manifest=json.loads((ROOT/'inputs/manifest.json').read_text())
    rows=[]
    for record in manifest['runs']:
        if record['panel']!='legacy' or record['dataset']!='hai':continue
        with np.load(ROOT/record['snapshot'],allow_pickle=False) as z:
            override=ROOT/'evaluation_overrides/labels_correct_60000.npy'
            y=(np.load(override,allow_pickle=False) if override.exists() else z['test_labels']).astype(bool)
            for policy in ['val_f1','normal_005','normal_010','normal_050']:
                t,cmp=calibrate(z['val_scores'],z['val_labels'],policy)
                raw=predict(z['test_scores'],t,cmp)
                for g in [0,1,5,10,30,60]:
                    m=alarm_detail(y,consolidate(raw,g))
                    rows.append(dict(run_id=record['run_id'],model=record['model'],seed=record['seed'],
                                     policy=policy,gap_points=g,**m))
    save('hai_consolidation_per_run.csv',rows)
    groups=collections.defaultdict(list)
    for row in rows:groups[(row['model'],row['policy'],row['gap_points'])].append(row)
    summaries=[]
    fields=['f1','fpr','event_f1','event_precision','event_recall','n_predicted_events',
            'event_tp','event_fp','event_fn','unmatched_overlapping_alarms','alarms_without_true_overlap']
    for key,rs in sorted(groups.items()):
        row=dict(zip(['model','policy','gap_points'],key));row['n_seeds']=len(rs)
        for k in fields:
            v=[r[k] for r in rs if r[k] is not None]
            row[k+'_mean']=float(np.mean(v)) if v else None
            row[k+'_sd']=float(np.std(v,ddof=1)) if len(v)>1 else None
        summaries.append(row)
    save('hai_consolidation_summary.csv',summaries)
    (R/'secondary_analysis_summary.json').write_text(json.dumps({'policy_family':results,
                    'hai_original_runs':len(set(r['run_id'] for r in rows)),
                    'hai_run_policy_width_records':len(rows),'gap_widths':[0,1,5,10,30,60],
                    'new_training_runs':0,'status':'post_hoc_sensitivity_not_deployment_validation'},indent=2)+'\n')
    print('Saved',len(results),'policy-family sensitivity rows;',len(rows),'HAI run/policy/width rows')
    for row in results:
        if row['reference_margin']==.01:print(row)
    for row in summaries:
        if row['model']=='ours' and row['policy'] in ['val_f1','normal_050'] and row['gap_points'] in [0,10,60]:
            print({k:v for k,v in row.items() if k.endswith('_mean') or k in ['model','policy','gap_points']})


if __name__=='__main__':main()
