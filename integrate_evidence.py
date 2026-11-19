"""Read-only source integration, corrected score analysis, no model execution."""
from pathlib import Path
import collections
import csv
import hashlib
import itertools
import json
import sys

import numpy as np

ROOT=Path(__file__).resolve().parent
OLD=ROOT
INC=ROOT/'received_evidence'
sys.path.insert(0,str(OLD/'code'))
from diagnostics import metrics,calibrate,predict,ranking,allocation
from reanalyze import affiliation
from secondary_analysis import consolidate,alarm_detail

POLICIES=['val_f1','normal_005','normal_010','normal_050']
OUT=ROOT/'results'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def write(name,rows):
    if not rows:return
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def grouped(rows,keys,fields):
    groups=collections.defaultdict(list)
    for r in rows:groups[tuple(r[k] for k in keys)].append(r)
    result=[]
    for key,rs in sorted(groups.items()):
        r=dict(zip(keys,key));r['n_records']=len(rs)
        for field in fields:
            values=[x[field] for x in rs if x[field] is not None]
            r[field+'_defined_n']=len(values)
            r[field+'_mean']=float(np.mean(values)) if values else None
            r[field+'_sd']=float(np.std(values,ddof=1)) if len(values)>1 else None
        result.append(r)
    return result


def segmented(y,pred,bounds):
    chunks=[metrics(y[a:b],pred[a:b]) for a,b in bounds]
    counts={k:sum(r[k] for r in chunks) for k in ['event_tp','event_fp','event_fn']}
    et,ef,en=[counts[k] for k in ['event_tp','event_fp','event_fn']]
    af=[affiliation(y[a:b],pred[a:b])['affiliation_f1'] for a,b in bounds]
    macro=[r['event_f1'] for r in chunks if r['event_f1'] is not None]
    av=[v for v in af if v is not None]
    return dict(**counts,event_f1=2*et/(2*et+ef+en) if 2*et+ef+en else None,
                event_f1_macro=float(np.mean(macro)) if macro else None,
                event_macro_defined_n=len(macro),affiliation_macro=float(np.mean(av)) if av else None,
                affiliation_macro_defined_n=len(av))


def main():
    OUT.mkdir(exist_ok=True)
    manifest=json.loads((OLD/'inputs/manifest.json').read_text())['runs']
    yc=np.load(INC/'04_p0_hai/labels_correct_60000.npy',allow_pickle=False)
    yl=np.load(INC/'04_p0_hai/labels_loader_60000.npy',allow_pickle=False)
    assert yc.shape==yl.shape==(60000,)
    changes=int(np.count_nonzero(yc!=yl));assert changes==32
    rows=[];merge=[];boundary=[]
    for rec in manifest:
        if rec['dataset']!='hai':continue
        p=OLD/rec['snapshot'];assert sha(p)==rec['snapshot_sha256']
        with np.load(p,allow_pickle=False) as z:
            assert np.array_equal(z['test_labels'],yl)
            for scenario,y,n in [('original_labels',yl,60000),('corrected_positional',yc,60000),('test1_verified_only',yc[:54000],54000)]:
                s=z['test_scores'][:n];rk=ranking(s,y)
                for policy in POLICIES:
                    t,cmp=calibrate(z['val_scores'],z['val_labels'],policy)
                    pred=predict(s,t,cmp);m=metrics(y,pred)
                    rows.append(dict(run_id=rec['run_id'],model=rec['model'],seed=rec['seed'],scenario=scenario,
                                     policy=policy,**m,**rk,**affiliation(y,pred)))
                    if scenario!='original_labels':
                        bounds=[(0,54000),(54000,60000)] if n==60000 else [(0,n)]
                        boundary.append(dict(run_id=rec['run_id'],model=rec['model'],seed=rec['seed'],
                                             scenario=scenario,policy=policy,**segmented(y,pred,bounds)))
                        for width in [0,1,5,10,30,60]:
                            detail=alarm_detail(y,consolidate(pred,width))
                            merge.append(dict(run_id=rec['run_id'],model=rec['model'],seed=rec['seed'],
                                              scenario=scenario,policy=policy,gap_points=width,**detail))
    write('hai_per_run_corrected_and_sensitivity.csv',rows)
    fields=['f1','event_f1','affiliation_f1','c','rin','segr','fpr','auc']
    sums=grouped(rows,['scenario','model','policy'],fields)
    write('hai_model_summary.csv',sums)
    write('hai_file_boundary_events.csv',boundary)
    write('hai_consolidation_per_run.csv',merge)
    write('hai_consolidation_summary.csv',grouped(merge,['scenario','model','policy','gap_points'],
              ['f1','event_f1','event_precision','event_recall','fpr','unmatched_overlapping_alarms','alarms_without_true_overlap']))
    cells=collections.defaultdict(dict)
    for r in rows:cells[(r['scenario'],r['policy'],r['seed'])][r['model']]=r
    parts=[]
    for (scenario,policy,seed),models in cells.items():
        for a,b in itertools.combinations(sorted(models),2):
            al=allocation(models[a],models[b]);assert al['allocation_error']<1e-12
            parts.append(dict(scenario=scenario,policy=policy,seed=seed,model_a=a,model_b=b,**al))
    write('hai_signed_allocations.csv',parts)

    trials=json.loads((INC/'04_p1_skab/p1_skab_boundary.json').read_text())['trials']
    reconstructed=np.load(ROOT/'evaluation_overrides/skab_reconstructed_labels.npy',allow_pickle=False)
    labels=[];filehash=[];offset=0
    for trial in trials:
        y=reconstructed[trial['offset_start']:trial['offset_end']]
        assert len(y)==trial['rows'] and int(y.sum())==trial['anomalies']
        assert trial['offset_start']==offset;offset+=len(y);assert trial['offset_end']==offset
        labels.append(y);filehash.append(dict(path=trial['rel'],label_vector_sha256=hashlib.sha256(y.tobytes()).hexdigest(),n_points=len(y)))
    ys=np.concatenate(labels);assert len(ys)==37401
    sb=[]
    for rec in manifest:
        if rec['dataset']!='skab':continue
        p=OLD/rec['snapshot'];assert sha(p)==rec['snapshot_sha256']
        with np.load(p,allow_pickle=False) as z:
            assert np.array_equal(z['test_labels'],ys)
            for policy in POLICIES:
                t,cmp=calibrate(z['val_scores'],z['val_labels'],policy);pred=predict(z['test_scores'],t,cmp)
                flat=metrics(ys,pred)
                for group in ['all_trials','other','valve1','valve2']:
                    selected=[q for q in trials if group=='all_trials' or q['group']==group]
                    bounds=[(q['offset_start'],q['offset_end']) for q in selected]
                    sb.append(dict(run_id=rec['run_id'],panel=rec['panel'],model=rec['model'],seed=rec['seed'],
                                   policy=policy,group=group,n_trials=len(bounds),flat_stream_event_f1=flat['event_f1'],
                                   **segmented(ys,pred,bounds)))
    write('skab_boundary_aware_per_run.csv',sb)
    write('skab_boundary_summary.csv',grouped(sb,['panel','model','policy','group'],
                 ['flat_stream_event_f1','event_f1','event_f1_macro','affiliation_macro']))
    native=[]
    for folder in sorted((INC/'04_runs').glob('timesnet_official*')):
        if not folder.is_dir():continue
        meta=json.loads((folder/'metrics.json').read_text())
        with np.load(folder/'scores.npz',allow_pickle=False) as z:
            rk=ranking(z['test_scores'],z['test_labels']);t=float(z['threshold'][0])
            for policy in ['archived_gt','archived_ge']+POLICIES:
                if policy.startswith('archived_'):tau,cmp=t,('>' if policy=='archived_gt' else '>=')
                else:tau,cmp=calibrate(z['val_scores'],z['val_labels'],policy)
                m=metrics(z['test_labels'],predict(z['test_scores'],tau,cmp))
                native.append(dict(run_id=folder.name,unit=meta['entity'],seed=meta['seed'],policy=policy,
                                   cohort='native_training_validation_overlap',threshold=tau,comparison=cmp,
                                   archived_F1=meta['F1'],**m,**rk))
    write('timesnet_native_verified.csv',native)
    write('timesnet_native_summary.csv',grouped(native,['unit','policy'],['f1','auc','event_f1','fpr']))
    summary=dict(hai_runs=30,hai_label_differences=changes,hai_policy_scenario_rows=len(rows),
                 hai_consolidation_rows=len(merge),hai_allocation_max_error=max(r['allocation_error']for r in parts),
                 skab_runs_verified=len(set(r['run_id']for r in sb)),skab_trials=len(trials),
                 timesnet_native_runs=len(set(r['run_id']for r in native)),
                 hai_test2_alignment='positional_assumption_not_verified_seconds',new_model_execution=False)
    (OUT/'integration_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (OUT/'skab_reference_files.json').write_text(json.dumps(filehash,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
