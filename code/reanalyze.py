"""Recompute the frozen score panel under the revision analysis contract.

Only local score arrays and metadata are read. Outputs are derived files in
results/. Original source experiments and frozen inputs are never edited.
"""
from pathlib import Path
import collections
import csv
import hashlib
import itertools
import json
import math
import platform
import sys

import numpy as np

from diagnostics import (allocation, calibrate, holm, metrics, paired_summary,
                         predict, ranking, validate, events, preserves_observed_order)

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'vendor/affiliation-metrics-py'))
from affiliation.metrics import pr_from_events

POLICIES=('val_f1','normal_005','normal_010','normal_050')
RESULTS=ROOT/'results'


def write_csv(name,rows):
    if not rows:return
    with (RESULTS/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)


def affiliation(y,pred):
    if not np.any(y) or not np.any(pred):
        return {'affiliation_precision':None,'affiliation_recall':None,'affiliation_f1':None}
    r=pr_from_events(events(pred),events(y),(0,len(y)))
    p,q=r['precision'],r['recall']
    return {'affiliation_precision':p,'affiliation_recall':q,
            'affiliation_f1':2*p*q/(p+q) if p+q else 0.}


def mean_optional(rs,key):
    vals=[r[key] for r in rs if r[key] is not None]
    return float(np.mean(vals)) if vals else None


def main():
    RESULTS.mkdir(exist_ok=True)
    manifest=json.loads((ROOT/'inputs/manifest.json').read_text())
    override=ROOT/'evaluation_overrides/labels_correct_60000.npy'
    corrected_hai=np.load(override,allow_pickle=False) if override.exists() else None
    loader_hai=np.load(ROOT/'evaluation_overrides/labels_loader_60000.npy',allow_pickle=False) if corrected_hai is not None else None
    rows=[];audit=[];base_rows=[];transform_checks=[]
    labels_by_unit={};validation_labels={}
    for ix,record in enumerate(manifest['runs']):
        panel,unit,model,seed=(record[k] for k in ['panel','unit','model','seed'])
        key=(panel,record['dataset'],unit)
        p=ROOT/record['snapshot']
        sha=hashlib.sha256(p.read_bytes()).hexdigest()
        if sha!=record['snapshot_sha256']:raise ValueError('snapshot hash mismatch')
        mp=p.parent/'metrics.json'
        if hashlib.sha256(mp.read_bytes()).hexdigest()!=record['metrics_sha256']:
            raise ValueError('metadata hash mismatch')
        meta=json.loads(mp.read_text())
        with np.load(p,allow_pickle=False) as z:
            s,y=validate(z['test_scores'],z['test_labels'])
            archived_y=y.copy()
            evaluation_labels='original_frozen'
            if record['dataset']=='hai' and corrected_hai is not None:
                if not np.array_equal(y,loader_hai):raise ValueError('HAI override not bound to frozen labels')
                _,y=validate(s,corrected_hai)
                evaluation_labels='corrected_headers_test2_positional'
            vs,vy=validate(z['val_scores'],z['val_labels'])
            yh=hashlib.sha256(y.tobytes()).hexdigest()
            if key in labels_by_unit and labels_by_unit[key]!=yh:
                raise ValueError('within-unit test labels do not align')
            labels_by_unit[key]=yh
            vk=key+(seed,)
            vh=hashlib.sha256(vy.tobytes()).hexdigest()
            val_aligned=vk not in validation_labels or validation_labels[vk]==vh
            validation_labels[vk]=vh
            archived_check=metrics(archived_y,s>=float(meta['threshold']))
            arch=metrics(y,s>=float(meta['threshold']))
            rk=ranking(s,y)
            info={k:record[k] for k in ['panel','dataset','unit','model','seed','run_id']}
            base_rows.append(dict(info,**arch,**rk,oracle_gap=rk['oracle_f1']-arch['f1']))
            recalc_thr,recalc_cmp=calibrate(vs,vy,'val_f1')
            same_pred=np.array_equal(s>=float(meta['threshold']),predict(s,recalc_thr,recalc_cmp))
            max_identity_error=0.;budgets_ok=True;invariant=True
            for policy in POLICIES:
                threshold,cmp=calibrate(vs,vy,policy)
                pred=predict(s,threshold,cmp)
                m=metrics(y,pred)
                vm=metrics(vy,predict(vs,threshold,cmp))
                rin=m['rin'] if m['rin'] is not None else 0.
                identity_error=abs(m['recall']-m['c']*rin)
                max_identity_error=max(max_identity_error,identity_error)
                alpha={'normal_005':.005,'normal_010':.01,'normal_050':.05}.get(policy)
                if alpha is not None:budgets_ok &= vm['fpr']<=alpha+1e-15
                for name,transform in [('affine_17x_plus5',lambda v:v*17.+5.),('arcsinh',np.arcsinh)]:
                    tt,cc=calibrate(transform(vs),vy,policy)
                    strict=preserves_observed_order(np.r_[vs,s],transform)
                    same=bool(np.array_equal(pred,predict(transform(s),tt,cc)))
                    transform_checks.append(dict(info,policy=policy,transform=name,
                                                 preserves_observed_order=strict,predictions_equal=same,
                                                 interpretation='admissible_invariance_test' if strict else 'finite_precision_order_collapse'))
                    if strict:invariant &= same
                scope='flattened_stream_not_trial_level' if record['dataset']=='skab' else 'archived_unit_timeline'
                rows.append(dict(info,policy=policy,threshold=threshold,comparison=cmp,
                                 **m,**rk,oracle_gap=rk['oracle_f1']-m['f1'],
                                 validation_normal_fpr=vm['fpr'],event_scope=scope,
                                 **affiliation(y,pred)))
            audit.append(dict(info,snapshot_sha256=sha,test_label_sha256=yh,
                              test_labels_aligned=True,validation_labels_aligned=val_aligned,
                              archived_f1_error=abs(archived_check['f1']-float(meta['f1'])),
                              evaluation_label_version=evaluation_labels,
                              corrected_minus_archived_f1=arch['f1']-archived_check['f1'],
                              archived_threshold_snapshot_error=abs(float(z['threshold'][0])-float(meta['threshold'])),
                              validation_reselection_prediction_match=same_pred,
                              normal_budget_checks=bool(budgets_ok),monotone_invariance=bool(invariant),
                              max_identity_error=max_identity_error))
        if (ix+1)%60==0:print('Processed',ix+1,'runs',flush=True)
    write_csv('per_run_metrics.csv',rows);write_csv('archived_operating_points.csv',base_rows)
    write_csv('integrity_audit.csv',audit)
    write_csv('monotone_transform_checks.csv',transform_checks)

    # Pair same-seed runs only within a dataset/entity and implementation panel.
    cells=collections.defaultdict(lambda:collections.defaultdict(dict))
    for r in rows:cells[(r['panel'],r['dataset'],r['unit'],r['policy'])][r['model']][r['seed']]=r
    pairs=[];allocations=[]
    for (panel,ds,unit,pol),models in sorted(cells.items()):
        for a,b in itertools.combinations(sorted(models),2):
            seeds=sorted(models[a].keys()&models[b].keys())
            if not seeds:continue
            diffs=[];evdiff=[]
            for seed in seeds:
                aa,bb=models[a][seed],models[b][seed]
                d=aa['f1']-bb['f1'];diffs.append(d)
                evdiff.append(aa['event_f1']-bb['event_f1'])
                al=allocation(aa,bb)
                if al['allocation_error']>1e-12:raise AssertionError('observed-gap allocation failed')
                allocations.append(dict(panel=panel,dataset=ds,unit=unit,policy=pol,model_a=a,model_b=b,
                                        seed=seed,f1_difference=d,**al))
            identifier='|'.join([panel,ds,unit,pol,a,b])
            ps=paired_summary(diffs,identifier)
            pairs.append(dict(panel=panel,dataset=ds,unit=unit,policy=pol,model_a=a,model_b=b,
                              **ps,event_f1_difference_mean=float(np.mean(evdiff)),
                              seed_pairing_scope='shared_seed_and_identical_test_labels_not_independent_sites'))
    byunit=collections.defaultdict(list)
    for r in pairs:byunit[(r['panel'],r['dataset'],r['unit'])].append(r)
    for rs in byunit.values():
        corrected=holm([r['sign_p'] for r in rs])
        for r,p in zip(rs,corrected):r['sign_p_holm']=p;r['holm_family_size']=len(rs)
    write_csv('paired_contrasts.csv',pairs);write_csv('signed_channel_allocations.csv',allocations)

    # Policy contrast envelope, always naming the finite family and unit.
    bypair=collections.defaultdict(list)
    for r in pairs:bypair[(r['panel'],r['dataset'],r['unit'],r['model_a'],r['model_b'])].append(r)
    envelopes=[]
    for (panel,ds,unit,a,b),rs in sorted(bypair.items()):
        means=[r['difference_mean'] for r in rs]
        lower=min(r['simultaneous_hoeffding_lo'] for r in rs)
        upper=max(r['simultaneous_hoeffding_hi'] for r in rs)
        envelopes.append(dict(panel=panel,dataset=ds,unit=unit,model_a=a,model_b=b,K=len(rs),
                              min_policy_difference=min(means),max_policy_difference=max(means),
                              descriptive_sign_reversal=min(means)<0<max(means),
                              simultaneous_mean_lower=lower,simultaneous_mean_upper=upper,
                              certified_positive_all_policies=lower>0,
                              certified_negative_all_policies=upper<0,
                              sign_consistent_corrected_positive=all(r['sign_p_holm']<.05 and r['difference_mean']>0 and r['wins']>r['losses'] for r in rs),
                              sign_consistent_corrected_negative=all(r['sign_p_holm']<.05 and r['difference_mean']<0 and r['losses']>r['wins'] for r in rs)))
    write_csv('calibration_envelopes.csv',envelopes)

    # Every model/unit retained; means and sample SD summarize seeds, not events.
    grouped=collections.defaultdict(list)
    for r in rows:grouped[(r['panel'],r['dataset'],r['unit'],r['model'],r['policy'])].append(r)
    summaries=[]
    metric_names=['f1','auc','ap','oracle_gap','fpr','rho','c','rin','segr','event_f1','affiliation_f1',
                  'false_alarm_spells_per_10000_normal','validation_normal_fpr']
    for key,rs in sorted(grouped.items()):
        item=dict(zip(['panel','dataset','unit','model','policy'],key));item['n']=len(rs)
        for k in metric_names:
            values=[r[k] for r in rs if r[k] is not None]
            item[k+'_defined_n']=len(values)
            item[k+'_mean']=float(np.mean(values)) if values else None
            item[k+'_sd']=float(np.std(values,ddof=1)) if len(values)>1 else None
        summaries.append(item)
    write_csv('model_policy_summary.csv',summaries)
    unit_alloc=collections.defaultdict(list)
    for r in allocations:
        if r['policy']=='val_f1':unit_alloc[(r['panel'],r['dataset'],r['unit'])].append(r)
    alloc_summ=[]
    for (panel,ds,unit),rs in sorted(unit_alloc.items()):
        vals=np.array([[r[k] for k in ['coverage_allocation','recall_allocation','fpr_allocation']] for r in rs])
        magnitude=np.abs(vals).mean(axis=0);tot=magnitude.sum()
        alloc_summ.append(dict(panel=panel,dataset=ds,unit=unit,n_seed_pair_contrasts=len(rs),
                               coverage_abs_allocation_mean=float(magnitude[0]),recall_abs_allocation_mean=float(magnitude[1]),
                               fpr_abs_allocation_mean=float(magnitude[2]),
                               coverage_absolute_share=float(magnitude[0]/tot) if tot else None,
                               recall_absolute_share=float(magnitude[1]/tot) if tot else None,
                               fpr_absolute_share=float(magnitude[2]/tot) if tot else None,
                               mean_absolute_net_gap=float(np.mean(np.abs([r['f1_difference'] for r in rs]))),
                               max_allocation_error=max(r['allocation_error'] for r in rs)))
    write_csv('unit_channel_summary.csv',alloc_summ)
    validity=[]
    for (panel,ds,unit),rs in sorted(byunit.items()):
        ee=[r for r in envelopes if (r['panel'],r['dataset'],r['unit'])==(panel,ds,unit)]
        validity.append(dict(panel=panel,dataset=ds,unit=unit,n_model_pairs=len(ee),
                              descriptive_sign_reversal_pairs=sum(r['descriptive_sign_reversal'] for r in ee),
                              all_policy_mean_certificates=sum(r['certified_positive_all_policies'] or r['certified_negative_all_policies'] for r in ee),
                              corrected_sign_stable_pairs=sum(r['sign_consistent_corrected_positive'] or r['sign_consistent_corrected_negative'] for r in ee)))
    write_csv('unit_calibration_summary.csv',validity)

    summary={'n_runs':len(manifest['runs']),'n_policy_rows':len(rows),'policy_family':list(POLICIES),
             'n_contrast_policy_cells':len(pairs),'n_model_pair_units':len(envelopes),
             'n_signed_allocations':len(allocations),'n_units':len(labels_by_unit),
             'panel_runs':dict(collections.Counter(r['panel'] for r in manifest['runs'])),
             'max_archived_f1_error':max(r['archived_f1_error'] for r in audit),
             'max_archived_threshold_snapshot_error':max(r['archived_threshold_snapshot_error'] for r in audit),
             'validation_reselection_prediction_matches':sum(r['validation_reselection_prediction_match'] for r in audit),
             'normal_budget_pass_runs':sum(r['normal_budget_checks'] for r in audit),
             'monotone_invariance_pass_runs':sum(r['monotone_invariance'] for r in audit),
             'monotone_admissible_cases':sum(r['preserves_observed_order'] for r in transform_checks),
             'monotone_admissible_failures':sum(r['preserves_observed_order'] and not r['predictions_equal'] for r in transform_checks),
             'finite_precision_collapse_cases':sum(not r['preserves_observed_order'] for r in transform_checks),
             'max_identity_error':max(r['max_identity_error'] for r in audit),
             'max_allocation_error':max(r['allocation_error'] for r in allocations),
             'validation_label_alignment_failures':sum(not r['validation_labels_aligned'] for r in audit),
             'units':validity,'channel_summaries':alloc_summ,'python':platform.python_version(),'numpy':np.__version__,
             'bootstrap_replicates':10000,'alpha':.05,'practical_margin':0.,
             'analysis_status':'retrospective_exploratory_not_preregistered','submission_ready':False}
    summary['evaluation_corrections']={'hai_override':str(override.relative_to(ROOT)) if corrected_hai is not None else None,
        'hai_label_sha256':hashlib.sha256(override.read_bytes()).hexdigest() if corrected_hai is not None else None,
        'original_snapshots_modified':False,'hai_test2_alignment':'positional_assumption' if corrected_hai is not None else None}
    (RESULTS/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['units','channel_summaries']},indent=2))


if __name__=='__main__':main()
