"""Generate paper tables, numerical macros and deterministic data figures."""
from pathlib import Path
import collections
import csv
import hashlib
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from diagnostics import metrics

ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'results';M=ROOT/'manuscript';F=M/'figures'
F.mkdir(exist_ok=True)
LABELS={'ours':'Diagnostic model','autoencoder':'Autoencoder','cnn-lstm':'CNN--LSTM',
        'lstm':'LSTM','gru':'GRU','transformer':'Transformer','timesnet':'TimesNet',
        'memto':'MEMTO','catch':'CATCH'}


def load(name):
    return list(csv.DictReader((R/name).open()))


def number(x,d=3):
    return '--' if x in ['',None] else f'{float(x):.{d}f}'


def table(name,header,rows,columns):
    text=['\\begin{tabular}{'+columns+'}','\\toprule',header+' \\\\','\\midrule']
    text += [' & '.join(map(str,row))+' \\\\' for row in rows]
    text += ['\\bottomrule','\\end{tabular}']
    (M/name).write_text('\n'.join(text)+'\n')


def main():
    summary=json.loads((R/'summary.json').read_text())
    sm=load('model_policy_summary.csv');uc=load('unit_calibration_summary.csv')
    cm=load('unit_channel_summary.csv');run=load('per_run_metrics.csv')
    macros={'NRun':summary['n_runs'],'NLegacy':summary['panel_runs']['legacy'],
            'NRecent':summary['panel_runs']['recent'],'NPolicyRows':summary['n_policy_rows'],
            'NContrastCells':summary['n_contrast_policy_cells'],'NAllocations':summary['n_signed_allocations'],
            'NAdmissibleTransforms':summary['monotone_admissible_cases'],
            'NCollapseCases':summary['finite_precision_collapse_cases']}
    for panel in ['legacy','recent']:
        rs=[r for r in uc if r['panel']==panel]
        key=panel.capitalize()
        macros[key+'Reversals']=sum(int(r['descriptive_sign_reversal_pairs']) for r in rs)
        macros[key+'Pairs']=sum(int(r['n_model_pairs']) for r in rs)
    raw=load('monotone_transform_checks.csv')
    collapsed=[r for r in raw if r['preserves_observed_order']=='False']
    macros['NCollapseRuns']=len(set(r['run_id'] for r in collapsed))
    macros['NChangedCollapseCases']=sum(r['predictions_equal']=='False' for r in collapsed)
    (M/'generated_macros.tex').write_text('\n'.join('\\newcommand{\\'+k+'}{'+str(v)+'}' for k,v in macros.items())+'\n')
    frozen=json.loads((ROOT/'inputs/manifest.json').read_text())
    units=collections.defaultdict(list)
    for r in frozen['runs']:units[(r['panel'],r['dataset'],r['unit'])].append(r)
    cohort=[]
    for (panel,ds,unit),rs in sorted(units.items()):
        counts=collections.Counter(r['model'] for r in rs)
        seeds=sorted(set(counts.values()))
        cohort.append([panel,ds.upper(),unit.replace('_','\\_'),len(counts),'/'.join(map(str,seeds)),len(rs)])
    table('table_cohort.tex','Panel & Dataset & Unit & Models & Seeds/model & Runs',cohort,'llllrr')
    table('table_calibration.tex','Panel & Unit & Model pairs & Sign reversals',
          [[r['panel'],r['dataset'].upper()+'/'+r['unit'],r['n_model_pairs'],r['descriptive_sign_reversal_pairs']]for r in uc], 'llrr')
    recent=[]
    for r in sm:
        if r['panel']=='recent' and r['policy']=='val_f1':
            recent.append([r['dataset'].upper(),LABELS[r['model']],
                           number(r['f1_mean'])+'$\\pm$'+number(r['f1_sd']),number(r['auc_mean']),
                           number(r['c_mean']),number(r['rin_mean']),number(r['oracle_gap_mean'])])
    table('table_recent.tex','Dataset & Adaptation & F1 & AUC & $c$ & $r_{\\rm in}$ & $g$',recent,'llrrrrr')
    channel=[]
    for r in cm:
        channel.append([r['panel'],r['dataset'].upper()+'/'+r['unit'],
                        f"{100*float(r['coverage_absolute_share']):.1f}",
                        f"{100*float(r['recall_absolute_share']):.1f}",
                        f"{100*float(r['fpr_absolute_share']):.1f}"])
    table('table_channels.tex','Panel & Unit & Coverage (\\%) & In-event (\\%) & FPR (\\%)',channel,'llrrr')
    hai=[]
    for pol in ['val_f1','normal_050']:
        for model in ['ours','transformer']:
            r=next(r for r in sm if r['panel']=='legacy' and r['dataset']=='hai' and r['model']==model and r['policy']==pol)
            hai.append([LABELS[model], 'F1-max' if pol=='val_f1' else 'Normal 5\\%',
                        *[number(r[k]) for k in ['f1_mean','event_f1_mean','affiliation_f1_mean','fpr_mean']]])
    table('table_hai.tex','Model & Calibration & Point F1 & Event F1 & Affil. F1 & Test FPR',hai,'llrrrr')
    budget=[]
    for panel in ['legacy','recent']:
        for pol,a in [('normal_005',.005),('normal_010',.01),('normal_050',.05)]:
            rr=[r for r in run if r['panel']==panel and r['policy']==pol]
            budget.append([panel,f'{100*a:g}\\%',len(rr),sum(float(r['fpr'])>a for r in rr)])
    table('table_budget.tex','Panel & Validation target & Runs & Test FPR above target',budget,'llrr')
    envelope=load('calibration_envelopes.csv')
    sensitivity=[]
    for panel in ['legacy','recent']:
        ee=[r for r in envelope if r['panel']==panel]
        for delta in [0.,.005,.01,.02]:
            sensitivity.append(dict(panel=panel,reference_margin=delta,n_model_pair_units=len(ee),
                                    n_descriptive_material_reversals=sum(float(r['min_policy_difference'])< -delta and float(r['max_policy_difference'])>delta for r in ee),
                                    status='post_hoc_sensitivity_not_operational_threshold'))
    with (R/'margin_sensitivity.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(sensitivity[0]));w.writeheader();w.writerows(sensitivity)
    table('table_margin_sensitivity.tex','Panel & Reference $\\delta$ & Pairs & Reversals',
          [[r['panel'],f"{r['reference_margin']:.3f}",r['n_model_pair_units'],r['n_descriptive_material_reversals']] for r in sensitivity],'lrrr')
    pf=load('policy_family_sensitivity.csv')
    table('table_policy_families.tex','Panel & Family & $\\delta$ & Pairs & Reversals',
          [[r['panel'],'Four policies' if r['family']=='all_four' else 'Normal only',number(r['reference_margin']),r['model_pair_units'],r['reversals']] for r in pf],'llrrr')
    cons=load('hai_consolidation_summary.csv')
    table('table_consolidation.tex','Gap & Policy & Point F1 & Event F1 & Event P & Test FPR',
          [[r['gap_points'],'F1-max' if r['policy']=='val_f1' else 'Normal 5\\%',
            *[number(r[k]) for k in ['f1_mean','event_f1_mean','event_precision_mean','fpr_mean']]]
           for r in cons if r['model']=='ours' and r['policy'] in ['val_f1','normal_050']],'rlrrrr')

    # sn-jnl's 31pc single-column text area scales a 174mm figure to about 75%.
    # Use 11pt source lettering so final-size lettering remains about 8pt.
    integration=load('hai_model_summary.csv')
    table('table_hai_alignment_sensitivity.tex','Labels / scope & Policy & Point F1 & Event F1 & Affil. F1',
          [[{'original_labels':'Original 60k','corrected_positional':'Corrected 60k','test1_verified_only':'Verified test1'}[r['scenario']],
            'F1-max' if r['policy']=='val_f1' else 'Normal 5\\%',
            *[number(r[k]) for k in ['f1_mean','event_f1_mean','affiliation_f1_mean']]]
           for r in integration if r['model']=='ours' and r['policy'] in ['val_f1','normal_050']],'llrrr')
    sk=load('skab_boundary_summary.csv')
    table('table_skab_boundary_sensitivity.tex','Model & Stream F1 & Pooled-trial F1 & Macro-trial F1',
          [[LABELS[r['model']],number(r['flat_stream_event_f1_mean']),number(r['event_f1_mean']),number(r['event_f1_macro_mean'])]
           for r in sk if r['panel']=='legacy' and r['group']=='all_trials' and r['policy']=='val_f1'],'lrrr')
    nat=load('timesnet_native_summary.csv')
    table('table_timesnet_native.tex','SMD entity & Runs & Strict F1 & AUC & Test FPR',
          [[r['unit'],r['n_records'],number(r['f1_mean'])+'$\\pm$'+number(r['f1_sd']),number(r['auc_mean']),number(r['fpr_mean'])]
           for r in nat if r['policy']=='archived_gt'],'lrrrr')
    style={'font.size':11,'axes.titlesize':11,'axes.labelsize':11,'legend.fontsize':11,
           'pdf.fonttype':42,'ps.fonttype':42,'font.family':'DejaVu Sans',
           'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'}
    provenance=[]
    def save(fig,stem,source,description):
        fig.savefig(F/(stem+'.pdf'))
        fig.savefig(F/(stem+'.png'),dpi=300)
        provenance.append({'figure':stem,'source':source,'description':description,
                           'pdf_sha256':hashlib.sha256((F/(stem+'.pdf')).read_bytes()).hexdigest()})
        plt.close(fig)
    with plt.rc_context(style):
        # Constructed counterexample, not empirical data and not a simulated run.
        y=np.zeros(24,dtype=int);y[2:7]=1;y[12:17]=1
        a=np.zeros_like(y);a[2:7]=1
        b=np.zeros_like(y);b[2:5]=1;b[12:14]=1
        ma,mb=metrics(y,a),metrics(y,b)
        fig,ax=plt.subplots(figsize=(6.85,2.25),layout='constrained')
        for row,(name,v) in enumerate([('Ground truth',y),('A: one event covered',a),('B: both events touched',b)]):
            for t in range(len(y)):
                ax.broken_barh([(t,1)],(2-row-.3,.6),facecolors='#333333' if v[t] else '#eeeeee',edgecolors='white',linewidth=.3)
        ax.set(yticks=[2,1,0],yticklabels=['Ground truth','A','B'],xlabel='Constructed time step',xlim=(0,24),ylim=(-.65,2.5))
        ax.text(18,1,'$c=0.5,\ r_{in}=1$',va='center')
        ax.text(18,0,'$c=1,\ r_{in}=0.5$',va='center')
        fixture={'ground_truth':y.tolist(),'prediction_a':a.tolist(),'prediction_b':b.tolist(),'metrics_a':ma,'metrics_b':mb}
        (R/'identifiability_fixture.json').write_text(json.dumps(fixture,indent=2)+'\n')
        save(fig,'Fig1_identifiability','results/identifiability_fixture.json','Constructed equal-F1 predictions with different event coverage; no uncertainty bars or empirical claims.')

        fig,axes=plt.subplots(1,2,figsize=(6.85,2.65),sharey=True,layout='constrained')
        policies=['val_f1','normal_005','normal_010','normal_050']
        for ax,ds in zip(axes,['smd','skab']):
            for model,color,marker,ls in [('timesnet','#0072B2','o','-'),('catch','#333333','s','--'),('memto','#777777','^',':')]:
                rr=[next(r for r in sm if r['panel']=='recent' and r['dataset']==ds and r['model']==model and r['policy']==p) for p in policies]
                ax.errorbar(range(4),[float(r['f1_mean']) for r in rr],yerr=[float(r['f1_sd']) for r in rr],
                            marker=marker,color=color,linestyle=ls,label=LABELS[model],capsize=2,linewidth=1)
            ax.set(xticks=range(4),xticklabels=['F1-max','0.5%','1%','5%'],ylim=(0,1),title='SMD machine-1-1' if ds=='smd' else 'SKAB flattened stream',xlabel='Validation policy\nnormal exceedance target')
        axes[0].set_ylabel('Strict point-wise F1');axes[1].legend(frameon=False)
        save(fig,'Fig2_recent_policies','results/model_policy_summary.csv','All three recent adaptations and four policies, mean plus/minus sample SD over five seeds; no superiority stars.')

        fig,axes=plt.subplots(1,2,figsize=(6.85,2.65),sharey=True,layout='constrained')
        for ax,pol in zip(axes,['val_f1','normal_050']):
            for model,color,marker in [('ours','#333333','o'),('transformer','#0072B2','s')]:
                r=next(r for r in sm if r['panel']=='legacy' and r['dataset']=='hai' and r['model']==model and r['policy']==pol)
                keys=['f1','event_f1','affiliation_f1']
                ax.errorbar(range(3),[float(r[k+'_mean']) for k in keys],yerr=[float(r[k+'_sd']) for k in keys],
                            marker=marker,linestyle='none',color=color,capsize=2,label=LABELS[model])
            ax.set(xticks=range(3),xticklabels=['Point F1','Event F1','Affiliation'],xlim=(-.4,2.4),ylim=(0,1),title='Validation F1-max' if pol=='val_f1' else 'Validation-normal 5% target')
        axes[0].set_ylabel('Metric value');axes[1].legend(frameon=False,loc='center right')
        save(fig,'Fig4_hai_event_burden','results/model_policy_summary.csv','HAI fixed illustrative contrast, mean plus/minus sample SD over five seeds; metrics have different targets and are not interchangeable.')

        fig,ax=plt.subplots(figsize=(6.85,4.65),layout='constrained')
        x=np.arange(len(cm));left=np.zeros(len(cm))
        for key,label,color,hatch in [('coverage_absolute_share','Coverage','#dddddd','//'),('recall_absolute_share','In-event recall','#888888','..'),('fpr_absolute_share','Normal-point FPR','#0072B2','')]:
            values=100*np.array([float(r[key]) for r in cm])
            ax.barh(x,values,left=left,label=label,color=color,hatch=hatch,edgecolor='white',linewidth=.5)
            left+=values
        ax.set(yticks=x,yticklabels=[('Recent ' if r['panel']=='recent' else '')+r['dataset'].upper()+'/'+r['unit'] for r in cm],
               xlim=(0,100),xlabel='Share of total absolute symmetric allocations (%)')
        ax.invert_yaxis();ax.legend(loc='lower center',bbox_to_anchor=(.5,1.01),ncol=3,frameon=False,fontsize=11)
        save(fig,'Fig3_channel_allocations','results/unit_channel_summary.csv','Within-unit equal-seed-pair absolute allocation composition, averaged over all model pairs; descriptive, not causal shares or independent replicates.')
    (F/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print('Generated',len(provenance),'figures and all paper tables/macros')


if __name__=='__main__':main()
