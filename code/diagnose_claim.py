"""Generate a claim-relative evidence card; never an editorial acceptance score.

This interface separates document/manual gates from computed score diagnostics.
It fails closed on unknown facts and preserves an explicit explanation per gate.
"""
from pathlib import Path
import argparse
import json


GATES={
 'G1':('artifact_identity','Run, score, label and implementation identity'),
 'G2':('metric_semantics','Metric implementation and target compatibility'),
 'G3':('information_access','Training/calibration information and split independence'),
 'G4':('temporal_contract','Timeline alignment, trial boundaries and causal scope'),
 'G5':('calibration_scope','Common policy family and contrast envelope'),
 'G6':('uncertainty_design','Sampling unit, assumptions, intervals and multiplicity'),
 'G7':('external_validity','Population and implementation generalization'),
 'G8':('operational_utility','Alarm grouping, delay, workload and costs')}
STATES={'within_scope','not_evaluable','contradicted','not_applicable'}
REQUIRED={
 'score':{'G1','G2','G3','G4','G6'},
 'pipeline':{'G1','G2','G3','G4','G5','G6'},
 'event':{'G1','G2','G3','G4','G5','G6'},
 'deployment':set(GATES),
 'generalization':{'G1','G2','G3','G4','G5','G6','G7'}}


def diagnose(card):
    if not card.get('claim') or card.get('target') not in ['score','pipeline','event','deployment','generalization']:
        raise ValueError('Explicit claim and one declared target required')
    gates=card.get('gates',{})
    if not isinstance(gates,dict):raise ValueError('gates must be an object')
    unknown=set(gates)-set(GATES)
    if unknown:raise ValueError('Unknown gate identifiers: '+','.join(sorted(unknown)))
    outcomes=[]
    for gid,(field,question) in GATES.items():
        supplied=card.get('gates',{}).get(gid,{})
        status=supplied.get('status','not_evaluable')
        if status not in STATES:raise ValueError('Invalid gate state')
        evidence=supplied.get('evidence',[])
        if not isinstance(evidence,list) or not all(isinstance(x,str) and x.strip() for x in evidence):
            raise ValueError('Evidence must be a list of nonempty locator strings')
        explanation=supplied.get('explanation','Required evidence has not been supplied.')
        if status in ['within_scope','contradicted'] and not evidence:
            status='not_evaluable';explanation='A verdict was supplied without an evidence locator.'
        if status=='not_applicable' and not supplied.get('explanation'):
            status='not_evaluable';explanation='Not-applicable status needs an explicit scope rationale.'
        if supplied.get('status')=='not_applicable' and gid in REQUIRED[card['target']]:
            status='not_evaluable';explanation='This gate is required for the declared target; narrow the target rather than bypass its evidence requirements.'
        outcomes.append(dict(gate=gid,field=field,question=question,status=status,
                             evidence=evidence,explanation=explanation,
                             human_verification=supplied.get('human_verification','pending')))
    relevant=[x['status'] for x in outcomes if x['status']!='not_applicable']
    overall=('contradicted' if 'contradicted' in relevant else 'not_evaluable' if not relevant or 'not_evaluable' in relevant else 'within_scope')
    return {'claim_id':card.get('claim_id'),'claim':card['claim'],'target':card['target'],
             'overall':overall,'gates':outcomes,'certificate':False,
             'automation_coverage':'aggregation of supplied gate states, not independent adjudication',
             'warning':'Only the declared claim and supplied evidence were assessed. Human verification remains required.'}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('card',type=Path);p.add_argument('--output',type=Path)
    args=p.parse_args();result=diagnose(json.loads(args.card.read_text()))
    text=json.dumps(result,ensure_ascii=False,indent=2)+'\n'
    if args.output:
        if args.output.exists():raise SystemExit('Refusing implicit output overwrite')
        args.output.write_text(text)
    else:print(text)


if __name__=='__main__':main()
