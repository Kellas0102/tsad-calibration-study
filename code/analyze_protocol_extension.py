"""Retain v1 challenge findings; report v2 replay as development regression only.

Public extraction is a recorded single-assistant reading, not rerun automatically.
This script validates its stored tables/locators and generates paper tables. Full
report PDFs are not redistributed, so contextual source verification remains manual.
"""
from pathlib import Path
import collections
import csv
import hashlib
import json
import subprocess
import sys

from diagnostics import claim_assessment

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT/'protocol_validation'


def load(path): return json.loads(path.read_text())


def csv_rows(path):
    with path.open() as f: return list(csv.DictReader(f))


def optional(folder, name):
    path = folder/(name+'.json')
    return load(path) if path.exists() else None


def assess_v2(folder):
    claim, metric, cal, temporal, design, identity = [optional(folder, n) for n in
        ['claim','metric','calibration','temporal','design','identity']]
    r = dict(target=claim['target'], claim_strict=claim['strict'],
             causal_online=claim['causal_online'], unit='synthetic_timeline')
    # V2 adapter explicitly carries the structured inference request. It does not
    # infer that request from natural language, and does not prove/refute equivalence.
    if claim.get('equivalence_asserted') is True: r['equivalence_asserted'] = True
    if metric:
        r['metric'] = 'PA-F1' if metric['postprocessing']=='point_adjustment' else 'strict_F1'
    if cal: r['calibration'] = 'test_oracle' if cal['selection_data']=='test_labels' else 'validation'
    if temporal:
        r.update(timeline_causal=temporal['max_future_offset'] <= 0,
                 event_boundaries_known=temporal['event_boundaries_known'])
    if design: r['independence_verified'] = design['declared_independence']
    if identity:
        path = folder/identity['file']
        assert path.resolve().is_relative_to(folder.resolve())
        mismatch = hashlib.sha256(path.read_bytes()).hexdigest() != identity['expected_sha256']
        r.update(snapshot_verified=not mismatch, snapshot_hash_mismatch=mismatch)
    return claim_assessment(r)


def main():
    # The v1 executable and pre-evaluation oracle remain unchanged.
    for record in load(EXT/'freeze_manifest.json')['files']:
        assert hashlib.sha256((EXT/record['frozen']).read_bytes()).hexdigest() == record['sha256']
    subprocess.run([sys.executable, str(EXT/'evaluate_challenges.py')], cwd=EXT, check=True)
    v1 = load(EXT/'challenge_summary.json')
    original = {r['case_id']:r for r in csv_rows(EXT/'challenge_results.csv')}
    rows = []
    for folder in sorted((EXT/'challenge_bundles').iterdir()):
        if not folder.is_dir(): continue
        old = original[folder.name]; new = assess_v2(folder)
        rows.append(dict(case_id=folder.name, group=old['group'],
                         v1_status=old['predicted'], original_semantic_expectation=old['expected'],
                         v2_status=new['status'], missing_fields=';'.join(new['missing_fields']),
                         contradictions=';'.join(new['contradictions']),
                         unsupported_inferences=';'.join(new['unsupported_inferences']),
                         interpretation='same-case development regression; no new independent validation'))
    with (ROOT/'results/protocol_v2_development_replay.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    groups = {}
    for group in sorted({r['group'] for r in rows}):
        rs=[r for r in rows if r['group']==group]
        groups[group]=dict(n=len(rs), v2_statuses=dict(collections.Counter(r['v2_status'] for r in rs)))
    assert all(r['v2_status']=='contradicted' and r['missing_fields'] for r in rows if r['group']=='mixed_fault_missing')
    assert all(r['v2_status']=='not_evaluable' and r['unsupported_inferences'] for r in rows if r['group']=='unimplemented_scope_probe')
    # Public export checks do not pretend to re-open copyrighted source PDFs.
    fields=csv_rows(EXT/'public_field_extraction.csv');cases=csv_rows(EXT/'public_case_summary.csv')
    recorded={r['path']:r['sha256'] for r in load(EXT/'artifact_manifest.json')['files']}
    for name in ['public_field_extraction.csv','public_case_summary.csv','sources.json','retrieval_followup.json']:
        assert hashlib.sha256((EXT/name).read_bytes()).hexdigest()==recorded[name]
    assert len(fields)==60 and len(cases)==10
    assert len({(r['case_id'],r['field']) for r in fields})==60
    assert all(r['human_verification']=='pending' for r in fields+cases)
    counts={field:dict(collections.Counter(r['status'] for r in fields if r['field']==field))
            for field in ['metric','calibration','temporal','replication','artifact_pointer','utility']}
    assert counts==load(EXT/'public_case_counts.json')['counts']
    sources=load(EXT/'sources.json')+load(EXT/'retrieval_followup.json')
    hashes={r['pdf_sha256'] for r in sources if r['status']=='retrieved'}
    assert all(r['source_sha256'] in hashes for r in fields+cases)
    assert all(r['pdf_page'] and r['quote'] for r in fields if r['status']=='explicit')
    report=dict(v1_frozen_challenges=v1, v2_development_replay=groups,
                known_mixed_contradictions_retained=8, unsupported_scope_probes_deferred=4,
                deployment_controls_now_deferred=4, public_reports=10, public_fields=60,
                public_field_counts=counts,
                public_check_scope='stored extraction consistency; contextual PDF checks recorded in research stage, not independently repeated here',
                new_independent_validation=False, human_verification='pending')
    (ROOT/'results/protocol_extension_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    labels={'implemented_fault':'Isolated implemented fault','valid_control':'Valid listed-rule control',
            'missing_evidence':'Missing evidence only','mixed_fault_missing':'Known fault + missing input',
            'unimplemented_scope_probe':'Unsupported inference probe'}
    lines=[r'\begin{tabular}{lrrl}',r'\toprule',r'Challenge group & $n$ & V1 matches & V1 output \\',r'\midrule']
    for group in ['implemented_fault','valid_control','missing_evidence','mixed_fault_missing','unimplemented_scope_probe']:
        g=v1['groups'][group];state=next(iter(g['predictions']))
        text={'contradicted':'Contradicted','within_scope':'Within listed scope','not_evaluable':'Not evaluable'}[state]
        lines.append(f"{labels[group]} & {g['n']} & {g['matched']} & {text} "+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}']
    (ROOT/'generated_tables').mkdir(exist_ok=True)
    (ROOT/'generated_tables/table_protocol_challenges.txt').write_text('\n'.join(lines)+'\n')
    lines=[r'\begin{tabular}{lrrrr}',r'\toprule',r'Field & Explicit & Partial & Unresolved & N/A \\',r'\midrule']
    labels={'metric':'Metric semantics','calibration':'Calibration/integration','temporal':'Temporal/aggregation scope',
            'replication':'Replication/sampling description','artifact_pointer':'Artifact access pointer','utility':'Utility of selected statement'}
    for field,count in counts.items():
        lines.append(labels[field]+' & '+' & '.join(str(count.get(s,0)) for s in ['explicit','partial','unresolved','not_applicable'])+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}']
    (ROOT/'generated_tables/table_protocol_public_fields.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
