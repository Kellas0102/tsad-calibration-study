"""Local consistency checks; never upgrades human verification status."""
from pathlib import Path
import csv
import hashlib
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    for script in ['evaluate_challenges.py', 'extract_public_cases.py']:
        subprocess.run([sys.executable, str(ROOT/script)], cwd=ROOT, check=True)
    challenge = json.loads((ROOT/'challenge_summary.json').read_text())
    expected = {'implemented_fault': (16, 16), 'missing_evidence': (8, 8),
                'mixed_fault_missing': (8, 0), 'unimplemented_scope_probe': (4, 0),
                'valid_control': (24, 24)}
    assert challenge['cases'] == 60
    for group, (n, matched) in expected.items():
        assert (challenge['groups'][group]['n'], challenge['groups'][group]['matched']) == (n, matched)
    with (ROOT/'challenge_results.csv').open() as f: cases = list(csv.DictReader(f))
    assert len({r['case_id'] for r in cases}) == 60
    public = json.loads((ROOT/'public_case_counts.json').read_text())
    with (ROOT/'public_field_extraction.csv').open() as f: fields = list(csv.DictReader(f))
    with (ROOT/'public_case_summary.csv').open() as f: reports = list(csv.DictReader(f))
    assert public['selected_reports'] == public['retrieved_reports'] == len(reports) == 10
    assert len(fields) == 60
    assert all(r['human_verification'] == 'pending' for r in fields+reports)
    assert all(r['whole_claim_verdict'].startswith('not_certified') for r in reports)
    assert all(sum(v.values()) == 10 for v in public['counts'].values())
    report = dict(local_consistency='passed', challenge_cases=60, public_reports=10,
                  field_records=60, known_mixed_fault_misses_retained=8,
                  known_scope_false_safe_outputs_retained=4,
                  human_validation_performed=False, full_paper_redistribution_authorized=False,
                  source_quote_checks='normalized source-page matches, not verification of every contextual interpretation',
                  manuscript_integration='proposed text only; previous candidate untouched')
    (ROOT/'validation_report.json').write_text(json.dumps(report, indent=2)+'\n')
    records = []
    for p in sorted(ROOT.rglob('*')):
        if not p.is_file() or '__pycache__' in p.parts: continue
        if p.name == 'artifact_manifest.json': continue
        records.append(dict(path=str(p.relative_to(ROOT)), sha256=hashlib.sha256(p.read_bytes()).hexdigest(), bytes=p.stat().st_size))
    (ROOT/'artifact_manifest.json').write_text(json.dumps({'status':'developer_led_research_extension', 'files':records}, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
