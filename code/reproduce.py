"""One-command local regeneration. Frozen inputs remain read-only."""
from pathlib import Path
import argparse
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--skip-analysis',action='store_true',help='Reuse already generated score-analysis tables; rebuild presentation and QA only')
    args=p.parse_args()
    if not args.skip_analysis:
        subprocess.run([sys.executable,str(ROOT/'integrate_evidence.py')],cwd=ROOT,check=True)
        subprocess.run([sys.executable,str(ROOT/'code/integrate_corrected_models.py')],cwd=ROOT,check=True)
    scripts=[] if args.skip_analysis else ['reanalyze.py','secondary_analysis.py']
    scripts+=['analyze_protocol_extension.py']
    scripts+=['audit_recent_literature.py','render_results.py','build_paper.py','validate_revision.py','record_provenance.py']
    for name in scripts:
        print('Running',name,flush=True)
        subprocess.run([sys.executable,str(ROOT/'code'/name)],cwd=ROOT,check=True)


if __name__=='__main__':main()
