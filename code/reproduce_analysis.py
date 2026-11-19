"""Recompute public score-level analyses and tests; no detector training or network access."""
from pathlib import Path
import argparse, subprocess, sys
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser();p.add_argument('--reuse-primary-tables',action='store_true');a=p.parse_args()
    commands=[]
    if not a.reuse_primary_tables:
        commands += [['integrate_evidence.py'],['code/integrate_corrected_models.py'],['code/reanalyze.py'],['code/secondary_analysis.py']]
    commands += [['code/analyze_protocol_extension.py'],['-m','unittest','discover','-s','code','-p','test_*.py'],['-m','unittest','test_integration']]
    for command in commands: subprocess.run([sys.executable]+command,cwd=ROOT,check=True)
    subprocess.run([sys.executable,'-m','unittest','discover'],cwd=ROOT/'vendor/affiliation-metrics-py',check=True)
    print('Public score-level analysis and tests completed.')
if __name__=='__main__':main()
