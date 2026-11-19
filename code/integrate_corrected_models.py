"""Reverify the corrected receipt and generate its supplementary table."""
from pathlib import Path
import csv
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    subprocess.run([sys.executable, str(ROOT/'received_corrected/verify_corrected.py'),
                    '--code', str(ROOT/'code')], cwd=ROOT, check=True)
    with (ROOT/'received_corrected/verified_summary.csv').open() as f:
        rows = list(csv.DictReader(f))
    lines = [r'\begin{tabular}{lllrr}', r'\toprule',
             r'Model & SMD entity & Calibration & Point F1 & Event F1 \\', r'\midrule']
    for model in ['MEMTO', 'CATCH']:
        for entity in ['machine-1-1', 'machine-3-1']:
            for policy in ['val_f1', 'normal_050']:
                r = next(r for r in rows if (r['model'], r['entity'], r['policy']) == (model, entity, policy))
                label = 'Injected F1-max' if policy == 'val_f1' else r'Normal 5\%'
                f1 = f"${float(r['f1_mean']):.4f}\\pm{float(r['f1_sd']):.4f}$"
                event = f"{float(r['event_f1_mean']):.4f}"
                lines.append(f'{model} & {entity} & {label} & {f1} & {event} '+r'\\')
    lines += [r'\bottomrule', r'\end{tabular}']
    (ROOT/'generated_tables').mkdir(exist_ok=True)
    (ROOT/'generated_tables/table_corrected_models.txt').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__': main()
