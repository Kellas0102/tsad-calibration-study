"""Run ON THE ORIGINAL TRAINING DEVICE. Standard library; no torch import.

Reads a bounded, explicit project root. Exports code/metadata identities and data
file hashes, not raw feature arrays, credentials, home directories or model
weights. No training, inference, package installation, network or uploads.

Example (PowerShell):
  python export_training_evidence.py --project-root "C:\path\to\tsad-paper" \
    --output "D:\exports\MTSAD_training_evidence" --dataset SKAB --dataset HAI

An output produced today documents CURRENT files. It cannot prove that these
files were used by historical runs unless historical hashes/logs independently
bind them. That distinction is preserved in the exported report.
"""
from pathlib import Path
import argparse
import csv
import datetime
import hashlib
import importlib.metadata
import json
import platform
import zipfile

CODE_NAMES=('common.py','data.py','evaluate.py','losses.py','models.py','train.py',
            'models_recent.py','run_recent.py','collect_recent.py')
FIELDS={'run_id','dataset','model','seed','machine','epochs','patience','batch_size','lr',
        'weight_decay','mask_ratio','train_anomaly_ratio','max_train_windows','win_len','stride',
        'implementation','code_fingerprint','timestamp','threshold','threshold_source','f1','auc',
        'n_test_points','n_test_anomaly','test_anomaly_ratio','best_epoch','epochs_run','params',
        'train_time','wall_time_s','model_kwargs','memto_phase1','device','best_val_loss'}
LOG_FIELDS={'epoch','train_loss','val_loss','lr','epoch_time'}
DATA_EXT={'.csv','.txt','.npy','.npz','.parquet','.json'}


def hash_file(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def inside(root,p):
    try:p.resolve().relative_to(root.resolve());return True
    except ValueError:return False


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--project-root',required=True,type=Path)
    a.add_argument('--output',required=True,type=Path)
    a.add_argument('--dataset',action='append',choices=['SKAB','HAI','SMD'],required=True)
    a.add_argument('--max-data-files',type=int,default=1000)
    args=a.parse_args();root=args.project_root.resolve();out=args.output.resolve()
    if not root.is_dir() or not (root/'code').is_dir():raise SystemExit('Explicit training project root with code/ required')
    if out.exists():raise SystemExit('Output exists; refusing overwrite')
    if inside(root/'data',out) or inside(root/'results',out):raise SystemExit('Output must be outside data/ and results/')
    out.mkdir(parents=True)
    report={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'scope':'current_training_device_files_not_proof_of_historical_usage',
            'historical_usage_verified':False,'raw_data_copied':False,'weights_copied':False,
            'model_code_executed':False,'network_used':False,
            'python':platform.python_version(),'platform':platform.system(),
            'packages':{},'code':[],'runs':[],'data_files':[],'missing':[]}
    for package in ['numpy','torch','scikit-learn','pandas','scipy']:
        try:report['packages'][package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:report['packages'][package]=None
    for name in CODE_NAMES:
        p=root/'code'/name
        if not p.is_file() or p.is_symlink():report['missing'].append('code/'+name);continue
        if p.stat().st_size>2*1024*1024:raise ValueError('Unexpectedly large source file')
        text=p.read_bytes();dest=out/'code'/name;dest.parent.mkdir(exist_ok=True);dest.write_bytes(text)
        report['code'].append({'path':'code/'+name,'sha256':hash_file(p),'bytes':p.stat().st_size})
    # Only supplied recent method/dataset result metadata; no recursive user files.
    runroot=root/'results/runs'
    if runroot.is_dir():
        for folder in sorted(runroot.iterdir()):
            if not folder.is_dir() or folder.is_symlink():continue
            p=folder/'metrics.json'
            if not p.is_file() or p.stat().st_size>2*1024*1024:continue
            m=json.loads(p.read_text(encoding='utf-8-sig'))
            if str(m.get('dataset','')).upper() not in args.dataset:continue
            if m.get('model') not in ['timesnet','memto','catch']:continue
            selected={k:v for k,v in m.items() if k in FIELDS}
            record={'run_id':folder.name,'metadata':selected,'original_metrics_sha256':hash_file(p),'logs':[]}
            for lp in sorted((folder/'logs').glob('*.csv')) if (folder/'logs').is_dir() else []:
                if lp.is_symlink() or lp.stat().st_size>5*1024*1024:continue
                with lp.open(encoding='utf-8-sig',newline='') as f:
                    reader=csv.DictReader(f);names=[n for n in (reader.fieldnames or []) if n in LOG_FIELDS]
                    rows=[{k:r[k] for k in names} for r in reader]
                if not names:continue
                target=out/'run_logs'/folder.name/lp.name;target.parent.mkdir(parents=True,exist_ok=True)
                with target.open('w',newline='',encoding='utf-8') as f:
                    writer=csv.DictWriter(f,fieldnames=names);writer.writeheader();writer.writerows(rows)
                record['logs'].append({'path':str(target.relative_to(out)),'original_sha256':hash_file(lp)})
            report['runs'].append(record)
    for ds in sorted(set(args.dataset)):
        base=root/'data'/ds
        if not base.is_dir():report['missing'].append('data/'+ds);continue
        for p in sorted(base.rglob('*')):
            if not p.is_file() or p.is_symlink() or not inside(base,p) or p.suffix.lower() not in DATA_EXT:continue
            if len(report['data_files'])>=args.max_data_files:raise SystemExit('File budget exceeded; inspect explicitly before raising limit')
            report['data_files'].append({'path':str(p.relative_to(root)),
                                         'bytes':p.stat().st_size,'sha256':hash_file(p),
                                         'historical_usage_verified':False})
    (out/'training_device_evidence.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    notice='Review locally before sharing: source code may contain proprietary material. Raw data and credentials were not requested or copied. Current hashes alone do not establish historical file use.\n'
    (out/'REVIEW_BEFORE_SHARING.txt').write_text(notice,encoding='utf-8')
    archive=out.with_suffix('.zip')
    if archive.exists():raise SystemExit('Archive path already exists; refusing overwrite')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file():z.write(p,str(p.relative_to(out)))
    print('Created',archive)
    print('Code files',len(report['code']),'run metadata',len(report['runs']),'data hashes',len(report['data_files']))
    print('SHA256',hash_file(archive))
    print(notice)


if __name__=='__main__':main()
