"""Compile author-working PDFs, flatten table inputs, and audit LaTeX logs."""
from pathlib import Path
import json
import os
import re
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
M=ROOT/'manuscript'
TEX=Path(os.environ.get('TEXBIN', str(Path.home()/'texlive/2026/bin/universal-darwin')))


def executable(name):
    candidate=TEX/name
    if candidate.is_file():return str(candidate)
    resolved=shutil.which(name)
    if resolved:return resolved
    raise RuntimeError('Missing TeX executable '+name+'; install TeX Live/MiKTeX or set TEXBIN')


def run(cmd,cwd):
    env=os.environ.copy();env['PATH']=str(TEX)+os.pathsep+env.get('PATH','')
    result=subprocess.run(cmd,cwd=cwd,env=env,text=True,capture_output=True)
    if result.returncode:
        print(result.stdout[-5000:]);print(result.stderr[-2000:])
        raise RuntimeError('Build command failed: '+' '.join(cmd))
    return result.stdout


def main():
    reports=[]
    for stem in ['main','supplement']:
        for cmd in [[executable('pdflatex'),'-no-shell-escape','-interaction=nonstopmode','-halt-on-error',stem+'.tex'],
                    [executable('bibtex'),stem],
                    [executable('pdflatex'),'-no-shell-escape','-interaction=nonstopmode','-halt-on-error',stem+'.tex'],
                    [executable('pdflatex'),'-no-shell-escape','-interaction=nonstopmode','-halt-on-error',stem+'.tex']]:
            run(cmd,M)
        log=(M/(stem+'.log')).read_text(errors='replace')
        errors=[l for l in log.splitlines() if l.startswith('!')]
        undef=[l for l in log.splitlines() if 'undefined' in l.lower()]
        overfull=[l for l in log.splitlines() if 'Overfull' in l]
        pages=re.search(r'Output written on .*?\((\d+) pages',log)
        report={'document':stem,'errors':errors,'undefined':undef,'overfull':overfull,
                'pages':int(pages.group(1)) if pages else None,'pdf':str((M/(stem+'.pdf')).relative_to(ROOT))}
        reports.append(report)
        # A flat TeX source for subsequent submission-source packaging.
        text=(M/(stem+'.tex')).read_text()
        text=re.sub(r'\\input\{([^}]+)\}',lambda m:(M/m.group(1)).read_text() if (M/m.group(1)).exists() else (M/(m.group(1)+'.tex')).read_text(),text)
        (M/(stem+'_flat.tex')).write_text(text)
    (ROOT/'results/build_report.json').write_text(json.dumps(reports,indent=2)+'\n')
    print(json.dumps(reports,indent=2))
    if any(r['errors'] or r['undefined'] for r in reports):raise SystemExit(1)


if __name__=='__main__':main()
