"""Save public dataset documentation and pinned repository identities.

Downloads documentation only, never private data or manuscript content. Current
documentation commits identify the checked public source, not historical inputs.
"""
from pathlib import Path
import datetime
import hashlib
import json
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
SOURCES=[
 ('smd','NetManAIOps/OmniAnomaly','master','README.md'),
 ('skab','waico/SKAB','master','README.md'),
 ('skab_structure','waico/SKAB','master','data/README.md'),
 ('hai','icsdataset/hai','master','README.md'),
 ('psm','eBay/RANSynCoders','main','README.md'),
 ('telemanom','khundman/telemanom','master','README.md')]


def fetch(url):
    req=urllib.request.Request(url,headers={'User-Agent':'MTSAD-author-revision-source-check/1.0'})
    with urllib.request.urlopen(req,timeout=45) as response:return response.read()


def main():
    dest=ROOT/'public_sources';dest.mkdir(exist_ok=True)
    entries=[];commits={}
    for name,repo,branch,path in SOURCES:
        if repo not in commits:
            data=json.loads(fetch('https://api.github.com/repos/'+repo+'/commits/'+branch))
            commits[repo]=data['sha']
        commit=commits[repo]
        url=f'https://raw.githubusercontent.com/{repo}/{commit}/{path}'
        data=fetch(url);file=dest/(name+'_README.md');file.write_bytes(data)
        entries.append(dict(dataset=name,repository='https://github.com/'+repo,
                            documentation_commit=commit,path=path,url=url,
                            retrieved_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                            documentation_sha256=hashlib.sha256(data).hexdigest(),
                            local_path=str(file.relative_to(ROOT)),
                            historical_input_version_claimed=False))
    (dest/'manifest.json').write_text(json.dumps(entries,indent=2)+'\n')
    print('Saved',len(entries),'public dataset documentation records; no raw dataset downloaded')


if __name__=='__main__':main()
