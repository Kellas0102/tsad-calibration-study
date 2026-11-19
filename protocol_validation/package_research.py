"""Package research outputs without redistributing paper PDFs or full extracted text."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parent


def main():
    archive = ROOT.parent/'DMKD_Protocol_Validation_Research_20261112.zip'
    assert not archive.exists(), 'refusing overwrite'
    files = []
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file(): continue
        rel = path.relative_to(ROOT)
        if 'local_sources' in rel.parts or '__pycache__' in rel.parts: continue
        if path.name.startswith('selected_pages_'): continue
        if path.suffix in ['.pdf', '.pyc']: continue
        files.append(path)
    manifest = [dict(path=str(p.relative_to(ROOT)), sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                     bytes=p.stat().st_size) for p in files]
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path in files: z.write(path, str(Path(ROOT.name)/path.relative_to(ROOT)))
        z.writestr(str(Path(ROOT.name)/'PACKAGE_MANIFEST.json'), json.dumps(manifest, indent=2)+'\n')
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for record in manifest:
            assert hashlib.sha256(z.read(str(Path(ROOT.name)/record['path']))).hexdigest() == record['sha256']
        assert not any('/local_sources/' in name or name.endswith('.pdf') or '/selected_pages_' in name for name in z.namelist())
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_suffix('.zip.sha256').write_text(digest+'  '+archive.name+'\n')
    print('Packaged', len(files), 'files;', archive.stat().st_size, 'bytes;', digest)


if __name__ == '__main__': main()
