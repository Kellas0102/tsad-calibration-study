"""Retrieve a fixed purposive public panel; keep failures, never replace cases."""
from pathlib import Path
import concurrent.futures
import datetime
import hashlib
import io
import json
import urllib.request
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent
PANEL = [
 ('timesnet', 'method', 'TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis',
  'https://arxiv.org/pdf/2210.02186v3'),
 ('memto', 'method', 'MEMTO: Memory-guided Transformer for Multivariate Time Series Anomaly Detection',
  'https://proceedings.neurips.cc/paper_files/paper/2023/file/b4c898eb1fb556b8d871fbe9ead92256-Paper-Conference.pdf'),
 ('catch', 'method', 'CATCH: Channel-Aware Multivariate Time Series Anomaly Detection via Frequency Patching',
  'https://arxiv.org/pdf/2410.12261'),
 ('omnianomaly', 'method', 'Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Network',
  'https://netman.aiops.org/wp-content/uploads/2019/08/OmniAnomaly_camera-ready.pdf'),
 ('rigorous', 'evaluation', 'Towards a Rigorous Evaluation of Time-Series Anomaly Detection',
  'https://cdn.aaai.org/ojs/20680/20680-13-24693-1-2-20220628.pdf'),
 ('affiliation', 'evaluation', 'Local Evaluation of Time Series Anomaly Detection Algorithms',
  'https://arxiv.org/pdf/2206.13167v1'),
 ('pate', 'evaluation', 'PATE: Proximity-Aware Time Series Anomaly Evaluation',
  'https://repository.tudelft.nl/file/File_c9886def-50ca-45bd-84e0-bded7e8c0cc8'),
 ('tsbuad', 'benchmark', 'TSB-UAD: An End-to-End Benchmark Suite for Univariate Time-Series Anomaly Detection',
  'https://www.vldb.org/pvldb/vol15/p1697-paparrizos.pdf'),
 ('timeeval', 'benchmark', 'TimeEval: A Benchmarking Toolkit for Time Series Anomaly Detection Algorithms',
  'https://www.vldb.org/pvldb/vol15/p3678-wenig.pdf'),
 ('tab', 'benchmark', 'TAB: Unified Benchmarking of Time Series Anomaly Detection Methods',
  'https://www.vldb.org/pvldb/vol18/p2775-hu.pdf'),
]


def get_source(record):
    key, group, title, url = record
    item = dict(case_id=key, group=group, expected_title=title, requested_url=url,
                retrieved_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                redistribution='not_authorized_by_this_download; local_audit_only')
    pdf = ROOT/'local_sources'/(key+'.pdf')
    text = ROOT/'local_sources'/(key+'.txt')
    try:
        assert not pdf.exists() and not text.exists(), 'refusing overwrite'
        request = urllib.request.Request(url, headers={'User-Agent': 'Academic evidence audit (read-only)'})
        with urllib.request.urlopen(request, timeout=90) as response:
            data = response.read(); item['resolved_url'] = response.url
        assert data.lstrip().startswith(b'%PDF'), 'response is not a PDF'
        pages = [page.extract_text() or '' for page in PdfReader(io.BytesIO(data)).pages]
        assert sum(map(len, pages)) > 1000, 'insufficient machine-readable text'
        pdf.write_bytes(data)
        text.write_text('\n'.join(f'\n=== PDF PAGE {i+1} ===\n'+s for i, s in enumerate(pages)))
        item.update(status='retrieved', pdf_sha256=hashlib.sha256(data).hexdigest(),
                    pages=len(pages), local_pdf=str(pdf.relative_to(ROOT)),
                    local_text=str(text.relative_to(ROOT)), title_identity='pending_reading')
    except Exception as error:
        item.update(status='not_retrieved', error=f'{type(error).__name__}: {error}')
    return item


def main():
    manifest = ROOT/'sources.json'
    assert not manifest.exists(), 'refusing to replace source history'
    (ROOT/'local_sources').mkdir(exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(get_source, PANEL))
    manifest.write_text(json.dumps(records, ensure_ascii=False, indent=2)+'\n')
    for record in records: print(record['case_id'], record['status'], record.get('pages', record.get('error')))


if __name__ == '__main__': main()
