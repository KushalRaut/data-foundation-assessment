"""Retrieve public FIFA reports and retain their source URLs and match headers."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import urllib.request, json, re
from pypdf import PdfReader
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'raw'/'fifa'; OUT.mkdir(exist_ok=True)
def get(r):
    url=f'https://fdp.fifa.org/assetspublic/ce281/r{r}/pdf/FullTimeMatchReport-English.pdf'
    f=OUT/f'{r}.pdf'
    try:
        if not f.exists(): f.write_bytes(urllib.request.urlopen(url,timeout=15).read())
        text=PdfReader(f).pages[0].extract_text()
        (OUT/f'{r}.txt').write_text(text)
        head=text.split('MATCH REPORT')[0].replace('\xa0',' ')
        return {'resource':r,'url':url,'header':head}
    except Exception as e: return {'resource':r,'url':url,'error':str(e)}
if __name__=='__main__':
    with ThreadPoolExecutor(max_workers=5) as ex: rows=list(ex.map(get,range(12449,12553)))
    (OUT/'index.json').write_text(json.dumps(rows,indent=2))
    good=[x for x in rows if 'header' in x]
    print('Reports',len(good));print(json.dumps(good[-12:],indent=2))
