from pathlib import Path
import urllib.request

ROOT = Path("data/raw/liar"); ROOT.mkdir(parents=True, exist_ok=True)
BASE = "https://raw.githubusercontent.com/Tariq60/LIAR-PLUS/master/dataset/tsv/"
FILES = ["train2.tsv", "val2.tsv", "test2.tsv"]

for f in FILES:
    out = ROOT / f
    if out.exists():
        print("Exists:", out); continue
    url = BASE + f
    print("Downloading", url)
    urllib.request.urlretrieve(url, out)
print("Done.")
