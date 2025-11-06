# scripts/prepare_liar.py
from pathlib import Path
import json, re
import pandas as pd

RAW  = Path("data/raw/liar")
PROC = Path("data/processed"); PROC.mkdir(parents=True, exist_ok=True)

# Accept both LIAR-PLUS (*2) and classic names
CANDIDATES = {
    "train": [RAW/"train2.tsv", RAW/"train.tsv"],
    "val":   [RAW/"val2.tsv",   RAW/"valid.tsv"],
    "test":  [RAW/"test2.tsv",  RAW/"test.tsv"],
}

TRUE_SET  = {"half-true","mostly-true","true"}
FALSE_SET = {"pants-fire","false","barely-true"}

def first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None

def read_wide_tsv(path: Path) -> pd.DataFrame:
    """Robust TSV parser with diagnostics"""
    print(f"\nReading: {path}")
    
    # Check file size
    file_size = path.stat().st_size
    print(f"File size: {file_size} bytes")
    
    if file_size < 100:
        raise ValueError(f"{path} is too small ({file_size} bytes) - likely download error")
    
    # Show first line for debugging
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        first_line = f.readline()
        print(f"First line preview: {first_line[:150]}...")
    
    # Try different separators
    for sep in ['\t', ',', '|']:
        try:
            df = pd.read_csv(
                path, 
                sep=sep, 
                header=None, 
                dtype=str, 
                quoting=3,
                on_bad_lines='skip',
                engine='python',
                encoding='utf-8',
                encoding_errors='ignore'
            )
            
            if df.shape[1] >= 4:
                print(f"✓ Parsed with separator {repr(sep)}: {df.shape[0]} rows × {df.shape[1]} cols")
                return df
        except Exception as e:
            continue
    
    # Last resort: try with error handling
    try:
        df = pd.read_csv(
            path,
            sep='\t',
            header=None,
            dtype=str,
            engine='python',
            encoding='latin-1',
            on_bad_lines='skip'
        )
        if df.shape[1] >= 4:
            print(f"✓ Parsed with latin-1 encoding: {df.shape}")
            return df
    except Exception as e:
        pass
    
    raise ValueError(
        f"{path} cannot be parsed correctly.\n"
        f"Expected at least 4 columns, but got {df.shape[1] if 'df' in locals() else 'unknown'}.\n"
        f"First line: {first_line[:200]}"
    )

def map_label(lbl: str) -> str | None:
    if not isinstance(lbl, str): 
        return None
    l = lbl.strip().lower()
    if l in TRUE_SET:  
        return "true"
    if l in FALSE_SET: 
        return "false"
    return None

def norm_id(json_file_id: str) -> str:
    if not isinstance(json_file_id, str): 
        return ""
    return re.sub(r"\.json$", "", json_file_id.strip(), flags=re.I)

def convert(split: str, src: Path, out: Path) -> int:
    df = read_wide_tsv(src)
    
    # LIAR-PLUS format (14 columns):
    # 0: ID, 1: json_file_id, 2: label, 3: statement, 4: subjects, 5: speaker, 
    # 6: job, 7: state, 8: party, 9: barely_true_counts, 10: false_counts,
    # 11: half_true_counts, 12: mostly_true_counts, 13: pants_on_fire_counts
    
    if df.shape[1] < 4:
        raise ValueError(f"{src}: Need at least 4 columns, got {df.shape[1]}")
    
    # Extract columns (adjust indices if needed)
    jfid  = df.iloc[:, 1]  # json_file_id
    label = df.iloc[:, 2]  # 6-way label
    claim = df.iloc[:, 3]  # claim text

    kept = 0
    skipped = 0
    
    with out.open("w", encoding="utf-8") as w:
        for idx, (jf, lab, cl) in enumerate(zip(jfid, label, claim)):
            _id = norm_id(jf)
            _y  = map_label(lab)
            _x  = (cl or "").strip() if isinstance(cl, str) else ""
            
            if not _id or not _y or not _x:
                skipped += 1
                continue
                
            w.write(json.dumps({
                "id": _id, 
                "text": _x, 
                "label": _y
            }, ensure_ascii=False) + "\n")
            kept += 1
    
    print(f"  ✓ {split}: kept {kept}, skipped {skipped} → {out}")
    return kept

if __name__ == "__main__":
    outs = {
        "train": PROC/"liar_train.jsonl",
        "val":   PROC/"liar_val.jsonl",
        "test":  PROC/"liar_test.jsonl",
    }
    
    print("="*60)
    print("LIAR Dataset Preparation")
    print("="*60)
    
    total = 0
    for split in ("train","val","test"):
        src = first_existing(CANDIDATES[split])
        if not src:
            raise SystemExit(
                f"Missing TSV for {split}. Tried: "
                f"{', '.join(str(p) for p in CANDIDATES[split])}"
            )
        total += convert(split, src, outs[split])
    
    print("="*60)
    print(f"✓ Prepared LIAR JSONL in {PROC.resolve()}")
    print(f"✓ Total samples: {total}")
    print("="*60)