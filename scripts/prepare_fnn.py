import json
import re
from pathlib import Path
import pandas as pd

RAW_MIN = Path("data/raw/fakenewsnet_min")
PROC = Path("data/processed")
PROC.mkdir(parents=True, exist_ok=True)

def clean(s): 
    """Clean text by normalizing whitespace"""
    return re.sub(r"\s+", " ", (s or "").strip())

def load_isot_dataset():
    """Load ISOT Fake News Dataset (Fake.csv and True.csv)"""
    fake_path = RAW_MIN / "Fake.csv"
    true_path = RAW_MIN / "True.csv"
    
    if not fake_path.exists() or not true_path.exists():
        print("✗ ERROR: Dataset files not found!")
        print(f"\nExpected files in: {RAW_MIN.absolute()}")
        print("  - Fake.csv")
        print("  - True.csv")
        print("\nRun 'python scripts/download_fnn.py' first to download the data.")
        return pd.DataFrame()
    
    rows = []
    
    # Load fake news
    print(f"Loading {fake_path.name}...")
    try:
        df_fake = pd.read_csv(fake_path)
        print(f"  Found {len(df_fake)} fake news articles")
        
        for _, row in df_fake.iterrows():
            title = clean(str(row.get('title', '')))
            text = clean(str(row.get('text', '')))
            
            # Combine title and text
            full_text = f"{title}. {text}" if title and text else (title or text)
            
            if len(full_text) > 50:  # Filter very short articles
                rows.append({
                    "text": full_text,
                    "label": "false"
                })
    except Exception as e:
        print(f"  ✗ Error loading Fake.csv: {e}")
        return pd.DataFrame()
    
    # Load real news
    print(f"Loading {true_path.name}...")
    try:
        df_true = pd.read_csv(true_path)
        print(f"  Found {len(df_true)} real news articles")
        
        for _, row in df_true.iterrows():
            title = clean(str(row.get('title', '')))
            text = clean(str(row.get('text', '')))
            
            # Combine title and text
            full_text = f"{title}. {text}" if title and text else (title or text)
            
            if len(full_text) > 50:  # Filter very short articles
                rows.append({
                    "text": full_text,
                    "label": "true"
                })
    except Exception as e:
        print(f"  ✗ Error loading True.csv: {e}")
        return pd.DataFrame()
    
    df = pd.DataFrame(rows)
    print(f"\n✓ Total loaded: {len(df)} articles")
    print(f"  - Fake: {(df['label'] == 'false').sum():>5} ({(df['label'] == 'false').sum()/len(df)*100:.1f}%)")
    print(f"  - Real: {(df['label'] == 'true').sum():>5} ({(df['label'] == 'true').sum()/len(df)*100:.1f}%)")
    
    return df

def split(df, train_ratio=0.8, val_ratio=0.1):
    """Split data into train/val/test sets"""
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train = df.iloc[:n_train]
    val = df.iloc[n_train:n_train + n_val]
    test = df.iloc[n_train + n_val:]
    
    return train, val, test

def dump(df, path):
    """Save dataset to JSONL format"""
    with open(path, "w", encoding="utf-8") as w:
        for i, row in df.iterrows():
            w.write(json.dumps({
                "id": f"isot_{i}",
                "text": row["text"],
                "label": row["label"]
            }, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    print("=" * 60)
    print("ISOT Fake News Dataset Preparation")
    print("=" * 60)
    print()
    
    # Load the dataset
    df = load_isot_dataset()
    
    if df.empty:
        raise SystemExit(1)
    
    # Split into train/val/test
    print("\nSplitting dataset (80% train, 10% val, 10% test)...")
    train, val, test = split(df)
    
    # Save to JSONL files
    print("Saving processed data...")
    dump(train, PROC / "fnn_train.jsonl")
    dump(val, PROC / "fnn_val.jsonl")
    dump(test, PROC / "fnn_test.jsonl")
    
    print("\n" + "=" * 60)
    print("✓ SUCCESS: Dataset prepared!")
    print("=" * 60)
    print(f"\nOutput files in {PROC}:")
    print(f"  Train: {len(train):>6} samples -> fnn_train.jsonl")
    print(f"  Val:   {len(val):>6} samples -> fnn_val.jsonl")
    print(f"  Test:  {len(test):>6} samples -> fnn_test.jsonl")
    
    print(f"\nTraining set distribution:")
    print(f"  Fake: {(train['label'] == 'false').sum():>6} ({(train['label'] == 'false').sum()/len(train)*100:.1f}%)")
    print(f"  Real: {(train['label'] == 'true').sum():>6} ({(train['label'] == 'true').sum()/len(train)*100:.1f}%)")
    print()