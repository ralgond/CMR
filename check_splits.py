"""확인: all_tracks와 test_tracks의 track_id 겹침 여부"""
from pathlib import Path
import pandas as pd

data_root = Path("./data")

for name in ["Track-Metadata", "Track-Embedding"]:
    d = data_root / name
    parquets = sorted(d.rglob("*.parquet"))
    
    splits = {}
    for p in parquets:
        split_name = p.stem.split("-")[0] + "_" + p.stem.split("-")[1] if "-" in p.stem else p.stem
        key = "all" if "all" in p.stem else "test"
        df = pd.read_parquet(p, columns=["track_id"])
        splits.setdefault(key, set()).update(df["track_id"].tolist())
    
    all_ids  = splits.get("all", set())
    test_ids = splits.get("test", set())
    overlap  = all_ids & test_ids
    
    print(f"\n{name}:")
    print(f"  all_tracks:  {len(all_ids):,} unique track_ids")
    print(f"  test_tracks: {len(test_ids):,} unique track_ids")
    print(f"  OVERLAP:     {len(overlap):,} track_ids in BOTH")
    print(f"  UNION:       {len(all_ids | test_ids):,} total unique")