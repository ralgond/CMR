"""
快速验证：GT tracks 有多少在 Track-Metadata 里？
python check_coverage.py --data-root ./data --n-samples 200
"""
import argparse
import json
import sys
import re
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from music_rec_baseline import load_all_data

UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I
)

def _to_dict(v):
    if isinstance(v, dict): return v
    if isinstance(v, str):
        try: return json.loads(v)
        except: return {}
    return {}

def _to_list(v):
    if isinstance(v, (list, np.ndarray)): return list(v)
    return []

def safe_join(v):
    if isinstance(v, (list, np.ndarray)):
        return " | ".join(str(x) for x in v if x)
    return str(v or "")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("./data"))
    parser.add_argument("--n-samples", type=int, default=200)
    args = parser.parse_args()

    print("[LOAD] Loading data...")
    dfs = load_all_data(args.data_root)

    track_meta = dfs["track_meta"]
    sessions = dfs.get("sessions")
    if sessions is None:
        sessions = dfs.get("blind")
    if sessions is None:
        print("ERROR: no sessions/blind data"); return

    track_ids_in_catalog = set(track_meta["track_id"].tolist())
    meta_idx = track_meta.set_index("track_id")

    print(f"Catalog size: {len(track_ids_in_catalog):,} tracks")
    print(f"Sessions: {len(sessions):,}")

    sample = sessions.sample(min(args.n_samples, len(sessions)), random_state=42)

    in_catalog    = 0
    not_in_catalog = 0
    no_gt         = 0
    not_in_examples = []

    for _, row in sample.iterrows():
        turns = _to_list(row.get("conversations"))

        # Find last music turn as GT
        gt = None
        for t in turns:
            t = _to_dict(t)
            if t.get("role") == "music":
                c = str(t.get("content", "") or "").strip()
                if UUID_RE.match(c):
                    gt = c

        if not gt:
            no_gt += 1
            continue

        if gt in track_ids_in_catalog:
            in_catalog += 1
        else:
            not_in_catalog += 1
            # Collect user turns for context
            user_msgs = [_to_dict(t).get("content","") for t in turns
                        if _to_dict(t).get("role") == "user"]
            conv_goal = _to_dict(row.get("conversation_goal"))
            not_in_examples.append({
                "gt": gt,
                "listener_goal": conv_goal.get("listener_goal",""),
                "last_user": user_msgs[-1][:80] if user_msgs else "",
            })

    total_with_gt = in_catalog + not_in_catalog
    print(f"\n=== GT Track Coverage ===")
    print(f"  Sessions with GT:     {total_with_gt}")
    print(f"  GT in catalog:        {in_catalog} ({100*in_catalog/max(total_with_gt,1):.1f}%)")
    print(f"  GT NOT in catalog:    {not_in_catalog} ({100*not_in_catalog/max(total_with_gt,1):.1f}%)")
    print(f"  No GT found:          {no_gt}")
    print()

    if not_in_catalog > 0:
        print(f"=== Sample GT tracks NOT in catalog (first 5) ===")
        for ex in not_in_examples[:5]:
            print(f"  GT id:         {ex['gt']}")
            print(f"  listener_goal: {ex['listener_goal'][:80]}")
            print(f"  last user:     {ex['last_user']}")
            print()

    # Also check: are the GT tracks in Track-Embedding?
    if "track_emb" in dfs:
        emb_ids = set(dfs["track_emb"]["track_id"].tolist())
        print(f"=== Embedding coverage ===")
        print(f"  Track-Embedding size: {len(emb_ids):,}")

        # Re-scan to check embedding coverage
        in_emb = 0
        not_in_emb = 0
        for _, row in sample.iterrows():
            turns = _to_list(row.get("conversations"))
            gt = None
            for t in turns:
                t = _to_dict(t)
                if t.get("role") == "music":
                    c = str(t.get("content","") or "").strip()
                    if UUID_RE.match(c): gt = c
            if not gt: continue
            if gt in emb_ids: in_emb += 1
            else: not_in_emb += 1

        print(f"  GT in embedding:     {in_emb} ({100*in_emb/max(in_emb+not_in_emb,1):.1f}%)")
        print(f"  GT NOT in embedding: {not_in_emb} ({100*not_in_emb/max(in_emb+not_in_emb,1):.1f}%)")

if __name__ == "__main__":
    main()