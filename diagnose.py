"""
诊断脚本：逐层测量每个模块的真实召回率
用法：python diagnose.py --data-root ./data --n-samples 100

输出每个阶段的 recall@K，定位瓶颈。
"""

import argparse
import json
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

# ── 把 baseline 加入 path ────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from music_rec_baseline import (
    load_all_data, MusicRecPipeline, ParsedIntent,
    BM25Retriever, _build_conversation_history,
    _extract_positive_ids_from_history,
    TOP_K_RETRIEVE,
)


def extract_gt_track(turns) -> str:
    """从 training session 提取最后一个 music turn 作为 ground truth。"""
    import re
    UUID_RE = re.compile(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I
    )
    def _to_dict(v):
        if isinstance(v, dict): return v
        if isinstance(v, str):
            try: return json.loads(v)
            except: return {}
        return {}

    last_gt = None
    for t in turns:
        t = _to_dict(t)
        if t.get("role") == "music":
            c = str(t.get("content", "") or "").strip()
            if UUID_RE.match(c):
                last_gt = c
    return last_gt


def diagnose(pipeline: MusicRecPipeline, sessions_df: pd.DataFrame,
             n_samples: int = 100, verbose: bool = False):

    def _to_dict(v):
        if isinstance(v, dict): return v
        if isinstance(v, str):
            try: return json.loads(v)
            except: return {}
        return {}

    def _to_list(v):
        if isinstance(v, (list, np.ndarray)): return list(v)
        return []

    results = []
    skipped = 0
    sample = sessions_df.sample(min(n_samples, len(sessions_df)), random_state=42)

    for _, row in sample.iterrows():
        turns        = _to_list(row.get("conversations"))
        conv_goal    = _to_dict(row.get("conversation_goal"))
        user_profile = _to_dict(row.get("user_profile"))
        user_id      = str(row.get("user_id", "") or "")

        # Ground truth = last music turn in this session
        gt = extract_gt_track(turns)
        if not gt:
            skipped += 1
            continue

        # Use all turns EXCEPT the last music turn as context
        # (simulate predicting the next recommendation)
        context_turns = []
        for t in turns:
            t = _to_dict(t)
            if t.get("role") == "music" and str(t.get("content","")).strip() == gt:
                break
            context_turns.append(t)

        intent_history, _ = _build_conversation_history(context_turns)
        played_ids = _extract_positive_ids_from_history(context_turns)
        listener_goal = str(conv_goal.get("listener_goal") or "").strip()
        user_culture  = str(user_profile.get("preferred_musical_culture") or "").strip()

        if not intent_history:
            skipped += 1
            continue

        # ── Step 1: Intent parsing (use fallback, no API needed) ──────────────
        from music_rec_baseline import _fallback_parse, BM25Retriever
        fake_conv = [{"role": m["role"], "content": m["content"]}
                     for m in intent_history]
        intent = _fallback_parse(fake_conv)
        if played_track_ids := played_ids:
            intent.positive_track_ids = list(dict.fromkeys(
                intent.positive_track_ids + played_track_ids
            ))
        if listener_goal:
            intent.listener_goal = listener_goal
        if user_culture:
            intent.user_culture = user_culture
        if not intent.retrieval_mode:
            intent.retrieval_mode = BM25Retriever._classify_mode(intent)

        # ── Step 2: Filter ────────────────────────────────────────────────────
        filtered = pipeline.filter.apply(intent)
        filter_active = any([intent.year_min, intent.year_max,
                              intent.popularity_min, intent.popularity_max])
        if filter_active and len(filtered) < len(pipeline.filter.df):
            cids = filtered["track_id"].tolist()
        else:
            cids = None

        filter_pass = (cids is None) or (gt in set(cids))

        # ── Step 3: BM25 recall ───────────────────────────────────────────────
        bm25_scores = pipeline.bm25.retrieve(intent, candidate_ids=cids,
                                              top_k=TOP_K_RETRIEVE)
        bm25_hit = gt in bm25_scores

        # ── Step 4: Dense recall ──────────────────────────────────────────────
        # Use listener_goal or semantic_query as dense query
        parts = [x for x in [intent.listener_goal, intent.semantic_query,
                               intent.user_culture] if x]
        dense_query = " ".join(parts).strip()
        dense_scores = {}
        if dense_query:
            intent.semantic_query = dense_query
            dense_scores = pipeline.dense.retrieve(intent, candidate_ids=cids,
                                                    top_k=TOP_K_RETRIEVE)
        dense_hit = gt in dense_scores

        # Example-based dense
        ex_scores = {}
        if intent.positive_track_ids:
            ex_scores = pipeline.dense.retrieve_by_example(
                intent.positive_track_ids, intent.negative_track_ids,
                candidate_ids=cids, top_k=TOP_K_RETRIEVE,
            )
        ex_hit = gt in ex_scores

        # ── Step 5: Union recall ──────────────────────────────────────────────
        union = set(bm25_scores) | set(dense_scores) | set(ex_scores)
        union_hit = gt in union

        # ── Step 6: RRF top-20 ────────────────────────────────────────────────
        from music_rec_baseline import rrf_fusion
        rrf_scores = rrf_fusion([bm25_scores, dense_scores], [0.4, 0.6])
        top20_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:20]
        rrf_hit = gt in top20_ids

        # ── Step 7: CF rerank top-20 ──────────────────────────────────────────
        ranked = pipeline.ranker.rank(user_id, rrf_scores, top_k=20,
                                       cold_handler=pipeline.cold_handler,
                                       user_culture=user_culture)
        cf_hit = any(st.track_id == gt for st in ranked)
        cf_rank = next((i+1 for i, st in enumerate(ranked)
                        if st.track_id == gt), None)

        if verbose:
            print(f"GT={gt[:8]}  filter={'✓' if filter_pass else '✗'}  "
                  f"bm25={'✓' if bm25_hit else '✗'}  "
                  f"dense={'✓' if dense_hit else '✗'}  "
                  f"ex={'✓' if ex_hit else '✗'}  "
                  f"union={'✓' if union_hit else '✗'}  "
                  f"rrf20={'✓' if rrf_hit else '✗'}  "
                  f"cf20={'✓' if cf_hit else '✗'}"
                  + (f"  rank={cf_rank}" if cf_rank else ""))

        results.append({
            "filter_pass": filter_pass,
            "bm25_hit":    bm25_hit,
            "dense_hit":   dense_hit,
            "ex_hit":      ex_hit,
            "union_hit":   union_hit,
            "rrf20_hit":   rrf_hit,
            "cf20_hit":    cf_hit,
            "cf_rank":     cf_rank,
            "intent_mode": intent.retrieval_mode,
            "has_artist":  bool(intent.artist_names),
            "has_genre":   bool(intent.genres or intent.mood),
            "listener_goal_len": len(listener_goal),
        })

    if not results:
        print("No results – check data.")
        return

    n = len(results)
    print(f"\n{'='*60}")
    print(f"Diagnosed {n} sessions  (skipped {skipped})")
    print(f"{'='*60}")

    def pct(key): return 100 * sum(r[key] for r in results) / n

    print(f"\n── Recall at each stage ──────────────────────────────────")
    print(f"  Filter pass (GT not filtered out): {pct('filter_pass'):5.1f}%")
    print(f"  BM25   recall@{TOP_K_RETRIEVE}:           {pct('bm25_hit'):5.1f}%")
    print(f"  Dense  recall@{TOP_K_RETRIEVE} (text):    {pct('dense_hit'):5.1f}%")
    print(f"  Dense  recall@{TOP_K_RETRIEVE} (example): {pct('ex_hit'):5.1f}%")
    print(f"  Union  recall@{TOP_K_RETRIEVE}:           {pct('union_hit'):5.1f}%  ← ceiling for ndcg")
    print(f"  RRF    hit@20:                  {pct('rrf20_hit'):5.1f}%")
    print(f"  CF     hit@20 (final):          {pct('cf20_hit'):5.1f}%")

    # Average rank when found
    found_ranks = [r["cf_rank"] for r in results if r["cf_rank"] is not None]
    if found_ranks:
        print(f"\n── When GT is in top-20 ──────────────────────────────────")
        print(f"  Mean rank:   {np.mean(found_ranks):.1f}")
        print(f"  Median rank: {np.median(found_ranks):.1f}")
        # Approximate ndcg
        dcg = np.mean([1/np.log2(r+1) for r in found_ranks])
        idcg = 1.0  # ideal: rank=1
        print(f"  Approx ndcg@20: {dcg:.4f}")

    # Break down by retrieval mode
    modes = defaultdict(list)
    for r in results: modes[r["intent_mode"]].append(r["cf20_hit"])
    print(f"\n── Hit@20 by retrieval_mode ──────────────────────────────")
    for mode, hits in sorted(modes.items()):
        print(f"  {mode:15s}: {100*np.mean(hits):5.1f}%  (n={len(hits)})")

    # Listener goal impact
    has_goal = [r for r in results if r["listener_goal_len"] > 0]
    no_goal  = [r for r in results if r["listener_goal_len"] == 0]
    if has_goal and no_goal:
        print(f"\n── Listener goal impact ──────────────────────────────────")
        print(f"  With listener_goal:    hit@20={100*np.mean([r['cf20_hit'] for r in has_goal]):.1f}%  (n={len(has_goal)})")
        print(f"  Without listener_goal: hit@20={100*np.mean([r['cf20_hit'] for r in no_goal]):.1f}%  (n={len(no_goal)})")

    print(f"\n{'='*60}")
    print("Interpretation:")
    print("  Filter pass < 95% → FilterEngine is too aggressive")
    print("  BM25 recall < 30% → BM25 query is wrong/empty for most sessions")
    print("  Dense recall < 30% → embedding space mismatch or empty query")
    print("  Union recall < 50% → fundamental retrieval problem")
    print("  Union recall > 60% but CF hit@20 < 24% → reranking problem")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("./data"))
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print("[LOAD] Loading data...")
    dfs = load_all_data(args.data_root)

    pipeline = MusicRecPipeline(
        track_meta=dfs["track_meta"],
        track_emb=dfs["track_emb"],
        user_emb=dfs["user_emb"],
        user_meta=dfs.get("user_meta"),
        sessions_df=dfs.get("sessions"),
        cache_file=Path("./cache/intent.jsonl"),
    )

    sess = dfs.get("sessions")
    if sess is None:
        print("[ERROR] sessions data not found")
        return

    # Use training sessions that have ground truth music turns
    # Filter to sessions with at least one music turn
    def has_music_turn(turns):
        if not isinstance(turns, (list, np.ndarray)):
            return False
        for t in turns:
            if isinstance(t, dict) and t.get("role") == "music":
                return True
        return False

    valid = sess[sess["conversations"].apply(has_music_turn)]
    print(f"[INFO] {len(valid)} sessions with music turns (out of {len(sess)})")

    diagnose(pipeline, valid, n_samples=args.n_samples, verbose=args.verbose)




def diagnose_single(pipeline, row, verbose=True):
    """深度分析单个 session 的失败原因"""
    import re
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

    turns        = _to_list(row.get("conversations"))
    conv_goal    = _to_dict(row.get("conversation_goal"))
    user_profile = _to_dict(row.get("user_profile"))
    user_id      = str(row.get("user_id", "") or "")

    gt = extract_gt_track(turns)
    if not gt:
        print("No GT found"); return

    # GT track metadata
    if gt in pipeline._meta.index:
        r = pipeline._meta.loc[gt]
        def _sj(v):
            if isinstance(v, (list, np.ndarray)): return " | ".join(str(x) for x in v)
            return str(v or "")
        print(f"\n=== GT Track ===")
        print(f"  track_id:    {gt}")
        print(f"  track_name:  {_sj(r.get('track_name'))}")
        print(f"  artist_name: {_sj(r.get('artist_name'))}")
        print(f"  album_name:  {_sj(r.get('album_name'))}")
        print(f"  tag_list:    {_sj(r.get('tag_list'))}")
        print(f"  release_date:{r.get('release_date','?')}")
        print(f"  popularity:  {r.get('popularity','?')}")

    print(f"\n=== Session Context ===")
    print(f"  listener_goal:  {conv_goal.get('listener_goal','')}")
    print(f"  culture:        {user_profile.get('preferred_musical_culture','')}")
    print(f"  category:       {conv_goal.get('category','')}")
    print(f"  specificity:    {conv_goal.get('specificity','')}")

    # User turns
    print(f"\n=== User Turns ===")
    for t in turns:
        t = _to_dict(t)
        if t.get("role") == "user":
            print(f"  USER: {t.get('content','')}")

    # Fallback intent
    from music_rec_baseline import _fallback_parse, BM25Retriever
    context_turns = []
    for t in turns:
        t2 = _to_dict(t)
        if t2.get("role") == "music" and str(t2.get("content","")).strip() == gt:
            break
        context_turns.append(t2)
    intent_history, _ = _build_conversation_history(context_turns)
    fake_conv = [{"role": m["role"], "content": m["content"]} for m in intent_history]
    intent = _fallback_parse(fake_conv)
    intent.listener_goal = str(conv_goal.get("listener_goal") or "")
    intent.user_culture  = str(user_profile.get("preferred_musical_culture") or "")
    intent.retrieval_mode = BM25Retriever._classify_mode(intent)

    print(f"\n=== Parsed Intent ===")
    print(f"  mode:          {intent.retrieval_mode}")
    print(f"  artist_names:  {intent.artist_names}")
    print(f"  track_names:   {intent.track_names}")
    print(f"  genres:        {intent.genres}")
    print(f"  mood:          {intent.mood}")
    print(f"  semantic_query:{intent.semantic_query}")
    print(f"  listener_goal: {intent.listener_goal}")

    # BM25 top-10
    bm25_scores = pipeline.bm25.retrieve(intent, top_k=500)
    bm25_rank = None
    ranked_bm25 = sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True)
    for i, (tid, s) in enumerate(ranked_bm25):
        if tid == gt:
            bm25_rank = i + 1
            break

    print(f"\n=== BM25 ===")
    print(f"  GT rank in BM25@500: {bm25_rank or 'NOT FOUND'}")
    print(f"  BM25 query tokens (mode={intent.retrieval_mode}):")
    # Show what tokens actually go into BM25
    tokens = intent.artist_names + intent.track_names + intent.album_names + intent.genres + intent.mood + intent.themes
    if intent.listener_goal:
        stop = {"a","the","and","or","in","of","to","for","with","that","this","is","it","from"}
        tokens += [w for w in intent.listener_goal.lower().split() if w not in stop]
    print(f"    {tokens[:20]}")
    print(f"  Top-5 BM25 results:")
    for tid, s in ranked_bm25[:5]:
        if tid in pipeline._meta.index:
            r2 = pipeline._meta.loc[tid]
            def _sj2(v):
                if isinstance(v, (list, np.ndarray)): return (list(v) or [""])[0]
                return str(v or "")
            print(f"    [{s:.3f}] {_sj2(r2.get('artist_name'))} – {_sj2(r2.get('track_name'))}")

    # Dense top-10
    dense_query = intent.listener_goal or intent.semantic_query or ""
    dense_rank = None
    dense_scores = {}
    if dense_query:
        intent.semantic_query = dense_query
        dense_scores = pipeline.dense.retrieve(intent, top_k=500)
        ranked_dense = sorted(dense_scores.items(), key=lambda x: x[1], reverse=True)
        for i, (tid, s) in enumerate(ranked_dense):
            if tid == gt:
                dense_rank = i + 1
                break

    print(f"\n=== Dense ===")
    print(f"  Query: {dense_query[:80]!r}")
    print(f"  GT rank in Dense@500: {dense_rank or 'NOT FOUND'}")
    if dense_scores:
        print(f"  Top-5 Dense results:")
        for tid, s in sorted(dense_scores.items(), key=lambda x: x[1], reverse=True)[:5]:
            if tid in pipeline._meta.index:
                r2 = pipeline._meta.loc[tid]
                def _sj3(v):
                    if isinstance(v, (list, np.ndarray)): return (list(v) or [""])[0]
                    return str(v or "")
                print(f"    [{s:.3f}] {_sj3(r2.get('artist_name'))} – {_sj3(r2.get('track_name'))}")

    # GT embedding similarity to top dense result
    if gt in pipeline.dense._id2idx and dense_scores:
        top_tid = sorted(dense_scores.items(), key=lambda x: x[1], reverse=True)[0][0]
        if top_tid in pipeline.dense._id2idx:
            gt_vec   = pipeline.dense._vecs_list[0][pipeline.dense._id2idx[gt]]
            top_vec  = pipeline.dense._vecs_list[0][pipeline.dense._id2idx[top_tid]]
            sim = float(np.dot(gt_vec, top_vec))
            print(f"\n  Cosine(GT, top-1 dense): {sim:.4f}")
            # Also show GT's similarity to query vector
            if dense_query:
                qv = pipeline.dense._encode(dense_query)
                qv_dim = pipeline.dense._vecs_list[0].shape[1]
                if qv.shape[0] > qv_dim: qv = qv[:qv_dim]
                else: qv = np.pad(qv, (0, qv_dim - qv.shape[0]))
                qv /= np.linalg.norm(qv) + 1e-9
                gt_sim = float(np.dot(qv, gt_vec))
                top_sim = float(np.dot(qv, top_vec))
                print(f"  Cosine(query, GT):       {gt_sim:.4f}")
                print(f"  Cosine(query, top-1):    {top_sim:.4f}")


def main_single():
    """诊断前10个失败 session 的具体原因"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("./data"))
    parser.add_argument("--n", type=int, default=5, help="Number of failed sessions to inspect")
    args = parser.parse_args()

    print("[LOAD] Loading data...")
    dfs = load_all_data(args.data_root)
    pipeline = MusicRecPipeline(
        track_meta=dfs["track_meta"],
        track_emb=dfs["track_emb"],
        user_emb=dfs["user_emb"],
        user_meta=dfs.get("user_meta"),
        sessions_df=dfs.get("sessions"),
        cache_file=Path("./cache/intent.jsonl"),
    )
    sess = dfs["sessions"]
    sample = sess.sample(min(50, len(sess)), random_state=42)

    shown = 0
    for _, row in sample.iterrows():
        if shown >= args.n:
            break
        turns = row.get("conversations")
        if not isinstance(turns, (list, np.ndarray)):
            continue
        gt = extract_gt_track(list(turns))
        if not gt:
            continue
        print(f"\n{'#'*70}")
        print(f"Session {shown+1}")
        diagnose_single(pipeline, row)
        shown += 1

if __name__ == "__main__":
    import sys
    if "--single" in sys.argv:
        sys.argv.remove("--single")
        main_single()
    else:
        main()