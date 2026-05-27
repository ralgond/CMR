"""
diagnose_entity_mode.py
=======================
统计 entity 模式（artist_names / track_names 非空）在测试集上的触发情况，
并区分来源：LLM 解析出 vs enrich_intent 从训练历史注入。

用法：
    python diagnose_entity_mode.py \
        --data_root /path/to/data \
        --cache     /path/to/intent_cache.pkl \
        --split     test          # test / train / all
        --max_sessions 200

输出：
    entity_mode_diagnosis.csv   每一个 (session, turn) 的详情
    entity_mode_summary.txt     汇总统计
"""

import argparse
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# ── 把 baseline 所在目录加入 path ──────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from music_rec_baseline import (
    MusicRecPipeline,
    ParsedIntent,
    UserHistoryIndex,
    BM25Retriever,
    IntentCache,
    parse_intent,
    load_all_data,
    DEEPSEEK_API_KEY,
    DEFAULT_DATA_ROOT,
    DEFAULT_CACHE_FILE,
)


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_history_artists(user_history, user_id, top_n=3):
    """Replicate what enrich_intent would inject for artist_names."""
    if user_history is None or not user_history.has_history(user_id):
        return []
    counter = user_history._profiles.get(user_id, {}).get("accepted_artists", {})
    return [a for a, _ in sorted(counter.items(), key=lambda x: x[1], reverse=True)[:top_n]]


def _would_inject_artists(user_history, user_id, intent_before_enrich):
    """True if enrich_intent *would* add artist_names to this intent."""
    mode = intent_before_enrich.retrieval_mode or BM25Retriever._classify_mode(intent_before_enrich)
    if mode not in ("abstract", "genre_mood", "default"):
        return False
    if intent_before_enrich.artist_names:          # already set by LLM
        return False
    return bool(_get_history_artists(user_history, user_id))


# ─────────────────────────────────────────────────────────────────────────────
# main diagnostic loop
# ─────────────────────────────────────────────────────────────────────────────

def run(args):
    t0 = time.time()
    print(f"[1/4] Loading data from {args.data_root} …")
    dfs = load_all_data(Path(args.data_root))

    sessions_df = dfs.get("sessions")
    if sessions_df is None:
        sys.exit("[ERROR] 'sessions' not found in data_root")

    if args.split != "all":
        col = "user_split" if "user_split" in sessions_df.columns else "split"
        if col in sessions_df.columns:
            sessions_df = sessions_df[sessions_df[col] == args.split]
            print(f"         split={args.split} → {len(sessions_df)} sessions")
        else:
            print(f"[WARN]   no split column found, using all {len(sessions_df)} sessions")

    sessions_df = sessions_df.head(args.max_sessions)
    print(f"         evaluating {len(sessions_df)} sessions")

    print("[2/4] Building UserHistoryIndex …")
    user_history = UserHistoryIndex(dfs.get("sessions", pd.DataFrame()), dfs["track_meta"]) \
        if "sessions" in dfs else None

    print("[3/4] Loading IntentCache …")
    cache = IntentCache(Path(args.cache))

    print("[4/4] Scanning turns …\n")

    rows = []          # one row per (session_id, turn_index, music_turn_index)

    # global counters
    c_total_music_turns  = 0
    c_entity_triggered   = 0
    c_from_llm_only      = 0   # LLM set artist/track, history did NOT inject
    c_from_history_only  = 0   # history injected, LLM set nothing
    c_from_both          = 0   # LLM set AND history would inject (shouldn't happen, but track it)
    c_from_neither       = 0   # entity flag not triggered (normal)

    bm25_sizes           = []  # how many BM25 results would come back (approximated by cache)
    entity_bm25_sizes    = []  # same, but only for entity turns

    for _, sess in sessions_df.iterrows():
        session_id = sess.get("session_id", sess.get("id", "?"))
        user_id    = str(sess["user_id"])
        turns      = sess["conversations"]
        conv_goal  = sess.get("conversation_goal", {}) or {}
        goal_type  = str(conv_goal.get("goal_type") or "?")
        specificity = str(conv_goal.get("specificity") or "?")

        history: list = []       # rolling conversation for intent parsing
        music_turn_idx = 0

        for t in turns:
            role    = t.get("role", "")
            content = (t.get("content") or "").strip()

            if role == "user":
                history.append({"role": "user", "content": content})

            elif role == "music" and content:
                c_total_music_turns += 1
                music_turn_idx      += 1

                # Parse intent *without* enrich — raw LLM signal
                intent_raw, from_cache = parse_intent(history, cache, DEEPSEEK_API_KEY)

                llm_has_artist = bool(intent_raw.artist_names)
                llm_has_track  = bool(intent_raw.track_names)
                llm_entity     = llm_has_artist or llm_has_track

                # Would enrich_intent inject artists here?
                would_inject   = _would_inject_artists(user_history, user_id, intent_raw)

                # Simulate enrich to get final state (mirrors pipeline logic)
                intent_enriched = ParsedIntent(
                    artist_names=list(intent_raw.artist_names),
                    track_names =list(intent_raw.track_names),
                    retrieval_mode=intent_raw.retrieval_mode,
                )
                if would_inject:
                    intent_enriched.artist_names = _get_history_artists(
                        user_history, user_id
                    )

                final_entity   = bool(intent_enriched.artist_names or intent_enriched.track_names)
                c_entity_triggered += int(final_entity)

                # Source classification
                if llm_entity and not would_inject:
                    source = "LLM_only"
                    c_from_llm_only += 1
                elif not llm_entity and would_inject:
                    source = "history_only"
                    c_from_history_only += 1
                elif llm_entity and would_inject:
                    source = "both"
                    c_from_both += 1
                else:
                    source = "none"
                    c_from_neither += 1

                rows.append({
                    "session_id":        session_id,
                    "user_id":           user_id,
                    "goal_type":         goal_type,
                    "specificity":       specificity,
                    "music_turn_idx":    music_turn_idx,
                    "from_cache":        from_cache,
                    "llm_artist_names":  json.dumps(intent_raw.artist_names),
                    "llm_track_names":   json.dumps(intent_raw.track_names),
                    "retrieval_mode":    intent_raw.retrieval_mode or "?",
                    "would_inject_hist": would_inject,
                    "final_entity_mode": final_entity,
                    "entity_source":     source,
                })

                history.append({"role": "assistant", "content": f"[played:{content}]"})

    # ── Build DataFrame ───────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    out_csv = SCRIPT_DIR / "entity_mode_diagnosis.csv"
    df.to_csv(out_csv, index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    lines = []
    lines.append("=" * 62)
    lines.append("  Entity-Mode Diagnostic Report")
    lines.append(f"  sessions={len(sessions_df)}  split={args.split}  "
                 f"elapsed={time.time()-t0:.1f}s")
    lines.append("=" * 62)
    lines.append("")

    pct = lambda n: f"{n:>6}  ({100*n/max(c_total_music_turns,1):5.1f}%)"

    lines.append(f"  Total music turns evaluated :  {c_total_music_turns}")
    lines.append("")
    lines.append("  ── Entity mode trigger breakdown ──────────────────────")
    lines.append(f"  entity mode triggered (final):  {pct(c_entity_triggered)}")
    lines.append(f"    ├─ source = LLM only          {pct(c_from_llm_only)}")
    lines.append(f"    ├─ source = history inject     {pct(c_from_history_only)}")
    lines.append(f"    ├─ source = both               {pct(c_from_both)}")
    lines.append(f"    └─ no entity (dense runs)      {pct(c_from_neither)}")
    lines.append("")

    if not df.empty:
        lines.append("  ── Entity mode by goal_type ────────────────────────────")
        gt = df.groupby("goal_type")["final_entity_mode"].agg(["sum", "count"])
        gt["rate"] = gt["sum"] / gt["count"]
        for gtype, row2 in gt.sort_values("rate", ascending=False).iterrows():
            bar = "█" * int(row2["rate"] * 20)
            lines.append(f"    {gtype:>3}  {row2['rate']:5.1%}  {bar}")
        lines.append("")

        lines.append("  ── Entity mode by retrieval_mode ───────────────────────")
        rm = df.groupby("retrieval_mode")["final_entity_mode"].agg(["sum", "count"])
        rm["rate"] = rm["sum"] / rm["count"]
        for rmode, row2 in rm.sort_values("count", ascending=False).iterrows():
            lines.append(f"    {rmode:<15}  triggered {row2['sum']:>4}/{row2['count']:<4}"
                         f"  ({row2['rate']:.1%})")
        lines.append("")

        lines.append("  ── History-injected artist names (top 10) ──────────────")
        hist_turns = df[df["entity_source"] == "history_only"]
        if not hist_turns.empty:
            # Count which session × turn combos had injection
            lines.append(f"    {len(hist_turns)} turns affected by history injection")
            lines.append(f"    goal_type distribution:")
            gt_hist = hist_turns["goal_type"].value_counts()
            for g, cnt in gt_hist.head(8).items():
                lines.append(f"      {g}: {cnt}")
        else:
            lines.append("    (none — history injection never triggered entity mode)")
        lines.append("")

        lines.append("  ── LLM-extracted artists / tracks (sample) ────────────")
        llm_turns = df[df["entity_source"] == "LLM_only"].head(8)
        for _, r in llm_turns.iterrows():
            a = r["llm_artist_names"]
            t = r["llm_track_names"]
            lines.append(f"    goal={r['goal_type']} mode={r['retrieval_mode']:<14} "
                         f"artists={a}  tracks={t}")
        lines.append("")

    lines.append(f"  Detailed CSV saved → {out_csv}")
    lines.append("=" * 62)

    summary = "\n".join(lines)
    print(summary)

    out_txt = SCRIPT_DIR / "entity_mode_summary.txt"
    out_txt.write_text(summary, encoding="utf-8")
    print(f"\n  Summary saved → {out_txt}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose entity-mode trigger rates")
    parser.add_argument("--data_root",    default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--cache",        default=str(DEFAULT_CACHE_FILE))
    parser.add_argument("--split",        default="test",
                        choices=["test", "train", "all"])
    parser.add_argument("--max_sessions", type=int, default=200)
    run(parser.parse_args())