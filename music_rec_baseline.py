"""
TalkPlay Conversational Music Recommendation Baseline
======================================================
Architecture: DeepSeek intent parsing → Filter → BM25 + Dense → CF-BPR Rerank

Local data layout (--data-root, default: ./data):
  <data_root>/Challenge-Data/         ← sessions (train + test splits)
  <data_root>/Track-Metadata/         ← track metadata parquet(s)
  <data_root>/Track-Embedding/        ← track embeddings parquet(s)
  <data_root>/User-Metadata/          ← user metadata parquet(s)
  <data_root>/User-Embedding/         ← user embeddings parquet(s)
  <data_root>/Challenge-Blind-A/      ← blind test set

Intent cache: cache/intent.jsonl  (keyed by sha256 of conversation text)
  - On first run: call DeepSeek, write result to cache
  - On subsequent runs: read from cache, skip API call

Pipeline per turn:
  1. IntentParser  → ParsedIntent  (DeepSeek + disk cache)
  2. FilterEngine  → candidate_ids (year / popularity / duration)
  3. BM25Retriever → bm25_scores
  4. DenseRetriever→ dense_scores  (FAISS over qwen3 1024d attrs)
  5. RRF Fusion    → rrf_scores
  6. CFBPRRanker   → final ranking (dot user_cf · track_cf)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# 0. Config
# ─────────────────────────────────────────────────────────────────────────────

DEEPSEEK_API_KEY  = os.getenv("DEEPSEEK_API_KEY", "YOUR_DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

DEFAULT_DATA_ROOT  = Path("./data")
DEFAULT_CACHE_FILE = Path("./cache/intent.jsonl")

DECADE_MAP: Dict[str, Tuple[int, int]] = {
    "50s":        (1950, 1959), "1950s":      (1950, 1959),
    "60s":        (1960, 1969), "1960s":      (1960, 1969),
    "70s":        (1970, 1979), "1970s":      (1970, 1979),
    "80s":        (1980, 1989), "1980s":      (1980, 1989),
    "90s":        (1990, 1999), "1990s":      (1990, 1999),
    "00s":        (2000, 2009), "2000s":      (2000, 2009),
    "10s":        (2010, 2019), "2010s":      (2010, 2019),
    "20s":        (2020, 2029), "2020s":      (2020, 2029),
    "early 2000s":(2000, 2004), "late 2000s": (2005, 2009),
    "early 90s":  (1990, 1994), "late 90s":   (1995, 1999),
    "early 80s":  (1980, 1984), "late 80s":   (1985, 1989),
    "early 70s":  (1970, 1974), "late 70s":   (1975, 1979),
}

TOP_K_RETRIEVE = 500   # larger candidate pool improves recall
TOP_K_FINAL    = 20    # match ndcg@20 evaluation metric
BM25_WEIGHT    = 0.4
DENSE_WEIGHT   = 0.6


# ─────────────────────────────────────────────────────────────────────────────
# 1. Data Loading  (datasets library → pandas)
# ─────────────────────────────────────────────────────────────────────────────

def _load_local(data_dir: Path, split: Optional[str] = None) -> pd.DataFrame:
    """
    Load a local HuggingFace dataset directory (parquet files) as a DataFrame.
    Uses the `datasets` library for consistent handling of nested types,
    then converts to pandas.

    If `split` is given (e.g. "train", "all_tracks"), only that split is loaded.
    Otherwise all splits are concatenated.
    """
    from datasets import load_dataset  # type: ignore

    data_str = str(data_dir.resolve())
    try:
        if split:
            ds = load_dataset("parquet", data_dir=data_str, split=split)
        else:
            ds_dict = load_dataset("parquet", data_dir=data_str)
            # Concatenate all splits
            frames = [s.to_pandas() for s in ds_dict.values()]
            return pd.concat(frames, ignore_index=True)
        return ds.to_pandas()
    except Exception:
        # Fallback: glob all parquet files ourselves
        parquet_files = sorted(data_dir.rglob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found under {data_dir}")
        frames = [pd.read_parquet(p) for p in parquet_files]
        df = pd.concat(frames, ignore_index=True)
        if split:
            # Try to filter by a 'split' column if present
            if "split" in df.columns:
                df = df[df["split"] == split]
        return df


def load_all_data(data_root: Path) -> Dict[str, pd.DataFrame]:
    """
    Load all five datasets from local directories.

    Returns dict with keys:
        sessions, track_meta, track_emb, user_meta, user_emb, blind
    """
    root = Path(data_root)

    dirs = {
        "sessions":   root / "Challenge-Data",
        "track_meta": root / "Track-Metadata",
        "track_emb":  root / "Track-Embedding",
        "user_meta":  root / "User-Metadata",
        "user_emb":   root / "User-Embedding",
        "blind":      root / "Challenge-Blind-A",
    }

    dfs: Dict[str, pd.DataFrame] = {}
    for name, d in dirs.items():
        if not d.exists():
            print(f"[WARN] {d} not found – skipping '{name}'")
            continue
        t0 = time.time()
        print(f"[LOAD] {name:12s} ← {d} …", end=" ", flush=True)
        dfs[name] = _load_local(d)
        print(f"{len(dfs[name]):,} rows  ({time.time()-t0:.1f}s)")

    return dfs


# ─────────────────────────────────────────────────────────────────────────────
# 2. Data Structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ParsedIntent:
    """Structured intent extracted from a conversation turn."""
    artist_names:    List[str]      = field(default_factory=list)
    track_names:     List[str]      = field(default_factory=list)
    album_names:     List[str]      = field(default_factory=list)
    genres:          List[str]      = field(default_factory=list)
    mood:            List[str]      = field(default_factory=list)
    themes:          List[str]      = field(default_factory=list)

    year_min:        Optional[int]  = None
    year_max:        Optional[int]  = None
    decade:          Optional[str]  = None

    popularity_min:  Optional[int]  = None
    popularity_max:  Optional[int]  = None
    duration_min_ms: Optional[int]  = None
    duration_max_ms: Optional[int]  = None

    is_abstract:     bool           = False
    semantic_query:  str            = ""

    positive_track_ids: List[str]   = field(default_factory=list)
    negative_track_ids: List[str]   = field(default_factory=list)

    # Extra context injected from session metadata (not from DeepSeek)
    listener_goal:      str         = ""   # conversation_goal.listener_goal
    user_culture:       str         = ""   # user_profile.preferred_musical_culture

    # Retrieval mode – set by classify_retrieval_mode() after intent parsing
    # "exact_track"  : user names a specific song title
    # "exact_artist" : user names a specific artist (all tracks by that artist)
    # "exact_album"  : user names a specific album
    # "genre_mood"   : genre / mood / era search (no specific entity)
    # "abstract"     : purely vibe-based, no text anchors
    retrieval_mode: str             = ""


@dataclass
class ScoredTrack:
    track_id:    str
    bm25_score:  float = 0.0
    dense_score: float = 0.0
    rrf_score:   float = 0.0
    cf_score:    float = 0.0
    final_score: float = 0.0
    metadata:    Dict[str, Any] = field(default_factory=dict)



# ─────────────────────────────────────────────────────────────────────────────
# 2b. Session Index  (built from training conversations)
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# 2b. UserHistoryIndex  (the RIGHT way to use training sessions)
# ─────────────────────────────────────────────────────────────────────────────

class UserHistoryIndex:
    """
    Builds a per-user profile from training sessions:

      user_id → {
        accepted_track_ids: [...]   # MOVES_TOWARD_GOAL tracks (ordered)
        rejected_track_ids: [...]   # DOES_NOT_MOVE tracks
        accepted_artists:   Counter # artist_name → accept count
        accepted_tags:      Counter # tag → accept count
      }

    At predict time, for warm users:
      1. The accepted_track_ids seed example-based dense retrieval directly
         (instead of relying on DeepSeek to parse UUIDs from conversation)
      2. The top accepted artists/tags are injected into BM25 query expansion
      3. Rejected tracks are excluded from the final result

    For cold users: no history → falls back to normal pipeline.

    Why this works for ndcg@20=0.5:
      A user who historically listened to "Nine Inch Nails" and "Rammstein"
      will get industrial metal tracks ranked #1-2, which is exactly right.
      The training data directly tells us the user's taste.
    """

    UUID_RE = re.compile(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
        re.I,
    )

    def __init__(
        self,
        sessions_df: pd.DataFrame,
        track_meta: pd.DataFrame,
    ):
        from collections import Counter, defaultdict

        def _to_list(v):
            if v is None: return []
            if isinstance(v, np.ndarray): return v.tolist()
            if isinstance(v, list): return v
            try:
                import pandas as _pd
                if _pd.isna(v): return []
            except (TypeError, ValueError): pass
            return []

        def _to_dict(v):
            if v is None or (isinstance(v, float) and np.isnan(v)): return {}
            if isinstance(v, dict): return v
            if isinstance(v, str):
                try: return json.loads(v)
                except: return {}
            return {}

        def _safe_join(v):
            if isinstance(v, (list, np.ndarray)):
                return " ".join(str(x) for x in v if x)
            return str(v) if v else ""

        # Build track_id → {artist_names, tags} lookup from track_meta
        self._track_artists: Dict[str, List[str]] = {}
        self._track_tags:    Dict[str, List[str]] = {}
        for _, row in track_meta.iterrows():
            tid = str(row.get("track_id", "") or "")
            if not tid: continue
            artists = _to_list(row.get("artist_name"))
            tags    = _to_list(row.get("tag_list"))
            self._track_artists[tid] = [str(a) for a in artists if a]
            self._track_tags[tid]    = [str(t) for t in tags if t]

        # user_id → profile
        self._profiles: Dict[str, Dict] = defaultdict(lambda: {
            "accepted_track_ids": [],
            "rejected_track_ids": [],
            "accepted_artists":   Counter(),
            "accepted_tags":      Counter(),
        })

        n_users = 0
        n_accepted = 0

        for _, row in sessions_df.iterrows():
            user_id     = str(row.get("user_id", "") or "")
            turns       = _to_list(row.get("conversations"))
            assessments = _to_list(row.get("goal_progress_assessments"))
            if not user_id: continue

            # build turn_number → assessment
            assess_map: Dict[int, str] = {}
            for a in assessments:
                a = _to_dict(a) if not isinstance(a, dict) else a
                tn  = a.get("turn_number")
                gpa = str(a.get("goal_progress_assessment") or "")
                if tn is not None:
                    try: assess_map[int(tn)] = gpa
                    except (TypeError, ValueError): pass

            profile = self._profiles[user_id]
            is_new  = len(profile["accepted_track_ids"]) == 0

            for t in turns:
                t = _to_dict(t) if not isinstance(t, dict) else t
                role    = str(t.get("role", "") or "")
                content = str(t.get("content", "") or "").strip()
                turn_no = t.get("turn_number")
                if role != "music" or not self.UUID_RE.match(content):
                    continue
                gpa = assess_map.get(int(turn_no) if turn_no is not None else -1, "")
                if gpa == "MOVES_TOWARD_GOAL":
                    profile["accepted_track_ids"].append(content)
                    # accumulate artist/tag counts
                    for a in self._track_artists.get(content, []):
                        profile["accepted_artists"][a] += 1
                    for tg in self._track_tags.get(content, [])[:5]:
                        profile["accepted_tags"][tg] += 1
                    n_accepted += 1
                elif gpa == "DOES_NOT_MOVE_TOWARD_GOAL":
                    profile["rejected_track_ids"].append(content)

            if is_new and len(profile["accepted_track_ids"]) > 0:
                n_users += 1

        # Deduplicate (preserve order, keep most recent)
        for uid, p in self._profiles.items():
            seen: set = set()
            deduped = []
            for t in p["accepted_track_ids"]:
                if t not in seen:
                    seen.add(t)
                    deduped.append(t)
            p["accepted_track_ids"] = deduped

            seen2: set = set()
            deduped2 = []
            for t in p["rejected_track_ids"]:
                if t not in seen2:
                    seen2.add(t)
                    deduped2.append(t)
            p["rejected_track_ids"] = deduped2

        print(f"[UserHistoryIndex] {n_users} users with history  "
              f"{n_accepted} accepted interactions")

    def has_history(self, user_id: str) -> bool:
        return len(self._profiles.get(user_id, {}).get("accepted_track_ids", [])) > 0

    def get_accepted_tracks(self, user_id: str, max_recent: int = 10) -> List[str]:
        """Most recent accepted track_ids (best seed for dense retrieval)."""
        tracks = self._profiles.get(user_id, {}).get("accepted_track_ids", [])
        return tracks[-max_recent:]

    def get_rejected_tracks(self, user_id: str) -> set:
        return set(self._profiles.get(user_id, {}).get("rejected_track_ids", []))

    def get_top_artists(self, user_id: str, top_n: int = 5) -> List[str]:
        """Top accepted artists by frequency → BM25 query expansion."""
        counter = self._profiles.get(user_id, {}).get("accepted_artists", {})
        return [a for a, _ in sorted(counter.items(), key=lambda x: x[1], reverse=True)[:top_n]]

    def get_top_tags(self, user_id: str, top_n: int = 8) -> List[str]:
        """Top accepted tags by frequency → BM25 query expansion."""
        counter = self._profiles.get(user_id, {}).get("accepted_tags", {})
        return [t for t, _ in sorted(counter.items(), key=lambda x: x[1], reverse=True)[:top_n]]

    def enrich_intent(self, user_id: str, intent: "ParsedIntent") -> "ParsedIntent":
        """
        Inject user history signals into the intent object:
          - accepted_track_ids → positive_track_ids (seeds dense retrieval)
          - top artists → artist_names if not already set
          - top tags → genres/mood
          - rejected tracks → negative_track_ids
        Current session's positive_track_ids take priority (appended last).
        """
        if not self.has_history(user_id):
            return intent

        hist_tracks = self.get_accepted_tracks(user_id, max_recent=10)
        # Merge: history first, current session last (more recent = higher weight)
        combined = list(dict.fromkeys(hist_tracks + intent.positive_track_ids))
        intent.positive_track_ids = combined

        # Enrich negative tracks
        rejected = self.get_rejected_tracks(user_id)
        intent.negative_track_ids = list(
            set(intent.negative_track_ids) | rejected
        )

        # Only inject history artist/tag when the query is abstract or vague
        # (don't override when user has specified a concrete entity)
        mode = intent.retrieval_mode or BM25Retriever._classify_mode(intent)
        if mode in ("abstract", "genre_mood", "default") and not intent.artist_names:
            top_artists = self.get_top_artists(user_id, top_n=3)
            if top_artists:
                intent.artist_names = top_artists

        if mode in ("abstract", "genre_mood") and not intent.genres and not intent.mood:
            top_tags = self.get_top_tags(user_id, top_n=5)
            if top_tags:
                intent.genres = top_tags

        return intent

# ─────────────────────────────────────────────────────────────────────────────
# 3. Intent Cache
# ─────────────────────────────────────────────────────────────────────────────

class IntentCache:
    """
    Persist DeepSeek intent-parse results to cache/intent.jsonl.

    Format: one JSON object per line, each with a "key" field (SHA-256 of
    the conversation) plus all ParsedIntent fields.

      {"key": "abc123...", "artist_names": [...], "year_min": 2000, ...}
      {"key": "def456...", "is_abstract": true, "semantic_query": "...", ...}

    On read : scan all lines once at init → in-memory dict, O(1) lookup.
    On write: append a single new line (no full rewrite needed).
              Duplicate keys are resolved at load time (last line wins),
              and the file is compacted when duplicates exceed 10 % of entries.
    """

    _COMPACT_THRESHOLD = 0.10

    def __init__(self, cache_file: Path = DEFAULT_CACHE_FILE):
        self._path = Path(cache_file)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._data: Dict[str, Dict] = {}
        self._total_lines = 0
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        errors = 0
        with open(self._path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    key = record.pop("key")
                    self._data[key] = record
                    self._total_lines += 1
                except (json.JSONDecodeError, KeyError):
                    errors += 1
        n = len(self._data)
        print(f"[CACHE] Loaded {n} cached intents from {self._path}"
              + (f"  ({errors} malformed lines skipped)" if errors else ""))

    def _append(self, key: str, record: Dict) -> None:
        """Append one line to the JSONL file."""
        entry = {"key": key, **record}
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._total_lines += 1

    def _compact(self) -> None:
        """Rewrite the file keeping only the latest entry per key."""
        tmp = self._path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for key, record in self._data.items():
                entry = {"key": key, **record}
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        tmp.replace(self._path)
        self._total_lines = len(self._data)
        print(f"[CACHE] Compacted → {self._total_lines} lines")

    @staticmethod
    def _key(conversation: List[Dict[str, str]], n_turns: int = 6) -> str:
        """Stable SHA-256 key from the last n_turns of the conversation."""
        tail = conversation[-n_turns:]
        blob = json.dumps(tail, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()

    def get(self, conversation: List[Dict[str, str]]) -> Optional[ParsedIntent]:
        key = self._key(conversation)
        if key not in self._data:
            return None
        return _dict_to_intent(self._data[key])

    def set(self, conversation: List[Dict[str, str]], intent: ParsedIntent) -> None:
        key = self._key(conversation)
        self._data[key] = asdict(intent)
        self._append(key, asdict(intent))
        # Compact if duplicate ratio exceeds threshold
        n_unique = len(self._data)
        if self._total_lines > n_unique and (
            (self._total_lines - n_unique) / self._total_lines > self._COMPACT_THRESHOLD
        ):
            self._compact()

    def __len__(self) -> int:
        return len(self._data)


# ─────────────────────────────────────────────────────────────────────────────
# 4. DeepSeek Intent Parser
# ─────────────────────────────────────────────────────────────────────────────

INTENT_SYSTEM_PROMPT = """
You are a music intent extractor for a conversational music recommendation system.
Given a conversation history (user + assistant turns), extract structured search
intent from the LATEST user message while taking context from previous turns.

Return ONLY valid JSON matching this schema (omit null/empty fields):
{
  "artist_names": ["string"],
  "track_names": ["string"],
  "album_names": ["string"],
  "genres": ["string"],
  "mood": ["string"],
  "themes": ["string"],
  "year_min": int,
  "year_max": int,
  "decade": "string",
  "popularity_min": int,
  "popularity_max": int,
  "duration_min_ms": int,
  "duration_max_ms": int,
  "is_abstract": bool,
  "semantic_query": "string",
  "positive_track_ids": ["uuid"],
  "negative_track_ids": ["uuid"]
}

Decade rules:
  "2000s"/"00s"     → year_min:2000 year_max:2009
  "early 2000s"     → year_min:2000 year_max:2004
  "late 2000s"      → year_min:2005 year_max:2009
  "90s"/"1990s"     → year_min:1990 year_max:1999
  "late 90s"        → year_min:1995 year_max:1999

is_abstract=true when the query has NO specific artist/title, only mood/vibe.
semantic_query: a concise English description for dense retrieval.
"""


def _conv_key_str(conversation: List[Dict[str, str]], n: int = 6) -> str:
    return " | ".join(
        f"{m['role']}:{m['content']}" for m in conversation[-n:]
    )


def _dict_to_intent(d: Dict[str, Any]) -> ParsedIntent:
    intent = ParsedIntent()
    for fname in ParsedIntent.__dataclass_fields__:
        if fname in d and d[fname] is not None:
            setattr(intent, fname, d[fname])
    # Resolve decade → year range if not yet set
    if intent.decade and not (intent.year_min or intent.year_max):
        key = intent.decade.lower().strip()
        if key in DECADE_MAP:
            intent.year_min, intent.year_max = DECADE_MAP[key]
    return intent


def _fallback_parse(conversation: List[Dict[str, str]]) -> ParsedIntent:
    """Keyword-based fallback when DeepSeek is unavailable."""
    text = next(
        (m["content"] for m in reversed(conversation) if m["role"] == "user"), ""
    )
    intent = ParsedIntent(semantic_query=text)
    tl = text.lower()

    # Decade detection (longest match first)
    for dk in sorted(DECADE_MAP, key=len, reverse=True):
        if dk in tl:
            intent.decade = dk
            intent.year_min, intent.year_max = DECADE_MAP[dk]
            break

    abstract_words = {
        "mellow", "chill", "intense", "dramatic", "sad", "happy",
        "energetic", "relaxing", "upbeat", "dark", "dreamy", "calm",
        "melancholic", "ambient", "soothing", "aggressive", "peaceful",
    }
    words = set(tl.split())
    found_mood = words & abstract_words
    if found_mood and not intent.artist_names and not intent.track_names:
        intent.is_abstract = True
        intent.mood = sorted(found_mood)

    return intent


def parse_intent(
    conversation: List[Dict[str, str]],
    cache: IntentCache,
    api_key: str = DEEPSEEK_API_KEY,
) -> Tuple[ParsedIntent, bool]:
    """
    Returns (ParsedIntent, from_cache).
    Checks cache first; on miss calls DeepSeek and writes result to cache.
    """
    # ── Cache hit ────────────────────────────────────────────────────────────
    cached = cache.get(conversation)
    if cached is not None:
        return cached, True

    # ── DeepSeek API call ────────────────────────────────────────────────────
    try:
        import requests  # type: ignore

        messages = [{"role": "system", "content": INTENT_SYSTEM_PROMPT}]
        conv_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}"
            for m in conversation[-6:]
        )
        messages.append({"role": "user", "content": conv_text})

        resp = requests.post(
            f"{DEEPSEEK_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "deepseek-chat",
                "messages": messages,
                "max_tokens": 512,
                "temperature": 0.0,
            },
            timeout=15,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
        data = json.loads(raw)
        intent = _dict_to_intent(data)

    except Exception as e:
        print(f"[WARN] DeepSeek failed ({type(e).__name__}: {e}). Using keyword fallback.")
        intent = _fallback_parse(conversation)

    # ── Write to cache ───────────────────────────────────────────────────────
    cache.set(conversation, intent)
    return intent, False


# ─────────────────────────────────────────────────────────────────────────────
# 5. Filter Engine
# ─────────────────────────────────────────────────────────────────────────────

class FilterEngine:
    """Boolean hard-filter on Track-Metadata."""

    def __init__(self, track_meta: pd.DataFrame):
        self.df = track_meta.copy()
        self.df["release_year"] = self.df["release_date"].apply(self._parse_year)

    @staticmethod
    def _parse_year(v: Any) -> Optional[int]:
        if not v or (isinstance(v, float) and np.isnan(v)):
            return None
        m = re.match(r"(\d{4})", str(v))
        return int(m.group(1)) if m else None

    # Minimum fraction of corpus that must survive hard filters.
    # If a filter would leave fewer tracks than this fraction × total,
    # that filter is silently relaxed (not applied).
    MIN_PASS_RATIO = 0.05   # keep at least 5 % of corpus after filtering

    def apply(self, intent: ParsedIntent) -> pd.DataFrame:
        df    = self.df
        total = len(df)

        def _safe_apply(df_in, mask_fn):
            """Apply mask_fn; fall back to df_in if result is too small."""
            df_out = df_in[mask_fn(df_in)]
            if len(df_out) >= max(1, total * self.MIN_PASS_RATIO):
                return df_out
            return df_in   # relax – do not apply this filter

        if intent.year_min is not None:
            df = _safe_apply(df, lambda d: d["release_year"].notna() & (d["release_year"] >= intent.year_min))
        if intent.year_max is not None:
            df = _safe_apply(df, lambda d: d["release_year"].notna() & (d["release_year"] <= intent.year_max))
        if intent.popularity_min is not None:
            df = _safe_apply(df, lambda d: d["popularity"].notna() & (d["popularity"] >= intent.popularity_min))
        if intent.popularity_max is not None:
            df = _safe_apply(df, lambda d: d["popularity"].notna() & (d["popularity"] <= intent.popularity_max))
        if intent.duration_min_ms is not None:
            df = _safe_apply(df, lambda d: d["duration"].notna() & (d["duration"] >= intent.duration_min_ms))
        if intent.duration_max_ms is not None:
            df = _safe_apply(df, lambda d: d["duration"].notna() & (d["duration"] <= intent.duration_max_ms))

        return df


# ─────────────────────────────────────────────────────────────────────────────
# 6. BM25 Retriever
# ─────────────────────────────────────────────────────────────────────────────

class BM25Retriever:
    """
    Stratified BM25 retrieval.

    Uses retrieval_mode to decide WHICH fields to match and HOW:

      exact_track  → query on track_name only; strong signal
      exact_artist → query on artist_name only; pre-filter by artist match
      exact_album  → query on album_name only; pre-filter by album match
      genre_mood   → query on tag_list + decade; no title/artist noise
      abstract     → query on tag_list + mood; pure vibe matching
      (default)    → full query, all fields combined (current behaviour)

    BM25 scores are normalised per retrieval mode so downstream RRF
    weights remain consistent.
    """

    def __init__(self, track_meta: pd.DataFrame):
        from rank_bm25 import BM25Okapi  # type: ignore

        self._ids    = track_meta["track_id"].tolist()
        self._id2idx = {tid: i for i, tid in enumerate(self._ids)}

        # Store raw field values for per-field lookup
        self._artist_names: List[str] = []
        self._album_names:  List[str] = []
        self._track_names:  List[str] = []
        self._tag_strings:  List[str] = []
        self._year_strings: List[str] = []

        for _, row in track_meta.iterrows():
            self._artist_names.append(self._safe_join(row.get("artist_name")).lower())
            self._album_names.append(self._safe_join(row.get("album_name")).lower())
            self._track_names.append(self._safe_join(row.get("track_name")).lower())
            self._tag_strings.append(self._safe_join(row.get("tag_list")).lower())
            rd = str(row.get("release_date") or "")
            m = re.match(r"(\d{4})", rd)
            if m:
                yr = int(m.group(1))
                self._year_strings.append(f"{(yr//10)*10}s {yr}")
            else:
                self._year_strings.append("")

        # Four separate BM25 indexes for different match strategies
        def _mk(docs):
            return BM25Okapi([d.split() for d in docs])

        # Full-text: all fields combined (fallback)
        full_corpus = [
            " ".join(filter(None, [an, aln, tn, tg, yr]))
            for an, aln, tn, tg, yr in zip(
                self._artist_names, self._album_names,
                self._track_names, self._tag_strings, self._year_strings
            )
        ]
        self._bm25_full   = _mk(full_corpus)
        self._corpus      = full_corpus   # kept for legacy

        # Artist-only index
        self._bm25_artist = _mk(self._artist_names)

        # Album-only index
        self._bm25_album  = _mk(self._album_names)

        # Track-name-only index
        self._bm25_track  = _mk(self._track_names)

        # Tag + year index (genre/mood/era)
        tag_year_corpus = [
            " ".join(filter(None, [tg, yr]))
            for tg, yr in zip(self._tag_strings, self._year_strings)
        ]
        self._bm25_tag    = _mk(tag_year_corpus)

    @staticmethod
    def _safe_join(v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, (list, np.ndarray)):
            return " ".join(str(x) for x in v if x is not None and str(x).strip())
        s = str(v).strip()
        return s if s not in ("nan", "None", "") else ""

    def _scores_to_dict(
        self,
        scores: np.ndarray,
        candidate_ids: Optional[List[str]],
        top_k: int,
    ) -> Dict[str, float]:
        id_score = dict(zip(self._ids, map(float, scores)))
        if candidate_ids is not None:
            cset = set(candidate_ids)
            id_score = {k: v for k, v in id_score.items() if k in cset}
        return dict(sorted(id_score.items(), key=lambda x: x[1], reverse=True)[:top_k])

    def retrieve(
        self,
        intent: ParsedIntent,
        candidate_ids: Optional[List[str]] = None,
        top_k: int = TOP_K_RETRIEVE,
    ) -> Dict[str, float]:
        mode = intent.retrieval_mode or self._classify_mode(intent)

        # ── exact_track: user named a specific song title ─────────────────────
        if mode == "exact_track" and intent.track_names:
            query = " ".join(intent.track_names).lower().split()
            # Also narrow by artist if given
            if intent.artist_names:
                artist_q = " ".join(intent.artist_names).lower()
                cids_artist = [
                    tid for tid, a in zip(self._ids, self._artist_names)
                    if any(name.lower() in a for name in intent.artist_names)
                ]
                cids_use = list(set(cids_artist) & set(candidate_ids)) if candidate_ids else cids_artist
                candidate_ids = cids_use or candidate_ids
            scores = self._bm25_track.get_scores(query)
            return self._scores_to_dict(scores, candidate_ids, top_k)

        # ── exact_artist: user named a specific artist ────────────────────────
        if mode == "exact_artist" and intent.artist_names:
            # Pre-filter: only tracks whose artist_name contains any named artist
            artist_lower = [a.lower() for a in intent.artist_names]
            artist_cids = [
                tid for tid, a in zip(self._ids, self._artist_names)
                if any(al in a for al in artist_lower)
            ]
            if candidate_ids:
                artist_cids = [t for t in artist_cids if t in set(candidate_ids)]
            if not artist_cids:
                # Fall back to BM25 artist search
                query = " ".join(intent.artist_names).lower().split()
                scores = self._bm25_artist.get_scores(query)
                return self._scores_to_dict(scores, candidate_ids, top_k)
            # Within artist tracks, rank by tag/mood/year match if any
            if intent.genres or intent.mood or intent.decade:
                tag_query = (
                    " ".join(intent.genres + intent.mood)
                    + " " + (intent.decade or "")
                ).strip().lower().split()
                if tag_query:
                    scores = self._bm25_tag.get_scores(tag_query)
                    return self._scores_to_dict(scores, artist_cids, top_k)
            # No further refinement: return all tracks by artist, scored equally
            # (BM25 artist name match gives natural ranking)
            query = " ".join(intent.artist_names).lower().split()
            scores = self._bm25_artist.get_scores(query)
            return self._scores_to_dict(scores, artist_cids, top_k)

        # ── exact_album: user named a specific album ──────────────────────────
        if mode == "exact_album" and intent.album_names:
            album_lower = [a.lower() for a in intent.album_names]
            album_cids = [
                tid for tid, a in zip(self._ids, self._album_names)
                if any(al in a for al in album_lower)
            ]
            if candidate_ids:
                album_cids = [t for t in album_cids if t in set(candidate_ids)]
            if album_cids:
                # All tracks from the album; no further ranking needed
                return {tid: 1.0 for tid in album_cids[:top_k]}
            # Fall back to BM25 album search
            query = " ".join(intent.album_names).lower().split()
            scores = self._bm25_album.get_scores(query)
            return self._scores_to_dict(scores, candidate_ids, top_k)

        # ── genre_mood / abstract: tag + year + mood matching ─────────────────
        if mode in ("genre_mood", "abstract"):
            tag_tokens = (
                intent.genres + intent.mood + intent.themes
            )
            if intent.decade:
                tag_tokens += intent.decade.lower().split()
            if intent.user_culture:
                tag_tokens += intent.user_culture.lower().split()
            if intent.listener_goal:
                # Only add content words from listener_goal, not stop words
                stop = {"a","the","and","or","in","of","to","for","with","that","this","is","it","from"}
                tag_tokens += [w for w in intent.listener_goal.lower().split() if w not in stop]
            if not tag_tokens:
                return {}
            scores = self._bm25_tag.get_scores([t.lower() for t in tag_tokens])
            return self._scores_to_dict(scores, candidate_ids, top_k)

        # ── default: full-text query (all fields) ─────────────────────────────
        tokens = (
            intent.artist_names + intent.track_names + intent.album_names
            + intent.genres + intent.mood + intent.themes
        )
        if intent.semantic_query:
            tokens += intent.semantic_query.split()
        if intent.listener_goal:
            stop = {"a","the","and","or","in","of","to","for","with","that","this","is","it","from"}
            tokens += [w for w in intent.listener_goal.lower().split() if w not in stop]
        if intent.user_culture:
            tokens += intent.user_culture.lower().split()
        if not tokens:
            return {}
        scores = self._bm25_full.get_scores([t.lower() for t in tokens])
        return self._scores_to_dict(scores, candidate_ids, top_k)

    @staticmethod
    def _classify_mode(intent: "ParsedIntent") -> str:
        """Infer retrieval mode from intent fields when not explicitly set."""
        if intent.track_names:
            return "exact_track"
        if intent.album_names and not intent.artist_names:
            return "exact_album"
        if intent.artist_names and not intent.track_names:
            return "exact_artist"
        if intent.genres or intent.mood:
            return "genre_mood"
        if intent.is_abstract:
            return "abstract"
        return "default"


# ─────────────────────────────────────────────────────────────────────────────
# 7. Dense Retriever  (FAISS)
# ─────────────────────────────────────────────────────────────────────────────

class DenseRetriever:
    """
    Multi-vector FAISS retrieval.

    Three embedding columns are indexed separately and fused via RRF:
      - attributes-qwen3_embedding_0.6b  (1024d) – musical attributes / style
      - metadata-qwen3_embedding_0.6b    (1024d) – artist / album / tags text
      - lyrics-qwen3_embedding_0.6b      (1024d) – lyrical themes / language

    At query time the same text encoder embeds the semantic_query and scores
    all three indices; results are merged with equal weights via RRF.
    """

    EMB_COLS = [
        "attributes-qwen3_embedding_0.6b",   # 1024d – musical style/mood
        "metadata-qwen3_embedding_0.6b",      # 1024d – artist/album/tag text
        "lyrics-qwen3_embedding_0.6b",        # 1024d – lyrical themes
        "audio-laion_clap",                   # 512d  – acoustic audio features
    ]
    # Weights for RRF fusion across indices.
    # audio-laion_clap gets a strong weight because it captures acoustic
    # similarity (tempo, timbre, energy) that text embeddings miss entirely.
    EMB_WEIGHTS = [0.35, 0.20, 0.15, 0.30]
    EMB_DIM = 1024   # default; audio-laion_clap uses 512 (handled per-index)

    # Keep EMB_COL alias so external code that references it still works
    EMB_COL = "attributes-qwen3_embedding_0.6b"

    def __init__(self, track_embeddings: pd.DataFrame):
        import faiss  # type: ignore

        self._ids: List[str] = track_embeddings["track_id"].tolist()
        self._id2idx: Dict[str, int] = {tid: i for i, tid in enumerate(self._ids)}

        self._indexes: List[Any] = []   # one FAISS index per embedding column
        self._vecs_list: List[np.ndarray] = []

        for col in self.EMB_COLS:
            if col not in track_embeddings.columns:
                # Column absent → store None, skip this index
                self._indexes.append(None)
                self._vecs_list.append(None)
                continue

            # Infer actual dimension from first non-empty row
            actual_dim = self.EMB_DIM
            for v in track_embeddings[col]:
                if isinstance(v, (list, np.ndarray)) and len(v) > 0:
                    actual_dim = len(v)
                    break

            def _to_vec(v, dim=actual_dim):
                if isinstance(v, (list, np.ndarray)) and len(v) > 0:
                    return np.asarray(v, dtype=np.float32)
                return np.zeros(dim, dtype=np.float32)

            vecs = np.vstack([_to_vec(v) for v in track_embeddings[col]])
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            vecs = (vecs / norms).astype(np.float32)

            idx = faiss.IndexFlatIP(vecs.shape[1])
            idx.add(vecs)
            self._indexes.append(idx)
            self._vecs_list.append(vecs)

        self._encoder = None   # lazy init

    # ── Query encoder ────────────────────────────────────────────────────────

    ENCODER_PATH = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-Embedding-0___6B"

    def _get_encoder(self):
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
                self._encoder = SentenceTransformer(
                    self.ENCODER_PATH,
                    local_files_only=True,
                )
                print(f"[DENSE] Loaded encoder from {self.ENCODER_PATH}")
            except Exception as e:
                print(f"[DENSE] Encoder unavailable ({e}). Using hash-based fallback.")

                class _HashEncoder:
                    def encode(self_, texts, normalize_embeddings=True):
                        out = []
                        for t in texts:
                            seed = abs(hash(t)) % (2 ** 31)
                            rng = np.random.default_rng(seed)
                            v = rng.normal(size=384).astype(np.float32)
                            if normalize_embeddings:
                                v /= np.linalg.norm(v) + 1e-9
                            out.append(v)
                        return np.array(out)

                self._encoder = _HashEncoder()
        return self._encoder

    # Qwen3-Embedding requires an instruction prefix for query encoding.
    # Documents (track embeddings) are stored WITHOUT prefix – so only the
    # query side needs it.  See: https://huggingface.co/Qwen/Qwen3-Embedding
    QUERY_INSTRUCTION = (
        "Instruct: Given a user music request, retrieve the most relevant tracks\n"
        "Query: "
    )

    def _encode(self, text: str, target_dim: int = None) -> np.ndarray:
        enc = self._get_encoder()
        prompted = self.QUERY_INSTRUCTION + text
        v = enc.encode([prompted], normalize_embeddings=True)[0].astype(np.float32)
        dim = target_dim or self.EMB_DIM
        if v.shape[0] < dim:
            v = np.pad(v, (0, dim - v.shape[0]))
        else:
            v = v[:dim]
        v /= np.linalg.norm(v) + 1e-9
        return v

    # ── Retrieval ────────────────────────────────────────────────────────────

    def _score_single_index(
        self,
        idx_pos: int,
        qv: np.ndarray,
        candidate_ids: Optional[List[str]],
        top_k: int,
    ) -> Dict[str, float]:
        """Score one FAISS index; returns {track_id: cosine_sim}."""
        faiss_idx = self._indexes[idx_pos]
        vecs      = self._vecs_list[idx_pos]
        if faiss_idx is None or vecs is None:
            return {}

        # Re-encode query at the correct dimension if it doesn't match the index
        index_dim = vecs.shape[1]
        if qv.shape[0] != index_dim:
            # The query was encoded at a different dimension – re-slice or re-encode
            if qv.shape[0] > index_dim:
                qv = qv[:index_dim]
            else:
                qv = np.pad(qv, (0, index_dim - qv.shape[0]))
            qv /= np.linalg.norm(qv) + 1e-9

        qv = qv.reshape(1, -1)
        if candidate_ids is None:
            scores, idxs = faiss_idx.search(qv, top_k)
            return {
                self._ids[i]: float(s)
                for s, i in zip(scores[0], idxs[0])
                if i >= 0
            }
        results = {
            tid: float(np.dot(qv[0], vecs[self._id2idx[tid]]))
            for tid in candidate_ids
            if tid in self._id2idx
        }
        return dict(sorted(results.items(), key=lambda x: x[1], reverse=True)[:top_k])

    def _score_candidates(
        self, qv: np.ndarray, candidate_ids: Optional[List[str]], top_k: int
    ) -> Dict[str, float]:
        """Fuse all available embedding indices with RRF."""
        ranked_lists = []
        weights = []
        for i, w in enumerate(self.EMB_WEIGHTS):
            scores = self._score_single_index(i, qv, candidate_ids, top_k)
            if scores:
                ranked_lists.append(scores)
                weights.append(w)
        if not ranked_lists:
            return {}
        # Inline RRF (avoid circular import with module-level rrf_fusion)
        fused: Dict[str, float] = {}
        for rl, w in zip(ranked_lists, weights):
            for rank, (doc_id, _) in enumerate(
                sorted(rl.items(), key=lambda x: x[1], reverse=True), start=1
            ):
                fused[doc_id] = fused.get(doc_id, 0.0) + w / (60 + rank)
        return dict(sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k])

    def retrieve(
        self,
        intent: ParsedIntent,
        candidate_ids: Optional[List[str]] = None,
        top_k: int = TOP_K_RETRIEVE,
    ) -> Dict[str, float]:
        if not intent.semantic_query:
            return {}
        qv = self._encode(intent.semantic_query)
        return self._score_candidates(qv, candidate_ids, top_k)

    def retrieve_by_example(
        self,
        pos_ids: List[str],
        neg_ids: List[str],
        candidate_ids: Optional[List[str]] = None,
        top_k: int = TOP_K_RETRIEVE,
    ) -> Dict[str, float]:
        """
        Score each available index independently using the centroid of played
        tracks, then fuse with RRF. This is better than using only index 0
        because different indices capture different aspects (style, lyrics, audio).
        """
        all_scores: List[Dict[str, float]] = []
        all_weights: List[float] = []

        for i, (vecs_mat, w) in enumerate(zip(self._vecs_list, self.EMB_WEIGHTS)):
            if vecs_mat is None:
                continue

            def centroid(ids, vm=vecs_mat):
                vs = [vm[self._id2idx[j]] for j in ids if j in self._id2idx]
                if not vs:
                    return None
                c = np.mean(vs, axis=0).astype(np.float32)
                c /= np.linalg.norm(c) + 1e-9
                return c

            pos_v = centroid(pos_ids)
            if pos_v is None:
                continue
            neg_v = centroid(neg_ids)
            qv = pos_v - 0.3 * neg_v if neg_v is not None else pos_v
            qv /= np.linalg.norm(qv) + 1e-9

            scores = self._score_single_index(i, qv, candidate_ids, top_k)
            if scores:
                all_scores.append(scores)
                all_weights.append(w)

        if not all_scores:
            return {}

        # Fuse across indices
        fused: Dict[str, float] = {}
        for rl, w in zip(all_scores, all_weights):
            for rank, (doc_id, _) in enumerate(
                sorted(rl.items(), key=lambda x: x[1], reverse=True), 1
            ):
                fused[doc_id] = fused.get(doc_id, 0.0) + w / (60 + rank)
        return dict(sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k])


# ─────────────────────────────────────────────────────────────────────────────
# 8. Hybrid RRF Fusion
# ─────────────────────────────────────────────────────────────────────────────

def rrf_fusion(
    ranked_lists: List[Dict[str, float]],
    weights: List[float],
    k: int = 60,
) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for rl, w in zip(ranked_lists, weights):
        for rank, (doc_id, _) in enumerate(
            sorted(rl.items(), key=lambda x: x[1], reverse=True), start=1
        ):
            scores[doc_id] = scores.get(doc_id, 0.0) + w / (k + rank)
    return scores



# ─────────────────────────────────────────────────────────────────────────────
# 9b. Cold User Handler
# ─────────────────────────────────────────────────────────────────────────────

class ColdUserHandler:
    """
    Provides a CF-BPR proxy vector for cold users (not in User-Embeddings).

    Strategy:
      1. Group warm users by preferred_musical_culture.
      2. For each culture, compute the average CF vector ("culture centroid").
      3. When a cold user's culture is known, return that centroid as their
         proxy vector so they get personalised CF ranking instead of falling
         back to pure RRF.
      4. If culture is unknown, fall back to a global popularity-weighted
         centroid ("global mean") which at least promotes well-liked tracks.

    Additionally provides a culture→BM25-tokens mapping so retrieval can be
    seeded with genre/culture keywords for cold users.
    """

    def __init__(
        self,
        user_meta: pd.DataFrame,
        user_emb: pd.DataFrame,
    ):
        # Build user_id → metadata lookup
        self._user_culture: Dict[str, str] = {}
        self._user_age_group: Dict[str, str] = {}
        self._user_country: Dict[str, str] = {}

        for _, row in user_meta.iterrows():
            uid = str(row.get("user_id", "") or "")
            if not uid:
                continue
            self._user_culture[uid]   = str(row.get("preferred_musical_culture") or "").strip()
            self._user_age_group[uid] = str(row.get("age_group") or "").strip()
            self._user_country[uid]   = str(row.get("country_code") or "").strip()

        # Build culture → list of CF vectors from warm users
        user_cf_map: Dict[str, np.ndarray] = {
            row["user_id"]: np.array(row["cf-bpr"], dtype=np.float32)
            for _, row in user_emb[["user_id", "cf-bpr"]].iterrows()
            if isinstance(row["cf-bpr"], (list, np.ndarray)) and len(row["cf-bpr"]) > 0
        }

        culture_vecs: Dict[str, List[np.ndarray]] = {}
        for uid, culture in self._user_culture.items():
            if uid in user_cf_map and culture:
                culture_vecs.setdefault(culture, []).append(user_cf_map[uid])

        # Pre-compute culture centroids
        self._culture_centroids: Dict[str, np.ndarray] = {}
        for culture, vecs in culture_vecs.items():
            c = np.mean(vecs, axis=0).astype(np.float32)
            c /= np.linalg.norm(c) + 1e-9
            self._culture_centroids[culture] = c

        # Global centroid as last-resort fallback
        all_vecs = list(user_cf_map.values())
        if all_vecs:
            g = np.mean(all_vecs, axis=0).astype(np.float32)
            g /= np.linalg.norm(g) + 1e-9
            self._global_centroid: Optional[np.ndarray] = g
        else:
            self._global_centroid = None

        n_cultures = len(self._culture_centroids)
        print(f"[ColdUser] {len(user_cf_map)} warm users, "
              f"{n_cultures} culture centroids built.")

    def get_proxy_vector(
        self,
        user_id: str,
        override_culture: str = "",
    ) -> Optional[np.ndarray]:
        """
        Return a CF proxy vector for a cold user.
        override_culture: culture string from user_profile (may be richer than
                          what's stored in User-Metadata for challenge users).
        """
        # Try the provided culture first (from blind_df.user_profile)
        culture = override_culture or self._user_culture.get(user_id, "")

        if culture and culture in self._culture_centroids:
            return self._culture_centroids[culture]

        # Fuzzy match: find the closest culture by shared words
        if culture:
            culture_words = set(culture.lower().split())
            best_score, best_vec = 0, None
            for c_key, c_vec in self._culture_centroids.items():
                overlap = len(culture_words & set(c_key.lower().split()))
                if overlap > best_score:
                    best_score, best_vec = overlap, c_vec
            if best_vec is not None and best_score >= 1:
                return best_vec

        return self._global_centroid

    def get_culture(self, user_id: str) -> str:
        return self._user_culture.get(user_id, "")

    def get_culture_tokens(self, user_id: str, override: str = "") -> List[str]:
        """Return BM25-friendly tokens from the user's musical culture."""
        culture = override or self._user_culture.get(user_id, "")
        return culture.lower().split() if culture else []

# ─────────────────────────────────────────────────────────────────────────────
# 9. CF-BPR Ranker
# ─────────────────────────────────────────────────────────────────────────────

class CFBPRRanker:
    """
    final = alpha * norm(rrf_score) + beta * dot(user_cf_bpr, track_cf_bpr)
    Cold users (not in embedding table) → pure RRF ordering.
    """

    def __init__(
        self,
        track_embeddings: pd.DataFrame,
        user_embeddings: pd.DataFrame,
        alpha: float = 0.5,
        beta: float = 0.5,
    ):
        self.alpha = alpha
        self.beta  = beta

        self._track_cf: Dict[str, np.ndarray] = {
            row["track_id"]: np.array(row["cf-bpr"], dtype=np.float32)
            for _, row in track_embeddings[["track_id", "cf-bpr"]].iterrows()
            if isinstance(row["cf-bpr"], (list, np.ndarray)) and len(row["cf-bpr"]) > 0
        }
        self._user_cf: Dict[str, np.ndarray] = {
            row["user_id"]: np.array(row["cf-bpr"], dtype=np.float32)
            for _, row in user_embeddings[["user_id", "cf-bpr"]].iterrows()
            if isinstance(row["cf-bpr"], (list, np.ndarray)) and len(row["cf-bpr"]) > 0
        }

    def rank(
        self,
        user_id: str,
        rrf_scores: Dict[str, float],
        top_k: int = TOP_K_FINAL,
        cold_handler: Optional["ColdUserHandler"] = None,
        user_culture: str = "",
    ) -> List[ScoredTrack]:
        if not rrf_scores:
            return []

        # Warm user: use their own CF vector
        # Cold user: use culture-centroid proxy from ColdUserHandler
        user_v = self._user_cf.get(user_id)
        is_cold = user_v is None
        if is_cold and cold_handler is not None:
            user_v = cold_handler.get_proxy_vector(user_id, override_culture=user_culture)

        max_rrf = max(rrf_scores.values()) or 1.0

        # Compute raw CF scores
        raw_cf: Dict[str, float] = {}
        if user_v is not None:
            for tid in rrf_scores:
                if tid in self._track_cf:
                    raw_cf[tid] = float(np.dot(user_v, self._track_cf[tid]))

        # Normalise to [0, 1] over the candidate set
        if raw_cf:
            cf_min = min(raw_cf.values())
            cf_max = max(raw_cf.values())
            cf_range = cf_max - cf_min or 1.0
            norm_cf = {tid: (s - cf_min) / cf_range for tid, s in raw_cf.items()}
        else:
            norm_cf = {}

        # Cold user with proxy: use slightly lower beta to avoid over-relying
        # on a centroid that may not perfectly represent the individual.
        # if is_cold and norm_cf:
        #     effective_alpha = 0.65
        #     effective_beta  = 0.35
        # elif not norm_cf:
        #     effective_alpha = 1.0
        #     effective_beta  = 0.0
        # else:
        #     effective_alpha = self.alpha
        #     effective_beta  = self.beta


        # --- 【方案三：激进重排策略】 ---
        # 逻辑：如果用户有历史行为（非冷启动），且 CF 模型能打分，则给 CF 高权重
        
        # 默认：如果有 CF 分数，就给高权重给 CF (0.7)，让个性化排序主导
        base_alpha = 0.3
        base_beta = 0.7
        
        # 1. 如果是冷用户（完全没历史），只能依赖 RRF
        if is_cold:
            effective_alpha = 1.0
            effective_beta = 0.0
            # 注：这里也可以保留 culture centroid，但为了稳定先关掉 CF
        else:
            # 2. 温用户/热用户：极度信任 CF 模型的排序能力
            effective_alpha = base_alpha  # 0.3
            effective_beta = base_beta    # 0.7
        
        # 强制归一化权重和为 1
        total = effective_alpha + effective_beta
        effective_alpha /= total
        effective_beta /= total
        # --- 【修改结束】 ---

        results: List[ScoredTrack] = []
        for tid, rrf_s in rrf_scores.items():
            norm_rrf = rrf_s / max_rrf
            cf_s     = norm_cf.get(tid, 0.0)
            results.append(
                ScoredTrack(
                    track_id=tid,
                    rrf_score=rrf_s,
                    cf_score=cf_s,
                    final_score=effective_alpha * norm_rrf + effective_beta * cf_s,
                )
            )

        results.sort(key=lambda x: x.final_score, reverse=True)
        return results[:top_k]


# ─────────────────────────────────────────────────────────────────────────────
# 10. Full Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class MusicRecPipeline:
    """End-to-end conversational music recommendation pipeline."""

    def __init__(
        self,
        track_meta: pd.DataFrame,
        track_emb: pd.DataFrame,
        user_emb: pd.DataFrame,
        user_meta: Optional[pd.DataFrame] = None,
        sessions_df: Optional[pd.DataFrame] = None,
        cache_file: Path = DEFAULT_CACHE_FILE,
        deepseek_api_key: str = DEEPSEEK_API_KEY,
    ):
        print("[INIT] IntentCache …")
        self.cache   = IntentCache(cache_file)

        print("[INIT] FilterEngine …")
        self.filter  = FilterEngine(track_meta)

        print("[INIT] BM25 …")
        self.bm25    = BM25Retriever(track_meta)

        print("[INIT] FAISS dense index …")
        self.dense   = DenseRetriever(track_emb)

        print("[INIT] CF-BPR ranker …")
        self.ranker  = CFBPRRanker(track_emb, user_emb)

        # UserHistoryIndex: per-user taste profile from training sessions
        if sessions_df is not None and len(sessions_df) > 0:
            print("[INIT] UserHistoryIndex …")
            self.user_history: Optional[UserHistoryIndex] = UserHistoryIndex(
                sessions_df, track_meta
            )
        else:
            print("[INIT] UserHistoryIndex skipped (no sessions_df provided)")
            self.user_history: Optional[UserHistoryIndex] = None

        # Cold user handler (for users with NO training history)
        if user_meta is not None and len(user_meta) > 0:
            print("[INIT] ColdUserHandler …")
            self.cold_handler: Optional[ColdUserHandler] = ColdUserHandler(
                user_meta, user_emb
            )
        else:
            print("[INIT] ColdUserHandler skipped (no user_meta provided)")
            self.cold_handler = None

        self._meta   = track_meta.set_index("track_id")
        self._api_key = deepseek_api_key
        print("[INIT] Pipeline ready.\n")

    # ── Factory ──────────────────────────────────────────────────────────────

    @classmethod
    def from_local(
        cls,
        data_root: Path = DEFAULT_DATA_ROOT,
        cache_file: Path = DEFAULT_CACHE_FILE,
        deepseek_api_key: str = DEEPSEEK_API_KEY,
    ) -> "MusicRecPipeline":
        dfs = load_all_data(data_root)
        return cls(
            track_meta=dfs["track_meta"],
            track_emb=dfs["track_emb"],
            user_emb=dfs["user_emb"],
            user_meta=dfs.get("user_meta"),
            sessions_df=dfs.get("sessions"),
            cache_file=cache_file,
            deepseek_api_key=deepseek_api_key,
        )

    # ── Recommend ────────────────────────────────────────────────────────────

    def recommend(
        self,
        user_id: str,
        conversation: List[Dict[str, str]],
        top_k: int = TOP_K_FINAL,
        verbose: bool = True,
        played_track_ids: Optional[List[str]] = None,
        listener_goal: str = "",
        user_culture: str = "",
    ) -> Tuple[List[ScoredTrack], ParsedIntent]:
        t0 = time.time()

        # 1. Intent
        intent, from_cache = parse_intent(conversation, self.cache, self._api_key)

        # Directly inject played tracks (from current session)
        if played_track_ids:
            intent.positive_track_ids = list(dict.fromkeys(
                intent.positive_track_ids + played_track_ids
            ))

        # Inject session-level metadata
        if listener_goal:
            intent.listener_goal = listener_goal
        if user_culture:
            intent.user_culture = user_culture

        # ── Enrich intent with user training history (warm users) ────────────
        if self.user_history:
            intent = self.user_history.enrich_intent(user_id, intent)
            if verbose and self.user_history.has_history(user_id):
                n_hist = len(self.user_history.get_accepted_tracks(user_id))
                top_a  = self.user_history.get_top_artists(user_id, 3)
                print(f"[1b] UserHistory  {n_hist} accepted  top_artists={top_a}")


        # Classify retrieval mode from intent structure
        if not intent.retrieval_mode:
            intent.retrieval_mode = BM25Retriever._classify_mode(intent)

        src = "cache" if from_cache else "API"
        if verbose:
            print(f"[1] Intent ({src})  mode={intent.retrieval_mode}  "
                  f"abstract={intent.is_abstract}  decade={intent.decade}  "
                  f"artists={intent.artist_names}  albums={intent.album_names}  "
                  f"tracks={intent.track_names}  genres={intent.genres}")

        # 2. Filter
        filtered = self.filter.apply(intent)
        # Use None (= full corpus) only when the filter is completely inactive
        # (i.e. no filter fields were set). When filter IS active but returns
        # the full set after relaxation, still use None to avoid passing a
        # 47k-element list everywhere.
        filter_active = any([
            intent.year_min, intent.year_max,
            intent.popularity_min, intent.popularity_max,
            intent.duration_min_ms, intent.duration_max_ms,
        ])
        if filter_active and len(filtered) < len(self.filter.df):
            cids: Optional[List[str]] = filtered["track_id"].tolist()
        else:
            cids = None
        if verbose:
            n_total = len(self.filter.df)
            n_pass  = len(filtered) if cids is not None else n_total
            print(f"[2] Filter  {n_pass:,}/{n_total:,} tracks"
                  + (" (no filter active)" if cids is None else ""))

        # 3. BM25
        bm25_scores = self.bm25.retrieve(intent, candidate_ids=cids,
                                          top_k=TOP_K_RETRIEVE)
        if verbose:
            print(f"[3] BM25    {len(bm25_scores)} candidates")

        # 4. Dense
        dense_scores: Dict[str, float] = {}

        # 4a. Build enriched semantic query:
        #     listener_goal (gold label) > semantic_query (DeepSeek) > fallback
        parts = []
        if intent.listener_goal:
            parts.append(intent.listener_goal)
        if intent.semantic_query and intent.semantic_query not in parts:
            parts.append(intent.semantic_query)
        if intent.user_culture:
            parts.append(intent.user_culture)
        enriched_query = " ".join(parts).strip()

        if enriched_query:
            # Temporarily override semantic_query for this call
            orig_sq = intent.semantic_query
            intent.semantic_query = enriched_query
            dense_scores = self.dense.retrieve(intent, candidate_ids=cids,
                                               top_k=TOP_K_RETRIEVE)
            intent.semantic_query = orig_sq

        # 4b. Example-based retrieval from played tracks (strongest recall signal).
        #     Run even when semantic_query is empty.
        if intent.positive_track_ids:
            ex_scores = self.dense.retrieve_by_example(
                intent.positive_track_ids, intent.negative_track_ids,
                candidate_ids=cids, top_k=TOP_K_RETRIEVE,
            )
            if dense_scores:
                # Fuse via RRF rather than max() so both signals contribute
                fused: Dict[str, float] = {}
                for rl, w in [
                    (dense_scores, 0.4),   # semantic query
                    (ex_scores,    0.6),   # example similarity (higher weight)
                ]:
                    for rank, (tid, _) in enumerate(
                        sorted(rl.items(), key=lambda x: x[1], reverse=True), 1
                    ):
                        fused[tid] = fused.get(tid, 0.0) + w / (60 + rank)
                dense_scores = fused
            else:
                dense_scores = ex_scores

        if verbose:
            print(f"[4] Dense   {len(dense_scores)} candidates  "
                  f"(played={len(intent.positive_track_ids)} example tracks)")

        # 5. RRF – fuse BM25 and Dense
        # bw = 0.2 if intent.is_abstract else BM25_WEIGHT
        # dw = 0.8 if intent.is_abstract else DENSE_WEIGHT
        # rrf_scores = rrf_fusion([bm25_scores, dense_scores], [bw, dw])

        # --- 【方案二修改开始】 --- 
        # 默认权重
        bm25_weight = 0.4
        dense_weight = 0.6
        
        # 1. 如果是精确匹配查询（指定了歌手、歌名或专辑），大幅提升 BM25 权重
        if intent.retrieval_mode in ["exact_track", "exact_artist", "exact_album"]:
            bm25_weight = 0.7  # 精确查询时，关键词匹配最重要
            dense_weight = 0.3
        
        # 2. 如果是抽象查询（只说心情、流派），则信任语义向量
        elif intent.retrieval_mode in ["abstract", "genre_mood"]:
            bm25_weight = 0.3
            dense_weight = 0.7
        
        # 3. 应用融合 (替换原来的 rrf_fusion 调用)
        rrf_scores = rrf_fusion([bm25_scores, dense_scores], [bm25_weight, dense_weight])
        # --- 【方案二修改结束】 --- 
        
        if verbose:
            print(f"[5] RRF     {len(rrf_scores)} candidates  "
                  f"(bm25:{bw:.1f} dense:{dw:.1f})")

        # 6. CF-BPR rerank
        # For cold users: augment BM25 with culture tokens if not already present
        # if self.cold_handler and (user_id not in self.ranker._user_cf):
        #     culture_tokens = self.cold_handler.get_culture_tokens(
        #         user_id, override=user_culture
        #     )
        #     if culture_tokens and not intent.user_culture:
        #         # Re-run BM25 with culture tokens injected (only if not already done)
        #         intent.user_culture = " ".join(culture_tokens)
        #         bm25_scores = self.bm25.retrieve(intent, candidate_ids=cids,
        #                                           top_k=TOP_K_RETRIEVE)
        #         rrf_scores = rrf_fusion([bm25_scores, dense_scores], [bw, dw])
        #         if verbose:
        #             print(f"[3b] BM25 re-run with cold-user culture tokens")

        # 【修改】：仅在非精确查询时注入文化特征
        is_cold = (user_id not in self.ranker._user_cf)
        if self.cold_handler and is_cold:
            # 仅当查询不是针对特定歌曲/专辑时，才使用文化特征扩充
            if intent.retrieval_mode in ["abstract", "genre_mood", "default"]:
                culture_tokens = self.cold_handler.get_culture_tokens(user_id, override=user_culture)
                if culture_tokens and not intent.user_culture:
                    intent.user_culture = " ".join(culture_tokens)
                    # 重新检索，加入文化偏好
                    bm25_scores = self.bm25.retrieve(intent, candidate_ids=cids, top_k=TOP_K_RETRIEVE)
                    rrf_scores = rrf_fusion([bm25_scores, dense_scores], [bm25_weight, dense_weight])
                    if verbose: print(f"[3b] BM25 re-run with cold-user culture tokens")

        n_played = len(intent.positive_track_ids)
        ranked = self.ranker.rank(
            user_id, rrf_scores,
            top_k=top_k + n_played + 5,
            cold_handler=self.cold_handler,
            user_culture=user_culture,
        )

        # Remove already-played AND historically rejected tracks
        exclude_set = set(intent.positive_track_ids)
        if self.user_history:
            exclude_set |= self.user_history.get_rejected_tracks(user_id)
        if exclude_set:
            ranked = [st for st in ranked if st.track_id not in exclude_set]
        ranked = ranked[:top_k]



        if verbose:
            print(f"[6] Rerank  top-{len(ranked)}  "
                  f"({time.time()-t0:.2f}s total)\n")

        # Attach display metadata
        def _first(v, default="?"):
            """Return first element of a list/ndarray field, or default."""
            if v is None:
                return default
            if isinstance(v, (list, np.ndarray)):
                return str(v[0]) if len(v) > 0 else default
            s = str(v).strip()
            return s if s not in ("nan", "None", "") else default

        def _to_list(v):
            """Coerce list/ndarray field to a plain Python list."""
            if v is None:
                return []
            if isinstance(v, np.ndarray):
                return v.tolist()
            if isinstance(v, list):
                return v
            return []

        for st in ranked:
            if st.track_id in self._meta.index:
                r = self._meta.loc[st.track_id]
                st.metadata = {
                    "track_name":   _first(r["track_name"]),
                    "artist_name":  _first(r["artist_name"]),
                    "release_date": str(r.get("release_date", "") or "")[:10],
                    "popularity":   float(r.get("popularity", 0) or 0),
                    "tags":         _to_list(r.get("tag_list"))[:5],
                }

        return ranked, intent

    def format_results(self, ranked: List[ScoredTrack]) -> str:
        lines = []
        for i, st in enumerate(ranked, 1):
            m  = st.metadata
            yr = m.get("release_date", "")[:4]
            lines.append(
                f"{i:2d}. {m.get('track_name','?'):40s}"
                f"  {m.get('artist_name','?'):25s}"
                f"  {yr}  pop={m.get('popularity',0):3.0f}"
                f"  rrf={st.rrf_score:.4f}"
                f"  cf={st.cf_score:+.3f}"
                f"  score={st.final_score:.4f}"
            )
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 11. Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    pipeline: MusicRecPipeline,
    sessions: pd.DataFrame,
    k: int = 10,
    max_sessions: int = 100,
) -> Dict[str, float]:
    """
    Evaluate hit-rate@k over a set of sessions.
    Ground truth: 'music' role turns contain the oracle track_id.
    """
    hit_rates = []
    n = 0
    for _, sess in sessions.iterrows():
        if n >= max_sessions:
            break
        n += 1
        user_id = sess["user_id"]
        turns = sess["conversations"]
        history: List[Dict[str, str]] = []
        hits = []
        for t in turns:
            role = t["role"]
            content = t["content"]
            if role == "user":
                history.append({"role": "user", "content": content})
            elif role == "music" and content:
                ranked, _ = pipeline.recommend(user_id, history,
                                               top_k=k, verbose=False)
                pids = [st.track_id for st in ranked]
                hits.append(1.0 if content in pids else 0.0)
                history.append({"role": "assistant",
                                 "content": f"[played:{content}]"})
        if hits:
            hit_rates.append(float(np.mean(hits)))

    return {
        "hit_rate@k": float(np.mean(hit_rates)) if hit_rates else 0.0,
        "n_sessions": len(hit_rates),
        "k": k,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 12. CLI
# ─────────────────────────────────────────────────────────────────────────────

def _demo_synthetic():
    """
    Offline smoke-test with 4 synthetic tracks.
    Does NOT require real data files.
    """
    rng = np.random.default_rng(0)

    tracks = pd.DataFrame([
        {"track_id": "t1", "track_name": ["American Idiot"],
         "artist_name": ["Green Day"], "album_name": ["American Idiot"],
         "tag_list": ["punk rock", "alternative", "00s"],
         "popularity": 82.0, "release_date": "2004-09-21", "duration": 172800},
        {"track_id": "t2", "track_name": ["Basket Case"],
         "artist_name": ["Green Day"], "album_name": ["Dookie"],
         "tag_list": ["punk rock", "alternative", "90s"],
         "popularity": 88.0, "release_date": "1994-02-01", "duration": 181000},
        {"track_id": "t3", "track_name": ["With Rainy Eyes"],
         "artist_name": ["Emancipator"], "album_name": ["Soon It Will Be Cold Enough"],
         "tag_list": ["ambient", "downtempo", "chill", "instrumental"],
         "popularity": 39.0, "release_date": "2006-12-06", "duration": 300920},
        {"track_id": "t4", "track_name": ["Boulevard of Broken Dreams"],
         "artist_name": ["Green Day"], "album_name": ["American Idiot"],
         "tag_list": ["punk rock", "alternative", "00s"],
         "popularity": 92.0, "release_date": "2004-10-18", "duration": 261600},
    ])

    track_emb = pd.DataFrame([
        {"track_id": r["track_id"],
         "attributes-qwen3_embedding_0.6b": rng.normal(size=1024).tolist(),
         "cf-bpr": rng.normal(size=128).tolist(),
         "audio-laion_clap": [],
         "image-siglip2": [],
         "lyrics-qwen3_embedding_0.6b": [],
         "metadata-qwen3_embedding_0.6b": []}
        for _, r in tracks.iterrows()
    ])

    user_emb = pd.DataFrame([
        {"user_id": "u1", "cf-bpr": rng.normal(size=128).tolist()}
    ])

    user_meta = pd.DataFrame([
        {"user_id": "u1", "age": 22, "age_group": "20s", "country_code": "US",
         "preferred_musical_culture": "Alternative Rock"},
    ])
    cache_file = Path("./cache/intent_demo.jsonl")
    pipeline = MusicRecPipeline(tracks, track_emb, user_emb,
                                user_meta=user_meta, cache_file=cache_file)

    print("=" * 65)
    print("TEST 1 — concrete: Green Day 2000s")
    print("=" * 65)
    conv1 = [{"role": "user",
              "content": "I want some early 2000s punk rock – Green Day vibes."}]
    r1, i1 = pipeline.recommend("u1", conv1, top_k=4)
    print(pipeline.format_results(r1))

    print("\n" + "=" * 65)
    print("TEST 2 — abstract: chill rainy day")
    print("=" * 65)
    conv2 = [{"role": "user",
              "content": "Something mellow and chill for a rainy afternoon."}]
    r2, i2 = pipeline.recommend("u1", conv2, top_k=4)
    print(pipeline.format_results(r2))

    # Second run – should be served from cache
    print("\n" + "=" * 65)
    print("TEST 1 again (expect: from_cache = True)")
    print("=" * 65)
    r3, i3 = pipeline.recommend("u1", conv1, top_k=4)
    print(pipeline.format_results(r3))

    print("\n[OK] Demo done. Cache written to", cache_file)



# ─────────────────────────────────────────────────────────────────────────────
# 12. Blind-A Prediction  (generate submission JSON)
# ─────────────────────────────────────────────────────────────────────────────

RESPONSE_SYSTEM_PROMPT = """
You are a music recommendation assistant having a natural conversation.
Given a conversation history and recommended tracks, write a response that:
1. Directly and warmly addresses what the user asked for.
2. Mentions 2-3 specific track or artist names from the list to feel concrete.
3. Explains WHY these fit (mood, genre, era, energy, lyrical theme).
4. Invites further refinement if needed ("Let me know if you want more like X").
Keep it conversational, 3-5 sentences, under 100 words. No bullet points.
"""


def _generate_response(
    conversation: List[Dict[str, str]],
    ranked: List[ScoredTrack],
    api_key: str = DEEPSEEK_API_KEY,
) -> str:
    """
    Ask DeepSeek to produce a natural-language response for the recommendation.
    Falls back to a richer template string if the API is unavailable.
    """
    # Build track list with year and tags for richer context (up to 8 tracks)
    track_lines = []
    for st in ranked[:8]:
        m      = st.metadata
        name   = m.get("track_name", "?")
        artist = m.get("artist_name", "")
        year   = m.get("release_date", "")[:4]
        tags   = ", ".join(m.get("tags", [])[:3])
        line   = f"- {name} by {artist}" + (f" ({year})" if year else "")
        if tags:
            line += f"  [{tags}]"
        track_lines.append(line)
    tracks_block = "\n".join(track_lines) if track_lines else "- (no tracks)"

    # conversation here is already response_history (clean user/assistant only).
    # Take last 6 turns; skip any accidentally injected non-natural-language lines.
    _UUID_RE = re.compile(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I
    )
    conv_text = "\n".join(
        f"{m['role'].capitalize()}: {m['content']}"
        for m in conversation[-6:]
        if m.get("content", "").strip()
        and not m["content"].startswith("[Previously played")
        and not _UUID_RE.search(m["content"])   # drop any UUID-containing lines
    )

    prompt = (
        f"Conversation so far:\n{conv_text}\n\n"
        f"Recommended tracks:\n{tracks_block}\n\n"
        "Write a natural recommendation response to continue this conversation."
    )

    try:
        import requests  # type: ignore
        resp = requests.post(
            f"{DEEPSEEK_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": RESPONSE_SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                "max_tokens": 180,
                "temperature": 0.6,
            },
            timeout=20,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        # Richer fallback using actual track metadata
        if ranked:
            m0     = ranked[0].metadata
            name0  = m0.get("track_name", "these tracks")
            artist0= m0.get("artist_name", "")
            tags0  = m0.get("tags", [])
            style  = ", ".join(tags0[:2]) if tags0 else "this style"
            opener = f"I've picked {name0}" + (f" by {artist0}" if artist0 else "")
        else:
            opener = "I've curated a selection"
            style  = "your taste"
        return (
            f"{opener} and similar tracks that match the {style} vibe you're after. "
            f"These should capture exactly what you described — "
            f"let me know if you'd like me to adjust the mood or explore a different direction!"
        )


def _build_conversation_history(
    turns: List[Dict],
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Convert blind-A conversation turns into two separate history lists:

    intent_history  – used for DeepSeek intent parsing.
                      Includes user turns, assistant turns, AND the 'thought'
                      field from music turns (internal reasoning about why a
                      track was chosen – useful context for understanding intent).

    response_history – used for DeepSeek response generation.
                       Only clean natural-language user ↔ assistant exchanges.
                       NO thought strings, NO UUID-containing lines.
                       This keeps the response prompt uncluttered.
    """
    intent_history: List[Dict[str, str]]   = []
    response_history: List[Dict[str, str]] = []

    for t in turns:
        role    = t.get("role", "")
        content = (t.get("content", "") or "").strip()
        thought = (t.get("thought", "") or "").strip()

        if role == "user" and content:
            intent_history.append({"role": "user", "content": content})
            response_history.append({"role": "user", "content": content})

        elif role == "assistant" and content:
            intent_history.append({"role": "assistant", "content": content})
            response_history.append({"role": "assistant", "content": content})

        elif role == "music":
            # thought → intent context only (may contain UUIDs / internal notes)
            if thought:
                intent_history.append({"role": "assistant", "content": thought})
            # response_history: skip entirely (track_id + thought not user-facing)

        # system turns: skip both

    return intent_history, response_history


def _extract_positive_ids_from_history(turns: List[Dict]) -> List[str]:
    """
    Collect track_ids from 'music' role turns that appear BEFORE the current
    position – these are songs already played / implicitly accepted by the user.
    """
    ids = []
    for t in turns:
        if t.get("role") == "music":
            content = (t.get("content") or "").strip()
            # Validate UUID-like format
            if content and re.match(
                r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
                content, re.I
            ):
                ids.append(content)
    return ids


def predict_blind_a(
    pipeline: "MusicRecPipeline",
    blind_df: pd.DataFrame,
    output_path: Path,
    top_k: int = 20,
    generate_response: bool = True,
    verbose: bool = True,
) -> List[Dict]:
    """
    For every session in blind_df, determine the current turn_number and
    generate a prediction entry:

      {
        "session_id": "<uuid>",
        "user_id":    "<uuid>",
        "turn_number": <int>,           # len(conversations) // 3 + 1
        "predicted_track_ids": [...],   # up to 20 track UUIDs
        "predicted_response": "..."     # natural-language explanation
      }

    Results are written atomically to output_path as a JSON array.
    Progress is shown via tqdm (falls back to plain print if not installed).
    """
    try:
        from tqdm import tqdm  # type: ignore
        _tqdm_available = True
    except ImportError:
        _tqdm_available = False

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    predictions: List[Dict] = []

    rows = list(blind_df.iterrows())
    iterator = (
        tqdm(rows, desc="Predicting", unit="session", dynamic_ncols=True)
        if _tqdm_available
        else rows
    )

    t_start = time.time()

    for idx, (_, row) in enumerate(iterator, 1):
        session_id = row["session_id"]
        user_id    = row["user_id"]
        turns      = row["conversations"]   # list of dicts

        # ── turn_number ───────────────────────────────────────────────────────
        turn_number = len(turns) // 3 + 1

        # Update tqdm postfix with current session info
        if _tqdm_available:
            iterator.set_postfix(
                session=session_id[:8],
                turn=turn_number,
                cached=len(pipeline.cache),
                refresh=False,
            )

        if verbose and not _tqdm_available:
            n_sessions = len(rows)
            elapsed = time.time() - t_start
            eta = (elapsed / idx) * (n_sessions - idx) if idx > 1 else 0
            print(f"[{idx}/{n_sessions}] session={session_id[:8]}  "
                  f"user={user_id[:8]}  turns_raw={len(turns)}  "
                  f"turn_number={turn_number}  "
                  f"elapsed={elapsed:.0f}s  eta={eta:.0f}s")

        # ── Build conversation context ────────────────────────────────────
        intent_history, response_history = _build_conversation_history(turns)

        if not intent_history:
            # No user turn at all – skip gracefully
            msg = f"[SKIP] No user turns in session {session_id}"
            if _tqdm_available:
                tqdm.write(msg)
            else:
                print(msg)
            continue

        # Extract previously played tracks and pass them directly into recommend()
        # rather than embedding them as a text hint (which DeepSeek often misses).
        played_ids = _extract_positive_ids_from_history(turns)

        # ── Extract session-level metadata (not in conversation text) ──────
        user_profile = row.get("user_profile") or {}
        if isinstance(user_profile, str):
            try: user_profile = json.loads(user_profile)
            except: user_profile = {}
        conv_goal = row.get("conversation_goal") or {}
        if isinstance(conv_goal, str):
            try: conv_goal = json.loads(conv_goal)
            except: conv_goal = {}

        listener_goal = str(conv_goal.get("listener_goal") or "").strip()
        user_culture  = str(user_profile.get("preferred_musical_culture") or "").strip()

        # ── Recommend ─────────────────────────────────────────────────────────
        try:
            ranked, intent = pipeline.recommend(
                user_id, intent_history,
                top_k=top_k,
                verbose=False,
                played_track_ids=played_ids if played_ids else None,
                listener_goal=listener_goal,
                user_culture=user_culture,
            )
        except Exception as e:
            msg = f"[ERROR] recommend() failed for {session_id[:8]}: {e}"
            if _tqdm_available:
                tqdm.write(msg)
            else:
                print(msg)
            ranked, intent = [], None

        # ranked comes from CFBPRRanker which iterates a dict → no duplicates
        # within a single call. We also deduplicate explicitly here as a
        # safety net (e.g. if the same track somehow appears via both BM25
        # and example-based dense retrieval paths before merging).
        seen: set = set()
        predicted_ids: List[str] = []
        for st in ranked:
            if st.track_id not in seen:
                seen.add(st.track_id)
                predicted_ids.append(st.track_id)

        # ── Generate natural language response ────────────────────────────────
        # Use response_history (clean turns only) so the prompt stays uncluttered.
        if generate_response and ranked:
            response_text = _generate_response(
                response_history, ranked,
                api_key=pipeline._api_key,
            )
        else:
            response_text = "Here are some songs you might enjoy."

        predictions.append({
            "session_id":           session_id,
            "user_id":              user_id,
            "turn_number":          turn_number,
            "predicted_track_ids":  predicted_ids,
            "predicted_response":   response_text,
        })

        if verbose:
            detail = (f"  → {len(predicted_ids)} tracks  "
                      f"response={response_text[:55]!r}...")
            if _tqdm_available:
                tqdm.write(detail)
            else:
                print(detail)

    # ── Write output ──────────────────────────────────────────────────────────
    tmp = output_path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    tmp.replace(output_path)

    print(f"\n[DONE] {len(predictions)} predictions saved to {output_path}")
    return predictions

def main():
    parser = argparse.ArgumentParser(
        description="TalkPlay Music Rec Baseline"
    )
    parser.add_argument(
        "--data-root", type=Path, default=DEFAULT_DATA_ROOT,
        help="Root directory containing the five data folders",
    )
    parser.add_argument(
        "--cache-file", type=Path, default=DEFAULT_CACHE_FILE,
        help="Path to intent JSON cache file",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run offline smoke-test with synthetic data (no data files needed)",
    )
    parser.add_argument(
        "--eval", action="store_true",
        help="Run evaluation on test split",
    )
    parser.add_argument(
        "--eval-k", type=int, default=10,
        help="k for hit-rate evaluation",
    )
    parser.add_argument(
        "--max-sessions", type=int, default=100,
        help="Max sessions to evaluate",
    )
    parser.add_argument(
        "--query", type=str, default=None,
        help="Ad-hoc query string for a quick test (requires --user-id)",
    )
    parser.add_argument(
        "--user-id", type=str, default=None,
        help="User UUID for --query",
    )
    parser.add_argument(
        "--predict-blind", action="store_true",
        help="Generate predictions for Blind-A dataset → predictions/blind_a_predictions.json",
    )
    parser.add_argument(
        "--predict-top-k", type=int, default=20,
        help="Max tracks per prediction (default: 20)",
    )
    parser.add_argument(
        "--predict-output", type=Path,
        default=Path("./predictions/blind_a_predictions.json"),
        help="Output path for blind predictions JSON",
    )
    parser.add_argument(
        "--no-response", action="store_true",
        help="Skip DeepSeek response generation (use placeholder text instead)",
    )
    args = parser.parse_args()

    if args.demo:
        _demo_synthetic()
        return

    # Real data
    pipeline = MusicRecPipeline.from_local(
        data_root=args.data_root,
        cache_file=args.cache_file,
        deepseek_api_key=DEEPSEEK_API_KEY,
    )

    if args.query:
        uid = args.user_id or "unknown-user"
        conv = [{"role": "user", "content": args.query}]
        ranked, _ = pipeline.recommend(uid, conv)
        print(pipeline.format_results(ranked))

    if args.eval:
        dfs = load_all_data(args.data_root)
        if "sessions" not in dfs:
            print("[ERROR] sessions data not found; cannot evaluate.")
            return
        # Use test split if a 'split' column exists
        sess_df = dfs["sessions"]
        if "split" in sess_df.columns:
            test_df = sess_df[sess_df["split"].str.startswith("test")]
        else:
            # Assume the dataset has test rows based on user_split in user_profile
            test_df = sess_df[
                sess_df["user_profile"].apply(
                    lambda x: "test" in str(x.get("user_split", "")) if isinstance(x, dict) else False
                )
            ]
        print(f"Evaluating on {len(test_df)} test sessions …")
        metrics = evaluate(pipeline, test_df, k=args.eval_k,
                           max_sessions=args.max_sessions)
        print(json.dumps(metrics, indent=2))

    if args.predict_blind:
        dfs = load_all_data(args.data_root)
        if "blind" not in dfs:
            print("[ERROR] Challenge-Blind-A data not found under", args.data_root)
            return
        blind_df = dfs["blind"]
        print(f"Generating predictions for {len(blind_df)} blind sessions …")
        predict_blind_a(
            pipeline=pipeline,
            blind_df=blind_df,
            output_path=args.predict_output,
            top_k=args.predict_top_k,
            generate_response=not args.no_response,
            verbose=True,
        )


if __name__ == "__main__":
    main()