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

TOP_K_RETRIEVE = 200
TOP_K_FINAL    = 10
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

    def apply(self, intent: ParsedIntent) -> pd.DataFrame:
        df = self.df

        if intent.year_min is not None:
            df = df[df["release_year"].notna() & (df["release_year"] >= intent.year_min)]
        if intent.year_max is not None:
            df = df[df["release_year"].notna() & (df["release_year"] <= intent.year_max)]
        if intent.popularity_min is not None:
            df = df[df["popularity"].notna() & (df["popularity"] >= intent.popularity_min)]
        if intent.popularity_max is not None:
            df = df[df["popularity"].notna() & (df["popularity"] <= intent.popularity_max)]
        if intent.duration_min_ms is not None:
            df = df[df["duration"].notna() & (df["duration"] >= intent.duration_min_ms)]
        if intent.duration_max_ms is not None:
            df = df[df["duration"].notna() & (df["duration"] <= intent.duration_max_ms)]

        return df


# ─────────────────────────────────────────────────────────────────────────────
# 6. BM25 Retriever
# ─────────────────────────────────────────────────────────────────────────────

class BM25Retriever:
    """BM25Okapi over concatenated text fields + decade tokens."""

    def __init__(self, track_meta: pd.DataFrame):
        from rank_bm25 import BM25Okapi  # type: ignore

        self._ids = track_meta["track_id"].tolist()
        corpus = [self._doc(row) for _, row in track_meta.iterrows()]
        self._bm25 = BM25Okapi([d.lower().split() for d in corpus])

    @staticmethod
    def _safe_join(v: Any) -> str:
        # parquet list columns may come back as list, np.ndarray, or other iterables
        if v is None:
            return ""
        if isinstance(v, (list, np.ndarray)):
            return " ".join(str(x) for x in v if x is not None and str(x).strip())
        # scalar (e.g. a plain string that wasn't stored as a list)
        s = str(v).strip()
        return s if s not in ("nan", "None", "") else ""

    def _doc(self, row: pd.Series) -> str:
        parts = [
            self._safe_join(row.get("track_name")),
            self._safe_join(row.get("artist_name")),
            self._safe_join(row.get("album_name")),
            self._safe_join(row.get("tag_list")),
        ]
        # Embed decade + individual years so "2000s" matches
        rd = str(row.get("release_date") or "")
        m = re.match(r"(\d{4})", rd)
        if m:
            yr = int(m.group(1))
            decade = (yr // 10) * 10
            parts.append(f"{decade}s {yr}")
        return " ".join(filter(None, parts))

    def retrieve(
        self,
        intent: ParsedIntent,
        candidate_ids: Optional[List[str]] = None,
        top_k: int = TOP_K_RETRIEVE,
    ) -> Dict[str, float]:
        tokens = (
            intent.artist_names + intent.track_names + intent.album_names
            + intent.genres + intent.mood + intent.themes
        )
        if intent.semantic_query:
            tokens += intent.semantic_query.split()
        if not tokens:
            return {}

        scores = self._bm25.get_scores([t.lower() for t in tokens])
        id_score = dict(zip(self._ids, map(float, scores)))

        if candidate_ids is not None:
            cset = set(candidate_ids)
            id_score = {k: v for k, v in id_score.items() if k in cset}

        return dict(sorted(id_score.items(), key=lambda x: x[1], reverse=True)[:top_k])


# ─────────────────────────────────────────────────────────────────────────────
# 7. Dense Retriever  (FAISS)
# ─────────────────────────────────────────────────────────────────────────────

class DenseRetriever:
    """
    FAISS inner-product search over L2-normalised track attribute embeddings
    (attributes-qwen3_embedding_0.6b, 1024d).
    """

    EMB_COL = "attributes-qwen3_embedding_0.6b"
    EMB_DIM = 1024

    def __init__(self, track_embeddings: pd.DataFrame):
        import faiss  # type: ignore

        # Drop rows with empty embeddings
        mask = track_embeddings[self.EMB_COL].apply(
            lambda x: isinstance(x, (list, np.ndarray)) and len(x) > 0
        )
        emb_df = track_embeddings[mask].reset_index(drop=True)

        self._ids: List[str] = emb_df["track_id"].tolist()
        self._id2idx: Dict[str, int] = {tid: i for i, tid in enumerate(self._ids)}

        vecs = np.array(emb_df[self.EMB_COL].tolist(), dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        self._vecs = (vecs / norms).astype(np.float32)

        self._index = faiss.IndexFlatIP(self._vecs.shape[1])
        self._index.add(self._vecs)

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

    def _encode(self, text: str) -> np.ndarray:
        enc = self._get_encoder()
        v = enc.encode([text], normalize_embeddings=True)[0].astype(np.float32)
        # Pad / truncate to EMB_DIM
        if v.shape[0] < self.EMB_DIM:
            v = np.pad(v, (0, self.EMB_DIM - v.shape[0]))
        else:
            v = v[: self.EMB_DIM]
        v /= np.linalg.norm(v) + 1e-9
        return v

    # ── Retrieval ────────────────────────────────────────────────────────────

    def _score_candidates(
        self, qv: np.ndarray, candidate_ids: Optional[List[str]], top_k: int
    ) -> Dict[str, float]:
        qv = qv.reshape(1, -1)
        if candidate_ids is None:
            scores, idxs = self._index.search(qv, top_k)
            return {
                self._ids[i]: float(s)
                for s, i in zip(scores[0], idxs[0])
                if i >= 0
            }
        results = {
            tid: float(np.dot(qv[0], self._vecs[self._id2idx[tid]]))
            for tid in candidate_ids
            if tid in self._id2idx
        }
        return dict(
            sorted(results.items(), key=lambda x: x[1], reverse=True)[:top_k]
        )

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
        def centroid(ids):
            vecs = [self._vecs[self._id2idx[i]] for i in ids if i in self._id2idx]
            if not vecs:
                return None
            c = np.mean(vecs, axis=0).astype(np.float32)
            c /= np.linalg.norm(c) + 1e-9
            return c

        pos_v = centroid(pos_ids)
        if pos_v is None:
            return {}
        neg_v = centroid(neg_ids)
        qv = pos_v - 0.3 * neg_v if neg_v is not None else pos_v
        qv /= np.linalg.norm(qv) + 1e-9
        return self._score_candidates(qv, candidate_ids, top_k)


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
    ) -> List[ScoredTrack]:
        if not rrf_scores:
            return []

        user_v  = self._user_cf.get(user_id)
        max_rrf = max(rrf_scores.values()) or 1.0

        results: List[ScoredTrack] = []
        for tid, rrf_s in rrf_scores.items():
            norm_rrf = rrf_s / max_rrf
            cf_s = 0.0
            if user_v is not None and tid in self._track_cf:
                cf_s = float(np.dot(user_v, self._track_cf[tid]))
            results.append(
                ScoredTrack(
                    track_id=tid,
                    rrf_score=rrf_s,
                    cf_score=cf_s,
                    final_score=self.alpha * norm_rrf + self.beta * cf_s,
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
    ) -> Tuple[List[ScoredTrack], ParsedIntent]:
        t0 = time.time()

        # 1. Intent
        intent, from_cache = parse_intent(conversation, self.cache, self._api_key)
        src = "cache" if from_cache else "API"
        if verbose:
            print(f"[1] Intent ({src})  abstract={intent.is_abstract}  "
                  f"decade={intent.decade}  [{intent.year_min},{intent.year_max}]  "
                  f"genres={intent.genres}  mood={intent.mood}")

        # 2. Filter
        filtered = self.filter.apply(intent)
        cids = filtered["track_id"].tolist() if len(filtered) else None
        if verbose:
            n_total = len(self.filter.df)
            n_pass  = len(filtered) if cids else n_total
            print(f"[2] Filter  {n_pass:,}/{n_total:,} tracks"
                  + (" (no filter active)" if cids is None else ""))

        # 3. BM25
        bm25_scores = self.bm25.retrieve(intent, candidate_ids=cids,
                                          top_k=TOP_K_RETRIEVE)
        if verbose:
            print(f"[3] BM25    {len(bm25_scores)} candidates")

        # 4. Dense
        dense_scores: Dict[str, float] = {}
        if intent.semantic_query:
            dense_scores = self.dense.retrieve(intent, candidate_ids=cids,
                                               top_k=TOP_K_RETRIEVE)
        if intent.positive_track_ids:
            ex_scores = self.dense.retrieve_by_example(
                intent.positive_track_ids, intent.negative_track_ids,
                candidate_ids=cids, top_k=TOP_K_RETRIEVE,
            )
            for tid, s in ex_scores.items():
                dense_scores[tid] = max(dense_scores.get(tid, 0.0), s)
        if verbose:
            print(f"[4] Dense   {len(dense_scores)} candidates")

        # 5. RRF (weights adapt to abstraction level)
        bw = 0.2 if intent.is_abstract else BM25_WEIGHT
        dw = 0.8 if intent.is_abstract else DENSE_WEIGHT
        rrf_scores = rrf_fusion([bm25_scores, dense_scores], [bw, dw])
        if verbose:
            print(f"[5] RRF     {len(rrf_scores)} candidates  "
                  f"(bm25:{bw:.1f} dense:{dw:.1f})")

        # 6. CF-BPR rerank
        ranked = self.ranker.rank(user_id, rrf_scores, top_k=top_k)
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

    cache_file = Path("./cache/intent_demo.jsonl")
    pipeline = MusicRecPipeline(tracks, track_emb, user_emb,
                                cache_file=cache_file)

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
You are a music recommendation assistant.
Given a conversation history and a list of recommended track titles/artists,
write a SHORT, natural, friendly response (2-3 sentences) that:
1. Directly addresses what the user asked for.
2. Briefly explains WHY these tracks were chosen (mood, genre, era, style).
3. Does NOT list the tracks again – just describe them as a group.
Keep it under 60 words.
"""


def _generate_response(
    conversation: List[Dict[str, str]],
    ranked: List[ScoredTrack],
    api_key: str = DEEPSEEK_API_KEY,
) -> str:
    """
    Ask DeepSeek to produce a natural-language response for the recommendation.
    Falls back to a template string if the API is unavailable.
    """
    # Build a short track summary for the prompt
    track_summaries = []
    for st in ranked[:5]:        # only describe the top-5 to keep prompt short
        m = st.metadata
        name   = m.get("track_name", st.track_id)
        artist = m.get("artist_name", "")
        track_summaries.append(f"{name} by {artist}" if artist else name)
    tracks_str = "; ".join(track_summaries) if track_summaries else "various tracks"

    # Last user message as context
    last_user = next(
        (m["content"] for m in reversed(conversation) if m["role"] == "user"), ""
    )

    prompt = (
        f"User request: \"{last_user}\"\n"
        f"Top recommended tracks: {tracks_str}\n"
        "Write a recommendation response."
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
                "max_tokens": 120,
                "temperature": 0.7,
            },
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        # Graceful fallback
        genre_hint = ""
        if ranked and ranked[0].metadata.get("tags"):
            genre_hint = f" ({', '.join(ranked[0].metadata['tags'][:2])})"
        return (
            f"Here are some tracks that match your request{genre_hint}. "
            f"I picked these based on your musical taste and what you described. "
            f"Hope you enjoy them!"
        )


def _build_conversation_history(turns: List[Dict]) -> List[Dict[str, str]]:
    """
    Convert blind-A conversation turns (which include role/content/thought/turn_number)
    into the simple {role, content} format expected by the pipeline.
    Skips 'music' turns (those contain track_ids, not natural language).
    """
    history = []
    for t in turns:
        role    = t.get("role", "")
        content = t.get("content", "") or ""
        if role == "user":
            history.append({"role": "user", "content": content})
        elif role in ("assistant", "system") and content:
            history.append({"role": "assistant", "content": content})
        # 'music' role turns contain a track_id string – skip for intent parsing
    return history


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

        # ── Build conversation context for intent parsing ──────────────────
        history = _build_conversation_history(turns)

        if not history:
            # No user turn at all – skip gracefully
            msg = f"[SKIP] No user turns in session {session_id}"
            if _tqdm_available:
                tqdm.write(msg)
            else:
                print(msg)
            continue

        # Inject previously played tracks as positive examples
        played_ids = _extract_positive_ids_from_history(turns)
        if played_ids and history:
            hint = f"[Previously played: {', '.join(played_ids[:5])}]"
            history = history + [{"role": "assistant", "content": hint}]

        # ── Recommend ─────────────────────────────────────────────────────────
        try:
            ranked, intent = pipeline.recommend(
                user_id, history,
                top_k=top_k,
                verbose=False,
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
        if generate_response and ranked:
            response_text = _generate_response(
                history, ranked,
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