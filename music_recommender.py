"""
会话式音乐推荐系统 v2
架构：DeepSeek intent → SQL filter → BM25+Dense(4空间) RRF → CF-BPR rerank
新增：UserHistoryIndex / ColdUserHandler / 多空间 Dense / CF-BPR 归一化融合
"""

import os, json, time, re
import numpy as np
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from openai import OpenAI
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# 常量
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_DATA_ROOT = Path("./data")
TOP_K_RETRIEVE    = 500
TOP_K_FINAL       = 20
RRF_K             = 60

EMB_COLS = [
    "attributes-qwen3_embedding_0.6b",  # 1024d  weight=0.35
    "metadata-qwen3_embedding_0.6b",    # 1024d  weight=0.20
    "lyrics-qwen3_embedding_0.6b",      # 1024d  weight=0.15
    "audio-laion_clap",                 # 512d   weight=0.30
]
EMB_WEIGHTS = [0.35, 0.20, 0.15, 0.30]
ENCODER_PATH = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-Embedding-0___6B"
QUERY_INSTRUCTION = "Instruct: Given a user music request, retrieve the most relevant tracks\nQuery: "

# ─────────────────────────────────────────────────────────────────────────────
# 0. 数据加载
# ─────────────────────────────────────────────────────────────────────────────

def _load_local(data_dir: Path, split: Optional[str] = None) -> pd.DataFrame:
    try:
        from datasets import load_dataset
        data_str = str(data_dir.resolve())
        if split:
            ds = load_dataset("parquet", data_dir=data_str, split=split)
        else:
            ds_dict = load_dataset("parquet", data_dir=data_str)
            frames = [s.to_pandas() for s in ds_dict.values()]
            return pd.concat(frames, ignore_index=True)
        return ds.to_pandas()
    except Exception:
        parquet_files = sorted(data_dir.rglob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files under {data_dir}")
        frames = [pd.read_parquet(p) for p in parquet_files]
        df = pd.concat(frames, ignore_index=True)
        if split and "split" in df.columns:
            df = df[df["split"] == split]
        return df


def load_all_data(data_root: Path = DEFAULT_DATA_ROOT) -> dict:
    root = Path(data_root)
    result = {}

    dirs = {
        "track_meta":  root / "Track-Metadata",
        "track_emb":   root / "Track-Embedding",
        "user_emb":    root / "User-Embedding",
        "user_meta":   root / "User-Metadata",
        "sessions":    root / "Challenge-Data",
        "blind_a":     root / "Challenge-Blind-A",
    }
    for key, d in dirs.items():
        if d.exists():
            t0 = time.time()
            print(f"[LOAD] {key:12s} ← {d} …", end=" ", flush=True)
            result[key] = _load_local(d)
            print(f"{len(result[key]):,} rows  ({time.time()-t0:.1f}s)")
        else:
            print(f"[WARN] {key:12s} not found: {d}")
            result[key] = pd.DataFrame()

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 1. 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def _to_list(v) -> list:
    if v is None: return []
    if isinstance(v, np.ndarray): return v.tolist()
    if isinstance(v, list): return v
    try:
        import pandas as _pd
        if _pd.isna(v): return []
    except (TypeError, ValueError): pass
    return []

def _first(v, default="") -> str:
    lst = _to_list(v)
    return str(lst[0]) if lst else default

def _join(v) -> str:
    lst = _to_list(v)
    return ",".join(str(x) for x in lst if x)


# ─────────────────────────────────────────────────────────────────────────────
# 2. UserHistoryIndex
# ─────────────────────────────────────────────────────────────────────────────

class UserHistoryIndex:
    """
    从训练 sessions 构建每个 user 的历史画像：
      - accepted_track_ids  (MOVES_TOWARD_GOAL)
      - rejected_track_ids  (DOES_NOT_MOVE)
      - accepted_artists / accepted_tags (Counter)
    """
    UUID_RE = re.compile(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
    )

    def __init__(self, sessions_df: pd.DataFrame, track_meta: pd.DataFrame):
        self._track_artists: Dict[str, List[str]] = {}
        self._track_tags:    Dict[str, List[str]] = {}
        for _, row in track_meta.iterrows():
            tid = str(row.get("track_id", "") or "")
            if not tid: continue
            self._track_artists[tid] = [str(a) for a in _to_list(row.get("artist_name")) if a]
            self._track_tags[tid]    = [str(t) for t in _to_list(row.get("tag_list")) if t]

        self._profiles: Dict[str, Dict] = defaultdict(lambda: {
            "accepted_track_ids": [],
            "rejected_track_ids": [],
            "accepted_artists":   Counter(),
            "accepted_tags":      Counter(),
        })

        n_users = n_accepted = 0
        for _, row in sessions_df.iterrows():
            user_id = str(row.get("user_id", "") or "")
            turns   = _to_list(row.get("conversations"))
            assessments = _to_list(row.get("goal_progress_assessments"))
            if not user_id: continue

            assess_map: Dict[int, str] = {}
            for a in assessments:
                if not isinstance(a, dict): continue
                tn  = a.get("turn_number")
                gpa = str(a.get("goal_progress_assessment") or "")
                if tn is not None:
                    try: assess_map[int(tn)] = gpa
                    except (TypeError, ValueError): pass

            profile = self._profiles[user_id]
            is_new  = len(profile["accepted_track_ids"]) == 0

            for t in turns:
                if not isinstance(t, dict): continue
                role    = str(t.get("role", "") or "")
                content = str(t.get("content", "") or "").strip()
                turn_no = t.get("turn_number")
                if role != "music" or not self.UUID_RE.match(content): continue
                gpa = assess_map.get(int(turn_no) if turn_no is not None else -1, "")
                if gpa == "MOVES_TOWARD_GOAL":
                    profile["accepted_track_ids"].append(content)
                    for a in self._track_artists.get(content, []):
                        profile["accepted_artists"][a] += 1
                    for tg in self._track_tags.get(content, [])[:5]:
                        profile["accepted_tags"][tg] += 1
                    n_accepted += 1
                elif gpa == "DOES_NOT_MOVE_TOWARD_GOAL":
                    profile["rejected_track_ids"].append(content)

            if is_new and len(profile["accepted_track_ids"]) > 0:
                n_users += 1

        # deduplicate
        for p in self._profiles.values():
            for key in ("accepted_track_ids", "rejected_track_ids"):
                seen, deduped = set(), []
                for t in p[key]:
                    if t not in seen: seen.add(t); deduped.append(t)
                p[key] = deduped

        print(f"[UserHistory] {n_users} users  {n_accepted} accepted interactions")

    def has_history(self, user_id: str) -> bool:
        return len(self._profiles.get(user_id, {}).get("accepted_track_ids", [])) > 0

    def get_accepted_tracks(self, user_id: str, max_recent: int = 30) -> List[str]:
        tracks = self._profiles.get(user_id, {}).get("accepted_track_ids", [])
        return tracks[-max_recent:]

    def get_accepted_tracks_weighted(
        self, user_id: str, max_recent: int = 30, lam: float = 0.15
    ) -> List[Tuple[str, float]]:
        tracks = self._profiles.get(user_id, {}).get("accepted_track_ids", [])[-max_recent:]
        n = len(tracks)
        return [(tid, float(np.exp(-lam * (n - 1 - i)))) for i, tid in enumerate(tracks)]

    def get_rejected_tracks(self, user_id: str) -> set:
        return set(self._profiles.get(user_id, {}).get("rejected_track_ids", []))

    def get_top_artists(self, user_id: str, top_n: int = 5) -> List[str]:
        counter = self._profiles.get(user_id, {}).get("accepted_artists", {})
        return [a for a, _ in Counter(counter).most_common(top_n)]

    def get_top_tags(self, user_id: str, top_n: int = 8) -> List[str]:
        counter = self._profiles.get(user_id, {}).get("accepted_tags", {})
        return [t for t, _ in Counter(counter).most_common(top_n)]

    def enrich(self, user_id: str, pos_ids: List[str], neg_ids: List[str],
               dense_query: str, artist_names: List[str], genres: List[str]
               ) -> Tuple[List[str], List[str], List[Tuple[str,float]], str, List[str], List[str]]:
        """
        返回 (enriched_pos_ids, enriched_neg_ids, weighted_pos, enriched_query,
               enriched_artists, enriched_genres)
        """
        if not self.has_history(user_id):
            return pos_ids, neg_ids, [(t,1.0) for t in pos_ids], dense_query, artist_names, genres

        weighted = self.get_accepted_tracks_weighted(user_id, max_recent=30)
        hist_tracks = [tid for tid, _ in weighted]
        combined_pos = list(dict.fromkeys(hist_tracks + pos_ids))

        session_weighted = [(tid, 1.0) for tid in pos_ids if tid not in {t for t,_ in weighted}]
        all_weighted = weighted + session_weighted

        rejected = self.get_rejected_tracks(user_id)
        combined_neg = list(set(neg_ids) | rejected)

        # query expansion
        enriched_query = dense_query
        enriched_artists = artist_names
        enriched_genres  = genres

        if not artist_names:
            top_artists = self.get_top_artists(user_id, top_n=3)
            if top_artists: enriched_artists = top_artists

        if not genres:
            top_tags = self.get_top_tags(user_id, top_n=5)
            if top_tags: enriched_genres = top_tags

        return combined_pos, combined_neg, all_weighted, enriched_query, enriched_artists, enriched_genres


# ─────────────────────────────────────────────────────────────────────────────
# 3. ColdUserHandler
# ─────────────────────────────────────────────────────────────────────────────

class ColdUserHandler:
    def __init__(self, user_meta: pd.DataFrame, user_emb: pd.DataFrame):
        self._user_culture: Dict[str, str] = {}
        for _, row in user_meta.iterrows():
            uid = str(row.get("user_id", "") or "")
            if uid:
                self._user_culture[uid] = str(row.get("preferred_musical_culture") or "").strip()

        user_cf_map: Dict[str, np.ndarray] = {}
        for _, row in user_emb.iterrows():
            v = row.get("cf-bpr")
            if isinstance(v, (list, np.ndarray)) and len(v) > 0:
                user_cf_map[str(row["user_id"])] = np.asarray(v, dtype=np.float32)

        culture_vecs: Dict[str, List[np.ndarray]] = {}
        for uid, culture in self._user_culture.items():
            if uid in user_cf_map and culture:
                culture_vecs.setdefault(culture, []).append(user_cf_map[uid])

        self._culture_centroids: Dict[str, np.ndarray] = {}
        for culture, vecs in culture_vecs.items():
            c = np.mean(vecs, axis=0).astype(np.float32)
            c /= np.linalg.norm(c) + 1e-9
            self._culture_centroids[culture] = c

        all_vecs = list(user_cf_map.values())
        if all_vecs:
            g = np.mean(all_vecs, axis=0).astype(np.float32)
            g /= np.linalg.norm(g) + 1e-9
            self._global_centroid: Optional[np.ndarray] = g
        else:
            self._global_centroid = None

        print(f"[ColdUser] {len(user_cf_map)} warm users  {len(self._culture_centroids)} cultures")

    def get_proxy_vector(self, user_id: str, override_culture: str = "") -> Optional[np.ndarray]:
        culture = override_culture or self._user_culture.get(user_id, "")
        if culture and culture in self._culture_centroids:
            return self._culture_centroids[culture]
        if culture:
            words = set(culture.lower().split())
            best_score, best_vec = 0, None
            for c_key, c_vec in self._culture_centroids.items():
                overlap = len(words & set(c_key.lower().split()))
                if overlap > best_score:
                    best_score, best_vec = overlap, c_vec
            if best_vec is not None and best_score >= 1:
                return best_vec
        return self._global_centroid


# ─────────────────────────────────────────────────────────────────────────────
# 4. 多空间 DenseIndex
# ─────────────────────────────────────────────────────────────────────────────

class DenseIndex:
    """
    4个 embedding 空间各自建索引，检索时用加权 RRF 融合。
    同时支持 query-text 检索 和 example-centroid 检索。
    """

    def __init__(self, track_emb_df: pd.DataFrame):
        self._id2idx: Dict[str, int] = {}
        self._ids:    List[str] = []
        self._vecs:   List[Optional[np.ndarray]] = []  # per EMB_COL, shape (N, dim)

        ids_col = track_emb_df["track_id"].astype(str).tolist()
        for i, tid in enumerate(ids_col):
            self._id2idx[tid] = i
        self._ids = ids_col

        for col in EMB_COLS:
            if col not in track_emb_df.columns:
                self._vecs.append(None)
                print(f"[DenseIndex] col {col} not found, skipped")
                continue
            rows = []
            for v in track_emb_df[col]:
                if isinstance(v, (list, np.ndarray)) and len(v) > 0:
                    rows.append(np.asarray(v, dtype=np.float32))
                else:
                    rows.append(None)

            # find dim
            dims = [r.shape[0] for r in rows if r is not None]
            if not dims:
                self._vecs.append(None); continue
            dim = max(set(dims), key=dims.count)

            mat = np.zeros((len(rows), dim), dtype=np.float32)
            for i, r in enumerate(rows):
                if r is not None:
                    r = r[:dim] if r.shape[0] >= dim else np.pad(r, (0, dim-r.shape[0]))
                    n = np.linalg.norm(r)
                    mat[i] = r / (n + 1e-9)
            self._vecs.append(mat)
            print(f"[DenseIndex] {col}: {mat.shape}")

        # encoder (lazy)
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer(ENCODER_PATH, local_files_only=True)
                print("[Encoder] Loaded Qwen3-Embedding")
            except Exception as e:
                print(f"[Encoder] Fallback hash encoder ({e})")
                class _Hash:
                    def encode(self_, texts, normalize_embeddings=True):
                        out = []
                        for t in texts:
                            rng = np.random.default_rng(abs(hash(t)) % (2**31))
                            v = rng.normal(size=1024).astype(np.float32)
                            if normalize_embeddings: v /= np.linalg.norm(v)+1e-9
                            out.append(v)
                        return np.array(out)
                self._encoder = _Hash()
        return self._encoder

    def encode(self, text: str, dim: int = 1024) -> np.ndarray:
        enc = self._get_encoder()
        v = enc.encode([QUERY_INSTRUCTION + text], normalize_embeddings=True)[0].astype(np.float32)
        if v.shape[0] < dim: v = np.pad(v, (0, dim - v.shape[0]))
        else: v = v[:dim]
        v /= np.linalg.norm(v) + 1e-9
        return v

    def _score_one(self, col_idx: int, qv: np.ndarray,
                   candidate_ids: Optional[List[str]] = None,
                   top_k: int = TOP_K_RETRIEVE) -> Dict[str, float]:
        mat = self._vecs[col_idx]
        if mat is None: return {}
        dim = mat.shape[1]
        qv  = qv[:dim] if qv.shape[0] >= dim else np.pad(qv, (0, dim - qv.shape[0]))
        qv  = qv / (np.linalg.norm(qv) + 1e-9)

        if candidate_ids is not None:
            idxs = [self._id2idx[t] for t in candidate_ids if t in self._id2idx]
            sims = mat[idxs] @ qv
            order = np.argsort(sims)[::-1][:top_k]
            return {candidate_ids[i]: float(sims[i]) for i in order}
        else:
            sims = mat @ qv
            order = np.argsort(sims)[::-1][:top_k]
            return {self._ids[i]: float(sims[i]) for i in order}

    def _rrf_fuse(self, score_dicts: List[Dict[str,float]],
                  weights: List[float]) -> Dict[str, float]:
        fused: Dict[str, float] = {}
        for sd, w in zip(score_dicts, weights):
            for rank, tid in enumerate(sorted(sd, key=sd.get, reverse=True)):
                fused[tid] = fused.get(tid, 0.0) + w / (RRF_K + rank + 1)
        return fused

    def retrieve_by_query(self, query: str,
                          candidate_ids: Optional[List[str]] = None,
                          top_k: int = TOP_K_RETRIEVE) -> Dict[str, float]:
        """文本 query → 4空间各检索 → 加权RRF"""
        all_scores, all_w = [], []
        for i, (col, w) in enumerate(zip(EMB_COLS, EMB_WEIGHTS)):
            if self._vecs[i] is None: continue
            dim = self._vecs[i].shape[1]
            qv  = self.encode(query, dim=dim)
            sd  = self._score_one(i, qv, candidate_ids, top_k)
            if sd: all_scores.append(sd); all_w.append(w)
        if not all_scores: return {}
        return dict(sorted(self._rrf_fuse(all_scores, all_w).items(),
                           key=lambda x: x[1], reverse=True)[:top_k])

    def retrieve_by_example(self, pos_ids: List[str], neg_ids: List[str],
                             weighted_pos: Optional[List[Tuple[str,float]]] = None,
                             candidate_ids: Optional[List[str]] = None,
                             top_k: int = TOP_K_RETRIEVE) -> Dict[str, float]:
        """已播放 track 的 centroid → 4空间各检索 → 加权RRF"""
        if not pos_ids: return {}
        w_map = {tid: wt for tid, wt in weighted_pos} if weighted_pos else None

        all_scores, all_w = [], []
        for i, (col, ew) in enumerate(zip(EMB_COLS, EMB_WEIGHTS)):
            mat = self._vecs[i]
            if mat is None: continue

            # positive centroid (weighted)
            pos_rows, pos_ws = [], []
            for tid in pos_ids:
                if tid not in self._id2idx: continue
                pos_rows.append(mat[self._id2idx[tid]])
                pos_ws.append(w_map[tid] if w_map and tid in w_map else 1.0)
            if not pos_rows: continue
            ws = np.array(pos_ws, dtype=np.float32); ws /= ws.sum()+1e-9
            pos_v = (np.stack(pos_rows).T @ ws).astype(np.float32)

            # negative centroid
            neg_rows = [mat[self._id2idx[t]] for t in neg_ids if t in self._id2idx]
            if neg_rows:
                neg_v = np.mean(neg_rows, axis=0).astype(np.float32)
                qv = pos_v - 0.3 * neg_v
            else:
                qv = pos_v
            qv /= np.linalg.norm(qv) + 1e-9

            sd = self._score_one(i, qv, candidate_ids, top_k)
            if sd: all_scores.append(sd); all_w.append(ew)

        if not all_scores: return {}
        return dict(sorted(self._rrf_fuse(all_scores, all_w).items(),
                           key=lambda x: x[1], reverse=True)[:top_k])


# ─────────────────────────────────────────────────────────────────────────────
# 5. NegationState
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class NegationState:
    hard_tracks:   set  = field(default_factory=set)
    hard_tags:     set  = field(default_factory=set)
    hard_artists:  set  = field(default_factory=set)
    exclude_embeddings: list = field(default_factory=list)
    embedding_threshold: float = 0.85
    exceptions:    set  = field(default_factory=set)

    def update(self, negation: dict, embeddings: dict = None):
        for t in negation.get("tags", []):    self.hard_tags.add(t.lower())
        for a in negation.get("artists", []): self.hard_artists.add(a.lower())
        for tid in negation.get("track_ids", []):
            self.hard_tracks.add(tid)
            if embeddings and tid in embeddings:
                self.exclude_embeddings.append(embeddings[tid])
        for e in negation.get("exceptions", []): self.exceptions.add(e)

    def reset(self):
        self.__init__()


# ─────────────────────────────────────────────────────────────────────────────
# 6. MusicTools（SQLite + BM25 + DenseIndex + CF-BPR）
# ─────────────────────────────────────────────────────────────────────────────

class MusicTools:
    def __init__(self, tracks_df: pd.DataFrame, track_emb_df: pd.DataFrame,
                 user_emb_df: pd.DataFrame, user_meta_df: pd.DataFrame,
                 sessions_df: pd.DataFrame):

        self.tracks = tracks_df
        # primary embeddings dict (attributes col) for negation / padding
        self.embeddings: Dict[str, np.ndarray] = {}
        primary_col = EMB_COLS[0]
        if primary_col in track_emb_df.columns:
            for _, row in track_emb_df.iterrows():
                v = row.get(primary_col)
                if isinstance(v, (list, np.ndarray)) and len(v) > 0:
                    self.embeddings[str(row["track_id"])] = np.asarray(v, dtype=np.float32)

        # SQLite
        self.conn = self._build_db()

        # BM25 (全量, 启动时建索引)
        self._bm25_tag    = None
        self._bm25_artist = None
        self._bm25_track  = None
        self._bm25_full   = None
        self._bm25_ids: List[str] = []
        self._build_bm25()

        # Multi-space dense
        self.dense = DenseIndex(track_emb_df)

        # CF-BPR
        self.track_cfbpr: Dict[str, np.ndarray] = {}
        if "cf-bpr" in track_emb_df.columns:
            for _, row in track_emb_df.iterrows():
                v = row.get("cf-bpr")
                if isinstance(v, (list, np.ndarray)) and len(v) > 0:
                    self.track_cfbpr[str(row["track_id"])] = np.asarray(v, dtype=np.float32)
            print(f"[CF-BPR] {len(self.track_cfbpr):,} track vectors")

        self.user_cfbpr: Dict[str, np.ndarray] = {}
        if not user_emb_df.empty and "cf-bpr" in user_emb_df.columns:
            for _, row in user_emb_df.iterrows():
                v = row.get("cf-bpr")
                if isinstance(v, (list, np.ndarray)) and len(v) > 0:
                    self.user_cfbpr[str(row["user_id"])] = np.asarray(v, dtype=np.float32)

        # ColdUserHandler
        self.cold_handler: Optional[ColdUserHandler] = None
        if not user_meta_df.empty and not user_emb_df.empty:
            self.cold_handler = ColdUserHandler(user_meta_df, user_emb_df)

        # UserHistoryIndex
        self.user_history: Optional[UserHistoryIndex] = None
        if not sessions_df.empty:
            self.user_history = UserHistoryIndex(sessions_df, tracks_df)

        # Conditional Dual Encoder（可选，ckpt 存在时自动加载）
        self.cde_retriever = None
        cde_ckpt = Path("./checkpoints/cde/best_cde.pt")
        if cde_ckpt.exists():
            try:
                from conditional_dual_encoder import CDERetriever
                self.cde_retriever = CDERetriever(
                    ckpt_path     = str(cde_ckpt),
                    track_meta_df = tracks_df,
                )
                print("[CDE] CDERetriever loaded")
            except Exception as e:
                print(f"[CDE] Failed to load: {e}")
        else:
            print(f"[CDE] No checkpoint found at {cde_ckpt}, CDE disabled")

    # ── SQLite ────────────────────────────────────────────────────────────────
    def _build_db(self):
        import sqlite3
        conn = sqlite3.connect(":memory:")
        conn.execute("""
            CREATE TABLE tracks (
                track_id     TEXT PRIMARY KEY,
                track_name   TEXT,
                artist_name  TEXT,
                album_name   TEXT,
                release_date TEXT,
                popularity   REAL
            )
        """)
        rows = []
        for _, r in self.tracks.iterrows():
            rows.append((
                str(r["track_id"]),
                _first(r["track_name"]),
                _first(r["artist_name"]),
                _first(r.get("album_name", "")),
                str(r.get("release_date", "")),
                float(r.get("popularity", 0) or 0),
            ))
        conn.executemany("INSERT OR IGNORE INTO tracks VALUES (?,?,?,?,?,?)", rows)
        conn.execute("CREATE INDEX idx_artist ON tracks(LOWER(artist_name))")
        conn.commit()
        print(f"[SQLite] {len(rows):,} tracks loaded")
        return conn

    # ── BM25 ──────────────────────────────────────────────────────────────────
    def _build_bm25(self):
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            print("[BM25] rank_bm25 not installed, BM25 disabled")
            return

        ids, tag_corpus, artist_corpus, track_corpus, full_corpus = [], [], [], [], []
        for _, r in self.tracks.iterrows():
            tid     = str(r["track_id"])
            tags    = [str(t).lower() for t in _to_list(r.get("tag_list"))]
            artists = [str(a).lower() for a in _to_list(r.get("artist_name"))]
            tnames  = [str(t).lower() for t in _to_list(r.get("track_name"))]
            album   = [str(a).lower() for a in _to_list(r.get("album_name"))]
            ids.append(tid)
            tag_corpus.append(tags)
            artist_corpus.append(artists)
            track_corpus.append(tnames)
            full_corpus.append(tags + artists + tnames + album)

        self._bm25_ids    = ids
        self._bm25_tag    = BM25Okapi(tag_corpus)
        self._bm25_artist = BM25Okapi(artist_corpus)
        self._bm25_track  = BM25Okapi(track_corpus)
        self._bm25_full   = BM25Okapi(full_corpus)
        print(f"[BM25] {len(ids):,} tracks indexed (tag/artist/track/full)")

    def bm25_retrieve(self, mode: str, artist_names: List[str], track_names: List[str],
                      album_names: List[str], genres: List[str], mood: List[str],
                      semantic_query: str, user_culture: str,
                      candidate_ids: Optional[List[str]] = None,
                      top_k: int = TOP_K_RETRIEVE) -> Dict[str, float]:
        if self._bm25_full is None: return {}

        cid_set = set(candidate_ids) if candidate_ids else None

        def _scores_to_dict(scores_arr, cids_filter=None) -> Dict[str, float]:
            result = {}
            for i, s in enumerate(scores_arr):
                if s <= 0: continue
                tid = self._bm25_ids[i]
                if cids_filter and tid not in cids_filter: continue
                result[tid] = float(s)
            return dict(sorted(result.items(), key=lambda x: x[1], reverse=True)[:top_k])

        if mode == "exact_track" and track_names:
            q = " ".join(track_names).lower().split()
            return _scores_to_dict(self._bm25_track.get_scores(q), cid_set)

        if mode == "exact_artist" and artist_names:
            artist_lower = [a.lower() for a in artist_names]
            artist_cids = [
                tid for tid, r in zip(self._bm25_ids,
                    [_first(row.get("artist_name")).lower()
                     for _, row in self.tracks.iterrows()])
                if any(al in r for al in artist_lower)
            ]
            if cid_set: artist_cids = [t for t in artist_cids if t in cid_set]
            if genres or mood:
                q = [t.lower() for t in genres + mood]
                return _scores_to_dict(self._bm25_tag.get_scores(q), set(artist_cids))
            q = " ".join(artist_names).lower().split()
            return _scores_to_dict(self._bm25_artist.get_scores(q), set(artist_cids))

        if mode in ("genre_mood", "abstract"):
            tokens = [t.lower() for t in genres + mood]
            if user_culture: tokens += user_culture.lower().split()
            if semantic_query:
                stop = {"a","the","and","or","in","of","to","for","with","is","it"}
                tokens += [w for w in semantic_query.lower().split() if w not in stop]
            if not tokens: return {}
            return _scores_to_dict(self._bm25_tag.get_scores(tokens), cid_set)

        # default
        tokens = [t.lower() for t in artist_names + track_names + album_names + genres + mood]
        if semantic_query: tokens += semantic_query.lower().split()
        if user_culture:   tokens += user_culture.lower().split()
        if not tokens: return {}
        return _scores_to_dict(self._bm25_full.get_scores(tokens), cid_set)

    # ── Info / Negation ───────────────────────────────────────────────────────
    def get_info(self, track_id: str) -> dict:
        r = self.tracks[self.tracks["track_id"] == track_id]
        if r.empty: return {}
        r = r.iloc[0]
        return {
            "track_id":    track_id,
            "track_name":  _first(r["track_name"]),
            "artist_name": _first(r["artist_name"]),
            "tags":        _to_list(r["tag_list"])[:8],
            "release_date": str(r.get("release_date", "")),
            "popularity":  float(r.get("popularity", 0) or 0),
        }

    def passes_negation(self, tid: str, ns: NegationState,
                        extra_neg_ids: set = None) -> bool:
        if tid in ns.exceptions: return True
        if tid in ns.hard_tracks: return False
        if extra_neg_ids and tid in extra_neg_ids: return False
        info = self.get_info(tid)
        tags = {t.lower() for t in info.get("tags", [])}
        if tags & ns.hard_tags: return False
        if info.get("artist_name","").lower() in ns.hard_artists: return False
        if ns.exclude_embeddings and tid in self.embeddings:
            emb = self.embeddings[tid]
            for ne in ns.exclude_embeddings:
                sim = float(np.dot(emb, ne) /
                            (np.linalg.norm(emb)*np.linalg.norm(ne)+1e-8))
                if sim > ns.embedding_threshold: return False
        return True

    def get_user_cfbpr(self, user_id: str) -> Optional[np.ndarray]:
        return self.user_cfbpr.get(user_id)


# ─────────────────────────────────────────────────────────────────────────────
# 7. DeepSeek 意图解析
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM = """你是音乐推荐系统的查询解析器。根据用户最新一句话输出严格 JSON，不要任何额外文字。

## Track 表结构
tracks(track_id, track_name, artist_name, album_name, release_date, popularity)

## 输出结构
{
  "negation_sql": "SELECT track_id FROM tracks WHERE ... （仅用于排除，无则null）",
  "dense_query": "完整的自然语言检索描述，所有肯定意图都放这里",
  "artist_names": [],
  "track_names": [],
  "album_names": [],
  "genres": [],
  "mood": [],
  "retrieval_mode": "exact_track|exact_artist|exact_album|genre_mood|abstract|default",
  "thought": "一句话推理"
}

## 规则
- SQL 只用于否定/排除：不想听某艺人、不要某风格、排除已播放等
- 所有肯定检索（想听什么）→ dense_query + artist_names/genres/mood
- negation_sql 只包含 WHERE 排除条件，不加 LIMIT/ORDER BY
- 若无排除条件，negation_sql 为 null

## 示例
用户: "播放 Queen 的 Bohemian Rhapsody"
{"negation_sql":null,"dense_query":"Bohemian Rhapsody by Queen","artist_names":["Queen"],"track_names":["Bohemian Rhapsody"],"album_names":[],"genres":[],"mood":[],"retrieval_mode":"exact_track","thought":"精确曲名+艺人，走CDE检索"}

用户: "来首能跳舞的，不要重金属"
{"negation_sql":"SELECT track_id FROM tracks WHERE LOWER(artist_name) IN (SELECT DISTINCT artist_name FROM tracks WHERE LOWER(tag_list) LIKE '%metal%')","dense_query":"energetic dance feel good upbeat","artist_names":[],"track_names":[],"album_names":[],"genres":["dance","electronic"],"mood":["energetic","upbeat"],"retrieval_mode":"genre_mood","thought":"跳舞走CDE dense，重金属用negation_sql排除"}

用户: "不想再听Taylor Swift的歌了"
{"negation_sql":"SELECT track_id FROM tracks WHERE LOWER(artist_name) LIKE '%taylor swift%'","dense_query":null,"artist_names":[],"track_names":[],"album_names":[],"genres":[],"mood":[],"retrieval_mode":"default","thought":"纯否定，只排除Taylor Swift"}
"""

def parse_intent(user_msg: str, history: list, user_profile: dict, client, model="deepseek-chat") -> dict:
    recent = history[-6:]
    hist_str = "\n".join(f"[{m['role']}]: {m['content']}" for m in recent)
    prompt = (
        f"用户画像: {json.dumps(user_profile, ensure_ascii=False)}\n\n"
        f"对话历史:\n{hist_str}\n\n"
        f"用户最新: \"{user_msg}\"\n\n输出JSON:"
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role":"system","content":SYSTEM},
                  {"role":"user","content":prompt}],
        temperature=0.1, max_tokens=600,
    )
    raw = resp.choices[0].message.content.strip().replace("```json","").replace("```","").strip()
    try:
        result = json.loads(raw)
    except Exception:
        result = {
            "negation_sql": None, "dense_query": user_msg,
            "artist_names":[], "track_names":[], "album_names":[],
            "genres":[], "mood":[], "retrieval_mode":"default",
            "thought": "parse error"
        }

    print(f"  💭 [{result.get('retrieval_mode','?')}] {result.get('thought','')}")
    if result.get("dense_query"):  print(f"  🧬 Dense: {result['dense_query']}")
    if result.get("negation_sql"): print(f"  🚫 Neg:   {result['negation_sql']}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 8. execute（SQL → BM25+Dense RRF → CF-BPR rerank）
# ─────────────────────────────────────────────────────────────────────────────

def execute(intent: dict, tools: MusicTools, ns: NegationState,
            current_id: Optional[str], user_id: Optional[str] = None,
            user_profile: dict = None, played_ids: list = None,
            history: list = None,
            top_k: int = TOP_K_FINAL) -> List[str]:
    """
    流程：
      1. SQL        → 候选池
      2. neg_sql    → 排除集合
      3. NegationState → 持久否定
      4. UserHistory enrich → pos/neg/weighted/query/artists/genres
      5. BM25 + Dense(4空间) → 加权 RRF 融合（自适应权重）
      6. CF-BPR 归一化融合最终重排
      7. 补齐 top_k
    """
    dense_query = intent.get("dense_query") or ""
    neg_sql     = intent.get("negation_sql") or ""
    artist_names= intent.get("artist_names") or []
    track_names = intent.get("track_names") or []
    album_names = intent.get("album_names") or []
    genres      = intent.get("genres") or []
    mood        = intent.get("mood") or []
    mode        = intent.get("retrieval_mode") or "default"
    user_culture= (user_profile or {}).get("preferred_musical_culture", "")
    conn        = tools.conn
    played_set  = set(played_ids or [])

    # ── Step 1: 全量候选池 ───────────────────────────────────────────────────
    pool = set(tools.embeddings.keys())

    # ── Step 2: negation_sql → 排除集合 ──────────────────────────────────────
    neg_ids: set = set()
    if neg_sql:
        try:
            neg_ids = {r[0] for r in conn.execute(neg_sql).fetchall()}
            print(f"  🚫 negation excluded: {len(neg_ids):,} tracks")
        except Exception as e:
            print(f"  [NEG SQL ERROR] {e}")

    # ── Step 3: NegationState + played + negation_sql 过滤 ────────────────────
    pool = {t for t in pool
            if t not in played_set and tools.passes_negation(t, ns, neg_ids)}
    print(f"  📦 pool after filter={len(pool):,}")

    cids = list(pool)

    # ── Step 4: UserHistory enrich ────────────────────────────────────────────
    pos_ids = []
    neg_hist_ids: List[str] = []
    weighted_pos: Optional[List[Tuple[str,float]]] = None

    if current_id: pos_ids = [current_id]

    if tools.user_history and user_id:
        pos_ids, neg_hist_ids, weighted_pos, dense_query, artist_names, genres = \
            tools.user_history.enrich(user_id, pos_ids, neg_hist_ids,
                                      dense_query, artist_names, genres)

    semantic_query = dense_query or " ".join(genres + mood)

    # ── Step 5: BM25 + Dense(4空间) + CDE → 加权 RRF ────────────────────────
    has_entity = bool(artist_names or track_names)

    # 自适应权重（BM25 / DenseText / DenseExample / CDE）
    if has_entity:
        bm25_w, dense_text_w, dense_ex_w, cde_w = 0.7, 0.1, 0.1, 0.1
    elif mode in ("exact_track","exact_artist","exact_album"):
        bm25_w, dense_text_w, dense_ex_w, cde_w = 0.6, 0.1, 0.1, 0.2
    elif mode == "abstract":
        bm25_w, dense_text_w, dense_ex_w, cde_w = 0.05, 0.3, 0.3, 0.35
    else:  # genre_mood / default
        bm25_w, dense_text_w, dense_ex_w, cde_w = 0.3, 0.25, 0.25, 0.2

    ranked_lists, weights = [], []

    # BM25
    bm25_scores = tools.bm25_retrieve(
        mode, artist_names, track_names, album_names,
        genres, mood, semantic_query, user_culture,
        candidate_ids=cids,
        top_k=TOP_K_RETRIEVE,
    )
    if bm25_scores:
        ranked_lists.append(bm25_scores); weights.append(bm25_w)

    # Dense text（Qwen3 4空间）
    if semantic_query:
        dt_scores = tools.dense.retrieve_by_query(
            semantic_query,
            candidate_ids=cids,
            top_k=TOP_K_RETRIEVE,
        )
        if dt_scores:
            ranked_lists.append(dt_scores); weights.append(dense_text_w)

    # Dense example（历史 track centroid）
    if pos_ids:
        de_scores = tools.dense.retrieve_by_example(
            pos_ids, neg_hist_ids, weighted_pos,
            candidate_ids=cids,
            top_k=TOP_K_RETRIEVE,
        )
        if de_scores:
            ranked_lists.append(de_scores); weights.append(dense_ex_w)

    # CDE（Conditional Dual Encoder，意图条件化检索）
    if tools.cde_retriever is not None:
        dialog_text = "\n".join(
            h["content"] for h in (history[-8:] if history else [])
        )
        cde_scores = tools.cde_retriever.retrieve(
            dialog_history = dialog_text,
            intent_mode    = mode,
            thought        = intent.get("thought", ""),
            candidate_ids  = cids,
            top_k          = TOP_K_RETRIEVE,
        )
        if cde_scores:
            ranked_lists.append(cde_scores); weights.append(cde_w)
            print(f"  🤖 CDE({mode}): {len(cde_scores)} candidates")

    # RRF 融合
    if not ranked_lists:
        # 最终 fallback：全量按 popularity
        rows = conn.execute(
            "SELECT track_id FROM tracks ORDER BY popularity DESC LIMIT 200"
        ).fetchall()
        fused_ids = [r[0] for r in rows if r[0] in pool]
        rrf_scores_map: Dict[str, float] = {tid: 1.0/(i+1) for i, tid in enumerate(fused_ids)}
    else:
        rrf_scores_map = {}
        for sd, w in zip(ranked_lists, weights):
            for rank, tid in enumerate(sorted(sd, key=sd.get, reverse=True)):
                rrf_scores_map[tid] = rrf_scores_map.get(tid, 0.0) + w / (RRF_K + rank + 1)
        # 只保留 pool 内的
        rrf_scores_map = {t: s for t, s in rrf_scores_map.items() if t in pool}

    print(f"  ✅ RRF candidates={len(rrf_scores_map)}")

    # ── Step 6: CF-BPR 归一化融合 ────────────────────────────────────────────
    user_vec = tools.get_user_cfbpr(user_id) if user_id else None
    is_cold  = user_vec is None
    if is_cold and tools.cold_handler:
        user_vec = tools.cold_handler.get_proxy_vector(user_id or "", user_culture)

    if is_cold:
        alpha, beta = 1.0, 0.0
    else:
        alpha, beta = 0.85, 0.15

    max_rrf = max(rrf_scores_map.values()) if rrf_scores_map else 1.0

    # CF raw scores
    raw_cf: Dict[str, float] = {}
    if user_vec is not None:
        for tid in rrf_scores_map:
            tv = tools.track_cfbpr.get(tid)
            if tv is not None:
                raw_cf[tid] = float(np.dot(user_vec, tv))

    # 归一化 CF 到 [0,1]
    norm_cf: Dict[str, float] = {}
    if raw_cf:
        cf_min, cf_max = min(raw_cf.values()), max(raw_cf.values())
        cf_range = cf_max - cf_min or 1.0
        norm_cf = {tid: (s - cf_min)/cf_range for tid, s in raw_cf.items()}

    final_scores: Dict[str, float] = {}
    for tid, rrf_s in rrf_scores_map.items():
        norm_rrf = rrf_s / max_rrf
        cf_s     = norm_cf.get(tid, 0.0)
        final_scores[tid] = alpha * norm_rrf + beta * cf_s

    ranked = sorted(final_scores, key=final_scores.get, reverse=True)[:top_k*3]
    print(f"  🎯 CF-BPR ranked  is_cold={is_cold}")

    result = ranked[:top_k]

    # ── Step 7: 补齐 ─────────────────────────────────────────────────────────
    if len(result) < top_k and tools.embeddings:
        existing = set(result)
        q = None
        if semantic_query:
            q = tools.dense.encode(semantic_query)
        if q is not None:
            all_scores = {}
            for tid, emb in tools.embeddings.items():
                if tid in existing or tid in played_set: continue
                e = emb / (np.linalg.norm(emb)+1e-8)
                all_scores[tid] = float(np.dot(q[:len(e)] if len(q)>len(e) else q, e))
            padding = sorted(all_scores, key=all_scores.get, reverse=True)[:top_k-len(result)]
            result = result + padding
            print(f"  ➕ padded {len(padding)} → total {len(result)}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 9. 主推荐器
# ─────────────────────────────────────────────────────────────────────────────

class MusicRecommender:
    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com",
                 data_root: Path = DEFAULT_DATA_ROOT):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        data = load_all_data(data_root)
        self.tools = MusicTools(
            tracks_df   = data["track_meta"],
            track_emb_df= data["track_emb"],
            user_emb_df = data["user_emb"],
            user_meta_df= data["user_meta"],
            sessions_df = data["sessions"],
        )
        # session state
        self.history:   list = []
        self.ns         = NegationState()
        self.current:   Optional[str] = None
        self.played:    list = []
        self.profile:   dict = {}
        self._user_id:  Optional[str] = None

    def chat(self, user_msg: str) -> dict:
        self.history.append({"role":"user","content":user_msg})
        intent = parse_intent(user_msg, self.history, self.profile, self.client)
        candidates = execute(
            intent, self.tools, self.ns, self.current,
            user_id=self._user_id, user_profile=self.profile,
            played_ids=self.played, history=self.history,
        )
        if not candidates:
            return {"action":"no_result","track_id":None}
        tid  = candidates[0]
        info = self.tools.get_info(tid)
        self.current = tid
        self.played.append(tid)
        self.history.append({"role":"assistant",
                             "content":f"Playing: {info.get('track_name')} by {info.get('artist_name')}"})
        return {"action":"play","track_id":tid,"info":info,
                "candidates":len(candidates),"intent":intent}


# ─────────────────────────────────────────────────────────────────────────────
# 10. Predict Blind-A
# ─────────────────────────────────────────────────────────────────────────────

def predict_blind(
    data_root:   Path = DEFAULT_DATA_ROOT,
    output_path: Path = Path("./predictions/blind_a_predictions.json"),
    top_k:       int  = TOP_K_FINAL,
    verbose:     bool = True,
):
    api_key = os.environ.get("DEEPSEEK_API_KEY", "sk-xxx")
    blind_dir = Path(data_root) / "Challenge-Blind-A"
    if not blind_dir.exists():
        raise FileNotFoundError(f"Blind-A not found: {blind_dir}")

    print(f"[LOAD] blind_a ← {blind_dir} …", end=" ", flush=True)
    blind_df = _load_local(blind_dir)
    print(f"{len(blind_df):,} sessions")

    # 共享 tools（只初始化一次）
    template = MusicRecommender(api_key, data_root=data_root)

    predictions = []
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for idx, row in blind_df.iterrows():
        session_id   = str(row["session_id"])
        user_id      = str(row["user_id"])
        _prof        = row["user_profile"]
        if isinstance(_prof, (list, np.ndarray)) and len(_prof) > 0: _prof = _prof[0]
        user_profile = _prof if isinstance(_prof, dict) else {}
        conversations = list(row["conversations"]) \
            if isinstance(row["conversations"], (list, np.ndarray)) else []

        if verbose:
            print(f"\n[{idx+1}/{len(blind_df)}] session={session_id[:8]}  turns={len(conversations)}")

        # 每 session 独立状态，共享 tools
        rec = MusicRecommender.__new__(MusicRecommender)
        rec.client   = template.client
        rec.tools    = template.tools
        rec.history  = []
        rec.ns       = NegationState()
        rec.current  = None
        rec.played   = []
        rec.profile  = user_profile
        rec._user_id = user_id

        predict_turn_number = None
        last_user_content   = None

        for turn in conversations:
            if not isinstance(turn, dict): continue
            role        = str(turn.get("role","") or "")
            content     = str(turn.get("content","") or "").strip()
            turn_number = turn.get("turn_number", 0)

            if role == "user":
                rec.history.append({"role":"user","content":content})
                last_user_content   = content
                predict_turn_number = turn_number

            elif role == "music":
                if content:
                    rec.history.append({"role":"assistant","content":f"[played:{content}]"})
                    rec.played.append(content)
                    rec.current = content
                    predict_turn_number = None  # 已有答案，不预测

        # 对最后未回复的 user 消息做推理
        if last_user_content and predict_turn_number is not None:
            result = rec.chat(last_user_content)
            if result["action"] == "play":
                # 取 top_k 候选
                predicted_ids = [result["track_id"]]
                intent = result.get("intent", {})
                extra = execute(intent, rec.tools, rec.ns, rec.current,
                                user_id=user_id, user_profile=user_profile,
                                played_ids=rec.played, top_k=top_k)
                extra = [t for t in extra if t not in set(predicted_ids)]
                predicted_ids = (predicted_ids + extra)[:top_k]
            else:
                predicted_ids = []

            predictions.append({
                "session_id":          session_id,
                "user_id":             user_id,
                "turn_number":         predict_turn_number,
                "predicted_track_ids": predicted_ids,
                "predicted_response":  "Here are some tracks you might enjoy.",
            })
            if verbose:
                print(f"  turn={predict_turn_number}  predicted {len(predicted_ids)} tracks")
        else:
            if verbose: print("  (no prediction needed)")

    tmp = output_path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    tmp.replace(output_path)
    print(f"\n[DONE] {len(predictions)} predictions → {output_path}")
    return predictions


def demo():
    api_key   = os.environ.get("DEEPSEEK_API_KEY","sk-xxx")
    data_root = Path(os.environ.get("DATA_ROOT","./data"))
    rec = MusicRecommender(api_key, data_root=data_root)
    rec.profile  = {"age_group":"20s","country":"China","preferred_musical_culture":"Western Rock"}
    rec._user_id = "demo-user"
    for msg in ["I want something intense and dramatic",
                "Play something similar",
                "No more heavy metal",
                "Give me something to dance to"]:
        print(f"\n👤 {msg}")
        r = rec.chat(msg)
        if r["action"] == "play":
            i = r["info"]
            print(f"🎵 {i['track_name']} — {i['artist_name']}")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=Path(os.environ.get("DATA_ROOT","./data")))
    p.add_argument("--output",    type=Path, default=Path("./predictions/blind_a_predictions.json"))
    p.add_argument("--top-k",     type=int,  default=TOP_K_FINAL)
    p.add_argument("--demo",      action="store_true")
    p.add_argument("--no-verbose",action="store_true")
    args = p.parse_args()
    if args.demo:
        demo()
    else:
        predict_blind(args.data_root, args.output, args.top_k, not args.no_verbose)


if __name__ == "__main__":
    main()