"""
TalkPlay Conversational Music Recommendation — SQL-Enhanced Baseline
=====================================================================
在 music_rec_baseline.py 基础上引入 SQLite 层，用于：
  1. 精确的年代过滤（include / exclude）
  2. 歌手的查找与排除（artist_names / exclude_artists）
  3. 歌曲名的查找与排除（track_names / exclude_tracks）

架构：
  DeepSeek intent parsing
       ↓
  [NEW] SQLFilterEngine  ← 在 track_meta 上建 SQLite 内存库
       ↓  candidate_ids (after SQL include / exclude)
  FilterEngine (year/popularity/duration 数值过滤，原有逻辑)
       ↓
  BM25Retriever + DenseRetriever → RRF Fusion → CFBPRRanker → LGBMReranker

ParsedIntent 新增字段（仅在本文件里扩展，不改动 baseline）：
  exclude_artists:  List[str]   — 排除的歌手（用户说"不要X"）
  exclude_tracks:   List[str]   — 排除的曲名（用户说"不要这首"）
  exclude_years:    List[int]   — 排除的具体年份
  exclude_decades:  List[str]   — 排除的年代段（如"80s"）

DeepSeek prompt 也同步扩充，请 LLM 输出这些新字段。
"""

from __future__ import annotations

# ── 先把 baseline 整包导入，复用其所有类 ──────────────────────────────────────
import music_rec_baseline as _base

# 直接从 baseline 模块引入所有公共名称，以减少重复代码
from music_rec_baseline import (
    # data
    load_all_data, _load_local,
    # dataclasses
    ScoredTrack,
    # pipeline components
    IntentCache, BM25Retriever, DenseRetriever, CFBPRRanker,
    FilterEngine, UserHistoryIndex, ColdUserHandler,
    LGBMReranker, RERANKER_MODEL_FILE,
    # intent
    parse_intent, _dict_to_intent, _fallback_parse,
    INTENT_SYSTEM_PROMPT, DECADE_MAP,
    # utils
    rrf_fusion, evaluate, predict_blind_a, _generate_response,
    # constants
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL,
    DEFAULT_DATA_ROOT, DEFAULT_CACHE_FILE,
    TOP_K_RETRIEVE, TOP_K_FINAL, BM25_WEIGHT, DENSE_WEIGHT,
)

import argparse
import hashlib
import json
import os
import re
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# 0. 扩展版 ParsedIntent
#    在 baseline 的 ParsedIntent 基础上增加 "排除" 字段
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ParsedIntentSQL(_base.ParsedIntent):
    """
    继承自 ParsedIntent，新增 SQL 层所需的排除字段。

    exclude_artists  : 用户明确不想听的歌手，如 ["Taylor Swift", "Drake"]
    exclude_tracks   : 用户明确不想听的曲名，如 ["Shape of You"]
    exclude_decades  : 排除某些年代，如 ["80s", "90s"]
    include_artists  : 仅限这些歌手（精确模式，覆盖 artist_names 含糊搜索）
    include_tracks   : 仅返回这些曲名（精确模式，覆盖 track_names）
    sql_mode         : "include_only" | "exclude_only" | "mixed" | "none"
                       由 SQLFilterEngine._infer_sql_mode() 自动计算
    """
    exclude_artists:  List[str] = field(default_factory=list)
    exclude_tracks:   List[str] = field(default_factory=list)
    exclude_decades:  List[str] = field(default_factory=list)
    include_artists:  List[str] = field(default_factory=list)  # strict include
    include_tracks:   List[str] = field(default_factory=list)  # strict include
    sql_mode:         str       = "none"   # auto-set


# ─────────────────────────────────────────────────────────────────────────────
# 1. 扩展版 DeepSeek Prompt
#    在 baseline prompt 基础上追加排除字段说明
# ─────────────────────────────────────────────────────────────────────────────

INTENT_SYSTEM_PROMPT_SQL = INTENT_SYSTEM_PROMPT.rstrip() + """

Additional SQL-filter fields (include when the user expresses exclusion or strict inclusion):
{
  "exclude_artists":  ["string"],   // artist names the user does NOT want
  "exclude_tracks":   ["string"],   // track titles the user does NOT want
  "exclude_decades":  ["string"],   // decade strings to exclude, e.g. "80s", "2000s"
  "include_artists":  ["string"],   // ONLY these artists (strict, overrides fuzzy BM25)
  "include_tracks":   ["string"]    // ONLY these tracks (strict, overrides fuzzy BM25)
}

Examples:
  "no Taylor Swift please"          → exclude_artists: ["Taylor Swift"]
  "not that old 80s stuff"          → exclude_decades: ["80s"]
  "only Radiohead songs"            → include_artists: ["Radiohead"]
  "don't play Shape of You again"   → exclude_tracks:  ["Shape of You"]
"""


def _dict_to_intent_sql(d: Dict[str, Any]) -> ParsedIntentSQL:
    """将 LLM 返回的 JSON dict 解析成 ParsedIntentSQL（含新字段）。"""
    # 先用 baseline 逻辑填充基础字段
    base = _dict_to_intent(d)
    intent = ParsedIntentSQL(**{k: getattr(base, k)
                                for k in _base.ParsedIntent.__dataclass_fields__})
    # 填充新字段
    for fname in ("exclude_artists", "exclude_tracks", "exclude_decades",
                  "include_artists", "include_tracks"):
        val = d.get(fname)
        if isinstance(val, list) and val:
            setattr(intent, fname, [str(v) for v in val if v])
    return intent


def parse_intent_sql(
    conversation: List[Dict[str, str]],
    cache: IntentCache,
    api_key: str = DEEPSEEK_API_KEY,
) -> Tuple[ParsedIntentSQL, bool]:
    """
    parse_intent 的 SQL 增强版：
      - 使用扩展后的 system prompt（含排除字段）
      - 解析结果为 ParsedIntentSQL
      - 同样读写 IntentCache（key 相同，向后兼容）
    """
    # ── 检查 cache ────────────────────────────────────────────────────────────
    cached = cache.get(conversation)
    if cached is not None:
        # cache 里是 ParsedIntent dict，可能没有 SQL 新字段——仍然有效
        if isinstance(cached, ParsedIntentSQL):
            return cached, True
        # 升级旧格式
        d = asdict(cached)
        return _dict_to_intent_sql(d), True

    # ── DeepSeek API 调用 ─────────────────────────────────────────────────────
    try:
        import requests  # type: ignore

        messages = [{"role": "system", "content": INTENT_SYSTEM_PROMPT_SQL}]
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
                "max_tokens": 600,
                "temperature": 0.0,
            },
            timeout=15,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
        data = json.loads(raw)
        intent = _dict_to_intent_sql(data)

    except Exception as e:
        print(f"[WARN] DeepSeek SQL failed ({type(e).__name__}: {e}). Fallback.")
        base = _fallback_parse(conversation)
        intent = ParsedIntentSQL(**{k: getattr(base, k)
                                    for k in _base.ParsedIntent.__dataclass_fields__})

    # ── 写入 cache（按 ParsedIntent 基础字段写，保证兼容性）────────────────────
    # 注意：SQL 专有字段不写入旧 cache，以避免和 baseline 产生格式冲突
    cache.set(conversation, intent)
    return intent, False


# ─────────────────────────────────────────────────────────────────────────────
# 2. SQLFilterEngine
#    将 track_meta 加载进 SQLite 内存库，执行精确 SQL 过滤
# ─────────────────────────────────────────────────────────────────────────────

class SQLFilterEngine:
    """
    基于 SQLite 的精确 track 过滤器。

    建库时机：__init__ 一次性把 track_meta 写入内存数据库。
    查询时机：每次 recommend() 调用 .apply(intent) 获取 candidate_ids。

    表结构（tracks）
    ──────────────────────────────────────────────
    track_id     TEXT PRIMARY KEY
    track_name   TEXT        (小写，已 strip)
    artist_str   TEXT        (所有 artist_name 拼接，小写)
    album_name   TEXT        (小写)
    release_year INTEGER     (NULL if unparseable)
    popularity   REAL
    duration_ms  REAL
    ──────────────────────────────────────────────

    SQL 过滤策略
    ────────────────────────────────────────────────────────────────────────
    Include（正向限定）优先：当 include_artists 或 include_tracks 非空时，
        用 LIKE 精确匹配，返回命中集合。

    Exclude（排除）次之：从全量或 include 结果中删去：
        - exclude_artists  → artist_str NOT LIKE '%name%'
        - exclude_tracks   → track_name NOT LIKE '%name%'
        - exclude_decades  → release_year NOT BETWEEN lo AND hi

    年代过滤（include/exclude）：
        intent.year_min / year_max         → include range（已由 baseline FilterEngine 处理，
                                               此处仅用于 SQL 验证可覆盖）
        intent.exclude_decades             → 显式排除年代段

    返回值：Set[str]（track_id 集合），空集表示"无约束"（不限制下游）。
    ────────────────────────────────────────────────────────────────────────
    """

    # 用于 LIKE 模糊匹配时的最小字符数，避免单字母误匹配
    MIN_NAME_LEN = 2

    def __init__(self, track_meta: pd.DataFrame):
        t0 = time.time()
        self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._build_db(track_meta)
        n = len(track_meta)
        print(f"[SQLFilter] Built in-memory DB: {n:,} tracks  ({time.time()-t0:.2f}s)")

    # ── DB construction ───────────────────────────────────────────────────────

    @staticmethod
    def _parse_year(v: Any) -> Optional[int]:
        if not v or (isinstance(v, float) and np.isnan(v)):
            return None
        m = re.match(r"(\d{4})", str(v))
        return int(m.group(1)) if m else None

    @staticmethod
    def _safe_str(v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, (list, np.ndarray)):
            return " ".join(str(x) for x in v if x is not None and str(x).strip())
        s = str(v).strip()
        return "" if s in ("nan", "None") else s

    def _build_db(self, df: pd.DataFrame) -> None:
        cur = self._conn.cursor()
        cur.execute("""
            CREATE TABLE tracks (
                track_id     TEXT PRIMARY KEY,
                track_name   TEXT,
                artist_str   TEXT,
                album_name   TEXT,
                release_year INTEGER,
                popularity   REAL,
                duration_ms  REAL
            )
        """)
        cur.execute("CREATE INDEX idx_artist ON tracks(artist_str)")
        cur.execute("CREATE INDEX idx_year   ON tracks(release_year)")
        cur.execute("CREATE INDEX idx_name   ON tracks(track_name)")

        rows = []
        for _, row in df.iterrows():
            tid   = str(row.get("track_id", "") or "").strip()
            if not tid:
                continue
            tname = self._safe_str(row.get("track_name")).lower()
            astr  = self._safe_str(row.get("artist_name")).lower()
            alb   = self._safe_str(row.get("album_name")).lower()
            yr    = self._parse_year(row.get("release_date"))
            pop   = float(row.get("popularity") or 0) if row.get("popularity") is not None else None
            dur   = float(row.get("duration")   or 0) if row.get("duration")   is not None else None
            rows.append((tid, tname, astr, alb, yr, pop, dur))

        cur.executemany(
            "INSERT OR IGNORE INTO tracks VALUES (?,?,?,?,?,?,?)", rows
        )
        self._conn.commit()

    # ── Public API ────────────────────────────────────────────────────────────

    def apply(self, intent: ParsedIntentSQL) -> Optional[Set[str]]:
        """
        根据 SQL intent 返回候选 track_id 集合。

        返回 None  → 无 SQL 约束，下游不限制（全量）
        返回 set   → 只允许集合内的 track_ids（可能为空集）
        """
        sql_mode = self._infer_sql_mode(intent)
        intent.sql_mode = sql_mode

        if sql_mode == "none":
            return None

        # ── 1. 建立 include 候选集 ────────────────────────────────────────────
        include_ids: Optional[Set[str]] = None

        if intent.include_artists:
            include_ids = self._query_include_artists(intent.include_artists)
            # 如果同时有 include_tracks，取交集
            if intent.include_tracks:
                track_ids = self._query_include_tracks(intent.include_tracks)
                include_ids = include_ids & track_ids if include_ids else track_ids

        elif intent.include_tracks:
            include_ids = self._query_include_tracks(intent.include_tracks)

        # 当 include_* 为空但 sql_mode != none，基础集合为全量
        base_ids = include_ids  # None means "all"

        # ── 2. 叠加年代范围（正向，来自 intent.year_min/year_max）─────────────
        #    baseline FilterEngine 也会处理，但 SQL 层提前做可减少后续计算量
        year_filter_ids = self._query_year_range(intent.year_min, intent.year_max)
        if year_filter_ids is not None:
            base_ids = (base_ids & year_filter_ids) if base_ids is not None else year_filter_ids

        # ── 3. 排除操作 ──────────────────────────────────────────────────────
        exclude_ids: Set[str] = set()

        if intent.exclude_artists:
            exclude_ids |= self._query_include_artists(intent.exclude_artists)

        if intent.exclude_tracks:
            exclude_ids |= self._query_include_tracks(intent.exclude_tracks)

        if intent.exclude_decades:
            for decade_str in intent.exclude_decades:
                dk = decade_str.lower().strip()
                if dk in DECADE_MAP:
                    lo, hi = DECADE_MAP[dk]
                    decade_excl = self._query_year_range(lo, hi)
                    if decade_excl:
                        exclude_ids |= decade_excl

        # ── 4. 合并 ──────────────────────────────────────────────────────────
        if base_ids is None:
            # 全量 - 排除集
            if not exclude_ids:
                return None   # 仍然无约束
            all_ids = self._query_all_ids()
            return all_ids - exclude_ids
        else:
            return base_ids - exclude_ids

    # ── Internal query helpers ────────────────────────────────────────────────

    def _query_all_ids(self) -> Set[str]:
        cur = self._conn.cursor()
        cur.execute("SELECT track_id FROM tracks")
        return {row[0] for row in cur.fetchall()}

    def _query_include_artists(self, artists: List[str]) -> Set[str]:
        """LIKE 模糊匹配艺人名，返回命中 track_id 集合。"""
        if not artists:
            return set()
        cur = self._conn.cursor()
        result: Set[str] = set()
        for name in artists:
            name_clean = name.strip().lower()
            if len(name_clean) < self.MIN_NAME_LEN:
                continue
            cur.execute(
                "SELECT track_id FROM tracks WHERE artist_str LIKE ?",
                (f"%{name_clean}%",)
            )
            result.update(row[0] for row in cur.fetchall())
        return result

    def _query_include_tracks(self, tracks: List[str]) -> Set[str]:
        """LIKE 模糊匹配曲名，返回命中 track_id 集合。"""
        if not tracks:
            return set()
        cur = self._conn.cursor()
        result: Set[str] = set()
        for name in tracks:
            name_clean = name.strip().lower()
            if len(name_clean) < self.MIN_NAME_LEN:
                continue
            cur.execute(
                "SELECT track_id FROM tracks WHERE track_name LIKE ?",
                (f"%{name_clean}%",)
            )
            result.update(row[0] for row in cur.fetchall())
        return result

    def _query_year_range(
        self, year_min: Optional[int], year_max: Optional[int]
    ) -> Optional[Set[str]]:
        """年份区间过滤，两者均为 None 时返回 None（不限制）。"""
        if year_min is None and year_max is None:
            return None
        cur = self._conn.cursor()
        if year_min is not None and year_max is not None:
            cur.execute(
                "SELECT track_id FROM tracks WHERE release_year BETWEEN ? AND ?",
                (year_min, year_max)
            )
        elif year_min is not None:
            cur.execute(
                "SELECT track_id FROM tracks WHERE release_year >= ?", (year_min,)
            )
        else:
            cur.execute(
                "SELECT track_id FROM tracks WHERE release_year <= ?", (year_max,)
            )
        return {row[0] for row in cur.fetchall()}

    @staticmethod
    def _infer_sql_mode(intent: ParsedIntentSQL) -> str:
        has_include = bool(intent.include_artists or intent.include_tracks)
        has_exclude = bool(
            intent.exclude_artists or intent.exclude_tracks or intent.exclude_decades
        )
        has_year    = bool(intent.year_min or intent.year_max)

        if has_include and has_exclude:
            return "mixed"
        if has_include:
            return "include_only"
        if has_exclude:
            return "exclude_only"
        if has_year:
            # year range alone → let baseline FilterEngine handle it, SQL is idle
            return "none"
        return "none"

    def debug_query(self, intent: ParsedIntentSQL) -> Dict[str, Any]:
        """
        调试辅助：独立于 apply()，直接读 intent 字段并执行带 EXPLAIN 的查询，
        返回完整的 SQL 过滤摘要。不修改 intent 的任何状态。
        """
        # ── 重新推断 sql_mode（不依赖 apply() 的副作用）──────────────────────
        sql_mode = self._infer_sql_mode(intent)

        # ── 构造并执行各子查询，收集结果摘要 ─────────────────────────────────
        cur = self._conn.cursor()
        queries_run: List[Dict[str, Any]] = []

        def _run(label: str, sql: str, params: tuple) -> List[str]:
            cur.execute(sql, params)
            rows = [r[0] for r in cur.fetchall()]
            queries_run.append({"label": label, "sql": sql,
                                 "params": list(params), "hit_count": len(rows)})
            return rows

        # include_artists
        include_artist_ids: Set[str] = set()
        for name in intent.include_artists:
            ids = _run(
                f"include_artist:{name!r}",
                "SELECT track_id FROM tracks WHERE artist_str LIKE ?",
                (f"%{name.lower()}%",),
            )
            include_artist_ids.update(ids)

        # include_tracks
        include_track_ids: Set[str] = set()
        for name in intent.include_tracks:
            ids = _run(
                f"include_track:{name!r}",
                "SELECT track_id FROM tracks WHERE track_name LIKE ?",
                (f"%{name.lower()}%",),
            )
            include_track_ids.update(ids)

        # year range (positive)
        year_ids: Optional[Set[str]] = None
        if intent.year_min is not None or intent.year_max is not None:
            if intent.year_min is not None and intent.year_max is not None:
                sql = "SELECT track_id FROM tracks WHERE release_year BETWEEN ? AND ?"
                params = (intent.year_min, intent.year_max)
            elif intent.year_min is not None:
                sql = "SELECT track_id FROM tracks WHERE release_year >= ?"
                params = (intent.year_min,)
            else:
                sql = "SELECT track_id FROM tracks WHERE release_year <= ?"
                params = (intent.year_max,)
            year_ids = set(_run("year_range", sql, params))

        # exclude_artists
        exclude_artist_ids: Set[str] = set()
        for name in intent.exclude_artists:
            ids = _run(
                f"exclude_artist:{name!r}",
                "SELECT track_id FROM tracks WHERE artist_str LIKE ?",
                (f"%{name.lower()}%",),
            )
            exclude_artist_ids.update(ids)

        # exclude_tracks
        exclude_track_ids: Set[str] = set()
        for name in intent.exclude_tracks:
            ids = _run(
                f"exclude_track:{name!r}",
                "SELECT track_id FROM tracks WHERE track_name LIKE ?",
                (f"%{name.lower()}%",),
            )
            exclude_track_ids.update(ids)

        # exclude_decades
        exclude_decade_ids: Set[str] = set()
        for dk in intent.exclude_decades:
            key = dk.lower().strip()
            if key in DECADE_MAP:
                lo, hi = DECADE_MAP[key]
                ids = _run(
                    f"exclude_decade:{dk!r}",
                    "SELECT track_id FROM tracks WHERE release_year BETWEEN ? AND ?",
                    (lo, hi),
                )
                exclude_decade_ids.update(ids)

        # ── 计算最终候选集大小 ────────────────────────────────────────────────
        # （镜像 apply() 的合并逻辑，但不写入 intent）
        base: Optional[Set[str]] = None
        if include_artist_ids or include_track_ids:
            base = include_artist_ids | include_track_ids
        if year_ids is not None:
            base = (base & year_ids) if base is not None else year_ids

        all_exclude = exclude_artist_ids | exclude_track_ids | exclude_decade_ids

        if base is None:
            if not all_exclude:
                result_count = "all (no constraints)"
                result_sample: List[str] = []
            else:
                cur.execute("SELECT COUNT(*) FROM tracks")
                total = cur.fetchone()[0]
                result_count = total - len(all_exclude)
                cur.execute("SELECT track_id FROM tracks LIMIT 200")
                result_sample = [
                    r[0] for r in cur.fetchall() if r[0] not in all_exclude
                ][:5]
        else:
            final = base - all_exclude
            result_count = len(final)
            result_sample = list(final)[:5]

        # ── 汇总 ─────────────────────────────────────────────────────────────
        return {
            "sql_mode":          sql_mode,
            "intent_fields": {
                "include_artists":  intent.include_artists,
                "include_tracks":   intent.include_tracks,
                "exclude_artists":  intent.exclude_artists,
                "exclude_tracks":   intent.exclude_tracks,
                "exclude_decades":  intent.exclude_decades,
                "year_range":       [intent.year_min, intent.year_max],
            },
            "sub_query_results": queries_run,
            "result_count":      result_count,
            "result_sample_ids": result_sample,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 3a. EditDistanceMatcher
#     在 track_meta 的 track_name 字段上做 Levenshtein 编辑距离查找。
#     用于 exact_track 模式：把距离 < MAX_EDIT_DIST 的所有匹配按距离升序
#     pin 在最终推荐列表的最前面，保证精确/近似命中不会被 CF/BPR 推后。
# ─────────────────────────────────────────────────────────────────────────────

MAX_EDIT_DIST = 3   # 编辑距离阈值（含），可按需调整


def _levenshtein(a: str, b: str) -> int:
    """标准 Levenshtein 距离，O(|a|·|b|) DP。"""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    # 只保留两行，节省内存
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * lb
        for j, cb in enumerate(b, 1):
            curr[j] = min(
                prev[j] + 1,          # 删除
                curr[j - 1] + 1,      # 插入
                prev[j - 1] + (ca != cb),  # 替换
            )
        prev = curr
    return prev[lb]


class EditDistanceMatcher:
    """
    对 track_meta 的所有 track_name 做编辑距离近似匹配。

    构建：O(N) — 把每条 track 的第一个 track_name 小写存入列表。
    查询：O(N·L) — 对每个 query token 扫描全表，L = 平均名称长度。
         对于 N≈47k 条目和平均 20 字符，约 100ms/query，可接受。

    返回：List[Tuple[str, int]]  — (track_id, edit_distance)，按距离升序，
          距离相同时按 popularity 降序。只返回 distance <= max_dist 的结果。
    """

    def __init__(self, track_meta: pd.DataFrame, max_dist: int = MAX_EDIT_DIST):
        self.max_dist = max_dist
        self._entries: List[Tuple[str, str, float]] = []  # (track_id, norm_name, popularity)

        def _safe_first(v) -> str:
            if isinstance(v, (list, np.ndarray)):
                return str(v[0]).strip().lower() if len(v) > 0 else ""
            s = str(v).strip().lower()
            return "" if s in ("nan", "none", "") else s

        for _, row in track_meta.iterrows():
            tid  = str(row.get("track_id", "") or "").strip()
            name = _safe_first(row.get("track_name"))
            pop  = float(row.get("popularity") or 0)
            if tid and name:
                self._entries.append((tid, name, pop))

        print(f"[EditDistanceMatcher] {len(self._entries):,} track names indexed "
              f"(max_dist={max_dist})")

    def search(
        self,
        query_names: List[str],
        candidate_ids: Optional[Set[str]] = None,
    ) -> List[Tuple[str, int]]:
        """
        对 query_names 中每个名称执行编辑距离扫描。

        参数
        ----
        query_names   : intent.track_names（可含多个，取距离最小值）
        candidate_ids : 若非 None，只在此集合内搜索（已经过 SQL / 数值过滤）

        返回
        ----
        [(track_id, min_edit_dist), ...]，按 dist ASC、popularity DESC 排序，
        只含 dist <= self.max_dist 的结果，track_id 不重复。
        """
        if not query_names:
            return []

        queries = [q.strip().lower() for q in query_names if q.strip()]
        if not queries:
            return []

        best: Dict[str, int] = {}   # track_id → 最小编辑距离

        for tid, name, _pop in self._entries:
            if candidate_ids is not None and tid not in candidate_ids:
                continue
            min_d = min(_levenshtein(q, name) for q in queries)
            if min_d <= self.max_dist:
                if tid not in best or min_d < best[tid]:
                    best[tid] = min_d

        if not best:
            return []

        # 构建结果，dist 相同时以 popularity 降序为第二关键字
        pop_map = {tid: pop for tid, _, pop in self._entries if tid in best}
        result = sorted(
            best.items(),
            key=lambda x: (x[1], -pop_map.get(x[0], 0.0))
        )
        return result   # [(track_id, dist), ...]


# ─────────────────────────────────────────────────────────────────────────────
# 3. SQL 增强版 MusicRecPipeline
# ─────────────────────────────────────────────────────────────────────────────

class MusicRecPipelineSQL(_base.MusicRecPipeline):
    """
    在 baseline MusicRecPipeline 基础上：
      1. 构建时额外创建 SQLFilterEngine 和 EditDistanceMatcher
      2. recommend() 中在 FilterEngine 之前先跑 SQL 过滤
      3. 使用 parse_intent_sql 解析 ParsedIntentSQL（含排除字段）
      4. exact_track 模式下用编辑距离匹配，把距离 < MAX_EDIT_DIST 的结果
         按距离升序 pin 在最终列表最前面
    """

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
        # 调用父类 __init__（构建 FilterEngine / BM25 / Dense / CF / History）
        super().__init__(
            track_meta=track_meta,
            track_emb=track_emb,
            user_emb=user_emb,
            user_meta=user_meta,
            sessions_df=sessions_df,
            cache_file=cache_file,
            deepseek_api_key=deepseek_api_key,
        )
        # 额外建立 SQL 过滤引擎
        print("[INIT] SQLFilterEngine …")
        self.sql_filter = SQLFilterEngine(track_meta)
        # 额外建立编辑距离匹配器（exact_track pin 用）
        print("[INIT] EditDistanceMatcher …")
        self.edit_matcher = EditDistanceMatcher(track_meta)
        print("[INIT] SQL-Enhanced Pipeline ready.\n")

    # ── Factory ──────────────────────────────────────────────────────────────

    @classmethod
    def from_local(
        cls,
        data_root: Path = DEFAULT_DATA_ROOT,
        cache_file: Path = DEFAULT_CACHE_FILE,
        deepseek_api_key: str = DEEPSEEK_API_KEY,
    ) -> "MusicRecPipelineSQL":
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

    # ── Recommend（覆盖父类）────────────────────────────────────────────────

    def recommend(
        self,
        user_id: str,
        conversation: List[Dict[str, str]],
        top_k: int = TOP_K_FINAL,
        verbose: bool = True,
        played_track_ids: Optional[List[str]] = None,
        listener_goal: str = "",
        user_culture: str = "",
        goal_type: str = "",
        specificity: str = "",
        turn_number: int = 0,
    ) -> Tuple[List[ScoredTrack], ParsedIntentSQL]:
        t0 = time.time()

        # ── Step 1. 解析 Intent（使用 SQL 扩展版 prompt）──────────────────────
        intent, from_cache = parse_intent_sql(
            conversation, self.cache, self._api_key
        )

        # 当 cache 返回旧格式 ParsedIntent，升级为 ParsedIntentSQL
        if not isinstance(intent, ParsedIntentSQL):
            d = asdict(intent)
            intent = ParsedIntentSQL(**{k: d.get(k, v)
                                        for k, v in asdict(ParsedIntentSQL()).items()})

        # 注入当前 session 已播放的 track
        if played_track_ids:
            intent.positive_track_ids = list(dict.fromkeys(
                intent.positive_track_ids + played_track_ids
            ))

        # 注入 session 元数据
        if listener_goal:
            intent.listener_goal = listener_goal
        if user_culture:
            intent.user_culture = user_culture

        # 用历史数据丰富 intent（warm users）
        if self.user_history:
            intent = self.user_history.enrich_intent(user_id, intent)
            if verbose and self.user_history.has_history(user_id):
                n_hist = len(self.user_history.get_accepted_tracks(user_id))
                top_a  = self.user_history.get_top_artists(user_id, 3)
                print(f"[1b] UserHistory  {n_hist} accepted  top_artists={top_a}")

        # 分类 retrieval_mode
        if not intent.retrieval_mode:
            intent.retrieval_mode = BM25Retriever._classify_mode(intent)

        src = "cache" if from_cache else "API"
        if verbose:
            print(
                f"[1] Intent ({src})  mode={intent.retrieval_mode}  "
                f"abstract={intent.is_abstract}  decade={intent.decade}  "
                f"artists={intent.artist_names}  albums={intent.album_names}  "
                f"tracks={intent.track_names}  genres={intent.genres}\n"
                f"    [SQL] include_artists={intent.include_artists}  "
                f"include_tracks={intent.include_tracks}  "
                f"exclude_artists={intent.exclude_artists}  "
                f"exclude_tracks={intent.exclude_tracks}  "
                f"exclude_decades={intent.exclude_decades}"
            )

        # ── Step 2. SQL 过滤（精确 include / exclude）────────────────────────
        sql_ids: Optional[Set[str]] = self.sql_filter.apply(intent)

        if verbose:
            if sql_ids is None:
                print(f"[2-SQL] No SQL constraints — full corpus")
            else:
                print(f"[2-SQL] SQL filter: {len(sql_ids):,} tracks  "
                      f"(mode={intent.sql_mode})")

        # 将 SQL 结果与原有数值过滤（year/popularity/duration）合并
        # 原 FilterEngine 基于 Pandas，继续保留用于数值区间过滤
        filtered_df = self.filter.apply(intent)
        filter_active = any([
            intent.year_min, intent.year_max,
            intent.popularity_min, intent.popularity_max,
            intent.duration_min_ms, intent.duration_max_ms,
        ])

        if filter_active and len(filtered_df) < len(self.filter.df):
            numeric_ids: Optional[Set[str]] = set(filtered_df["track_id"].tolist())
        else:
            numeric_ids = None

        # 取 SQL 结果 ∩ 数值过滤结果
        if sql_ids is not None and numeric_ids is not None:
            merged_ids = sql_ids & numeric_ids
        elif sql_ids is not None:
            merged_ids = sql_ids
        elif numeric_ids is not None:
            merged_ids = numeric_ids
        else:
            merged_ids = None   # 无约束

        # 转成有序列表（供 BM25 / Dense 使用）
        cids: Optional[List[str]] = list(merged_ids) if merged_ids is not None else None

        if verbose:
            n_total = len(self.filter.df)
            n_pass  = len(cids) if cids is not None else n_total
            print(f"[2] Filter  {n_pass:,}/{n_total:,} tracks"
                  + (" (no filter active)" if cids is None else ""))

        # ── Step 3. BM25 检索 ────────────────────────────────────────────────
        bm25_scores = self.bm25.retrieve(
            intent, candidate_ids=cids, top_k=TOP_K_RETRIEVE
        )
        if verbose:
            print(f"[3] BM25    {len(bm25_scores)} candidates")

        # ── Step 2.5. 编辑距离匹配（仅 exact_track 模式）────────────────────
        # 在 BM25 之后拿到 cids（已经过 SQL + 数值过滤），对 query track names
        # 做编辑距离扫描，结果将在最终输出时 pin 到列表最前面。
        edit_pinned: List[Tuple[str, int]] = []   # [(track_id, dist), ...]
        if intent.retrieval_mode == "exact_track" and intent.track_names:
            cids_set = set(cids) if cids is not None else None
            edit_pinned = self.edit_matcher.search(intent.track_names, cids_set)
            if verbose:
                print(f"[2.5] EditDist  {len(edit_pinned)} pinned tracks "
                      f"(dist<={MAX_EDIT_DIST})  "
                      + (f"top={edit_pinned[0]} " if edit_pinned else ""))

        # ── Step 4. Dense 检索 ───────────────────────────────────────────────
        dense_scores: Dict[str, float] = {}
        _has_entity = bool(intent.artist_names or intent.track_names)

        parts = []
        if intent.listener_goal:
            parts.append(intent.listener_goal)
        if intent.semantic_query and intent.semantic_query not in parts:
            parts.append(intent.semantic_query)
        if intent.user_culture:
            parts.append(intent.user_culture)
        enriched_query = " ".join(parts).strip()

        if enriched_query:
            orig_sq = intent.semantic_query
            intent.semantic_query = enriched_query
            dense_scores = self.dense.retrieve(
                intent, candidate_ids=cids, top_k=TOP_K_RETRIEVE
            )
            intent.semantic_query = orig_sq

        # 基于已播放曲目的 example-based 检索
        if intent.positive_track_ids:
            if intent.weighted_positive_track_ids:
                _w_map = {tid: wt for tid, wt in intent.weighted_positive_track_ids}
                pos_weights = [_w_map.get(tid, 1.0) for tid in intent.positive_track_ids]
            else:
                pos_weights = None

            ex_scores = self.dense.retrieve_by_example(
                intent.positive_track_ids, intent.negative_track_ids,
                candidate_ids=cids, top_k=TOP_K_RETRIEVE,
                pos_weights=pos_weights,
            )
            if dense_scores:
                fused: Dict[str, float] = {}
                for rl, w in [(dense_scores, 0.4), (ex_scores, 0.6)]:
                    for rank, (tid, _) in enumerate(
                        sorted(rl.items(), key=lambda x: x[1], reverse=True), 1
                    ):
                        fused[tid] = fused.get(tid, 0.0) + w / (10 + rank)
                dense_scores = fused
            else:
                dense_scores = ex_scores

        if verbose:
            print(f"[4] Dense   {len(dense_scores)} candidates  "
                  f"(played={len(intent.positive_track_ids)} example tracks)")

        # ── Step 5. RRF 融合 ─────────────────────────────────────────────────
        bm25_weight  = 0.4
        dense_weight = 0.6

        if _has_entity:
            bm25_weight, dense_weight = 0.8, 0.2
        elif intent.retrieval_mode in ["exact_track", "exact_artist", "exact_album"]:
            bm25_weight, dense_weight = 0.7, 0.3
        elif intent.retrieval_mode in ["abstract", "genre_mood"]:
            bm25_weight, dense_weight = 0.3, 0.7

        rrf_scores = rrf_fusion(
            [bm25_scores, dense_scores], [bm25_weight, dense_weight]
        )

        if verbose:
            print(f"[5] RRF     {len(rrf_scores)} candidates  "
                  f"(bm25:{bm25_weight:.1f} dense:{dense_weight:.1f})")

        # ── Step 6. CF-BPR 重排 ──────────────────────────────────────────────
        is_cold = (user_id not in self.ranker._user_cf)
        if self.cold_handler and is_cold:
            if intent.retrieval_mode in ["abstract", "genre_mood", "default"]:
                culture_tokens = self.cold_handler.get_culture_tokens(
                    user_id, override=user_culture
                )
                if culture_tokens and not intent.user_culture:
                    intent.user_culture = " ".join(culture_tokens)
                    bm25_scores = self.bm25.retrieve(
                        intent, candidate_ids=cids, top_k=TOP_K_RETRIEVE
                    )
                    rrf_scores = rrf_fusion(
                        [bm25_scores, dense_scores], [bm25_weight, dense_weight]
                    )
                    if verbose:
                        print(f"[3b] BM25 re-run with cold-user culture tokens")

        n_played = len(intent.positive_track_ids)
        ranked = self.ranker.rank(
            user_id, rrf_scores,
            top_k=top_k + n_played + 5,
            cold_handler=self.cold_handler,
            user_culture=user_culture,
        )

        # 排除已播放 + 历史拒绝 + SQL 排除（double-check）
        exclude_set = set(intent.positive_track_ids)
        if self.user_history:
            exclude_set |= self.user_history.get_rejected_tracks(user_id)
        # 如果 SQL 层返回了约束集，额外再过滤一次（容错）
        if sql_ids is not None:
            ranked = [st for st in ranked if st.track_id in sql_ids]
        if exclude_set:
            ranked = [st for st in ranked if st.track_id not in exclude_set]

        # ── Step 7. LightGBM 重排（如已训练）────────────────────────────────
        if self.reranker_lgbm._model is not None:
            ranked = self.reranker_lgbm.rerank(
                ranked, intent,
                track_meta_index=self._meta,
                user_history=self.user_history,
                user_id=user_id,
                turn_number=turn_number,
                goal_type=goal_type,
                specificity=specificity,
            )
            if verbose:
                print(f"[7] LGBMReranker applied")

        # ── Step 8. 编辑距离 Pin（exact_track 模式）──────────────────────────
        # 把编辑距离命中的 track 按距离升序 pin 在列表最前面；
        # 后续普通 ranked 去重追加，总数截断到 top_k。
        # 此步在 LightGBM 重排之后执行，确保 pin 不被任何后续 reranker 打乱。
        if edit_pinned:
            pinned_ids_ordered = [
                tid for tid, _dist in edit_pinned
                if tid not in exclude_set
            ]
            if pinned_ids_ordered:
                ranked_map: Dict[str, ScoredTrack] = {st.track_id: st for st in ranked}
                pinned_sts: List[ScoredTrack] = []
                for tid, dist in edit_pinned:
                    if tid in exclude_set:
                        continue
                    if tid in ranked_map:
                        st = ranked_map[tid]
                    else:
                        # track 命中了编辑距离但不在 CF/BPR 候选里，强制插入
                        st = ScoredTrack(track_id=tid, final_score=0.0)
                    st.metadata["edit_dist"] = dist   # 调试用
                    pinned_sts.append(st)

                pinned_set = {st.track_id for st in pinned_sts}
                rest = [st for st in ranked if st.track_id not in pinned_set]
                ranked = (pinned_sts + rest)[:top_k]

                if verbose:
                    top3 = [(t, d) for t, d in edit_pinned[:3]]
                    print(f"[8] EditPin  {len(pinned_sts)} pinned  "
                          f"{len(rest)} remaining  top3={top3}")
        else:
            ranked = ranked[:top_k]

        if verbose:
            print(f"[6] Rerank  top-{len(ranked)}  ({time.time()-t0:.2f}s total)\n")

        # 附加展示用元数据（与 baseline 保持一致，含 tags 字段供 _generate_response 使用）
        def _first(v, default="?"):
            if isinstance(v, (list, np.ndarray)):
                return str(v[0]) if len(v) > 0 else default
            return str(v) if v not in (None, "", "nan", "None") else default

        def _to_list(v):
            if v is None:
                return []
            if isinstance(v, np.ndarray):
                return v.tolist()
            if isinstance(v, list):
                return v
            return []

        for st in ranked:
            if st.track_id in self._meta.index:
                row = self._meta.loc[st.track_id]
                st.metadata = {
                    "track_name":   _first(row.get("track_name")),
                    "artist_name":  _first(row.get("artist_name")),
                    "album_name":   _first(row.get("album_name")),
                    "release_date": str(row.get("release_date") or "")[:10],
                    "popularity":   float(row.get("popularity") or 0),
                    "tags":         _to_list(row.get("tag_list"))[:5],  # _generate_response 需要
                }

        return ranked, intent


# ─────────────────────────────────────────────────────────────────────────────
# 4. SQL-Aware Response Generation
#    在 baseline _generate_response 基础上，把 SQL 排除/限定信息注入 prompt，
#    让 LLM 在回复里自然体现用户的约束（如"已帮你排除了 Taylor Swift 的歌"）
# ─────────────────────────────────────────────────────────────────────────────

RESPONSE_SYSTEM_PROMPT_BASE = """
You are a music recommendation assistant having a natural conversation.
Given a conversation history and recommended tracks, write a response that:
1. Directly and warmly addresses what the user asked for.
2. Mentions 2-3 specific track or artist names from the list to feel concrete.
3. Explains WHY these fit (mood, genre, era, energy, lyrical theme).
4. Invites further refinement if needed ("Let me know if you want more like X").
Keep it conversational, 3-5 sentences, under 100 words. No bullet points.

CRITICAL — vary your opening every single time. Forbidden openers (never use):
  "I've picked", "I've curated", "Here are", "I found", "I've selected", "Check out"
Instead, rotate freely among styles like:
  - Lead with the mood/vibe  e.g. "Perfect for a rainy afternoon — ..."
  - Lead with an artist name e.g. "Radiohead's OK Computer fits right in here, ..."
  - Echo the user's request  e.g. "You're after something mellow? ..."
  - Lead with the era        e.g. "Deep in 90s alt-rock territory: ..."
  - Lead with a feeling      e.g. "There's a wistful, late-night quality to these picks ..."
  - Ask a light question     e.g. "Ever caught yourself humming Interpol? This one builds on that..."
Each response should feel like it was written fresh, not from a template.
"""

RESPONSE_SYSTEM_PROMPT_SQL = """
You are a music recommendation assistant having a natural conversation.
Given a conversation history, recommended tracks, and any active filters, write a response that:
1. Directly and warmly addresses what the user asked for.
2. Mentions 2-3 specific track or artist names from the list to feel concrete.
3. Explains WHY these fit (mood, genre, era, energy, lyrical theme).
4. If any artists, tracks, or decades were excluded, briefly acknowledge it naturally
   (e.g. "Since you wanted to skip Taylor Swift, I've focused on...").
5. Invites further refinement if needed ("Let me know if you want more like X").
Keep it conversational, 3-5 sentences, under 110 words. No bullet points.

CRITICAL — vary your opening every single time. Forbidden openers (never use):
  "I've picked", "I've curated", "Here are", "I found", "I've selected", "Check out"
Instead, rotate freely among styles like:
  - Lead with the mood/vibe  e.g. "Perfect for a rainy afternoon — ..."
  - Lead with an artist name e.g. "Radiohead's OK Computer fits right in here, ..."
  - Echo the user's request  e.g. "You're after something mellow? ..."
  - Lead with the era        e.g. "Deep in 90s alt-rock territory: ..."
  - Lead with a feeling      e.g. "There's a wistful, late-night quality to these picks ..."
  - Ask a light question     e.g. "Ever caught yourself humming Interpol? This one builds on that..."
Each response should feel like it was written fresh, not from a template.
"""


def _generate_response_sql(
    conversation: List[Dict[str, str]],
    ranked: List[ScoredTrack],
    intent: Optional[ParsedIntentSQL] = None,
    api_key: str = DEEPSEEK_API_KEY,
) -> str:
    """
    SQL 增强版 response 生成器。

    在 baseline _generate_response 的基础上，将 SQL 排除/限定信息拼入 prompt，
    让 LLM 能在回复里自然提及用户的约束。

    当 intent 为 None 或无 SQL 约束时，行为与 baseline 完全相同。
    """
    # ── 构造 track 列表（同 baseline）────────────────────────────────────────
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

    # ── 构造 SQL 约束摘要（新增部分）────────────────────────────────────────
    constraint_lines: List[str] = []
    if intent is not None:
        if intent.exclude_artists:
            constraint_lines.append(
                "Excluded artists: " + ", ".join(intent.exclude_artists)
            )
        if intent.exclude_tracks:
            constraint_lines.append(
                "Excluded tracks: " + ", ".join(intent.exclude_tracks)
            )
        if intent.exclude_decades:
            constraint_lines.append(
                "Excluded decades: " + ", ".join(intent.exclude_decades)
            )
        if intent.include_artists:
            constraint_lines.append(
                "Restricted to artists: " + ", ".join(intent.include_artists)
            )
        if intent.include_tracks:
            constraint_lines.append(
                "Restricted to tracks: " + ", ".join(intent.include_tracks)
            )
        if intent.year_min or intent.year_max:
            yr_range = f"{intent.year_min or '?'} – {intent.year_max or '?'}"
            constraint_lines.append(f"Year range: {yr_range}")

    constraints_block = (
        "\nActive filters:\n" + "\n".join(f"  • {c}" for c in constraint_lines)
        if constraint_lines else ""
    )

    # ── 清洗对话历史（同 baseline）──────────────────────────────────────────
    _UUID_RE = re.compile(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I
    )
    conv_text = "\n".join(
        f"{m['role'].capitalize()}: {m['content']}"
        for m in conversation[-6:]
        if m.get("content", "").strip()
        and not m["content"].startswith("[Previously played")
        and not _UUID_RE.search(m["content"])
    )

    prompt = (
        f"Conversation so far:\n{conv_text}\n\n"
        f"Recommended tracks:\n{tracks_block}"
        + constraints_block
        + "\n\nWrite a natural recommendation response to continue this conversation."
    )

    # ── 选择 system prompt ──────────────────────────────────────────────────
    system_prompt = (
        RESPONSE_SYSTEM_PROMPT_SQL if constraint_lines
        else RESPONSE_SYSTEM_PROMPT_BASE
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
                    {"role": "system", "content": system_prompt},
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
        # 降级 fallback：多样化开头，避免每次都是 "I've picked"
        import random as _random
        if ranked:
            m0      = ranked[0].metadata
            m1      = ranked[1].metadata if len(ranked) > 1 else m0
            name0   = m0.get("track_name", "these tracks")
            artist0 = m0.get("artist_name", "")
            artist1 = m1.get("artist_name", "")
            tags0   = m0.get("tags", [])
            style   = ", ".join(tags0[:2]) if tags0 else "this style"
            year0   = m0.get("release_date", "")[:4]
            era     = f"{year0[:3]}0s" if year0 else "this era"

            excl_note = ""
            if intent and intent.exclude_artists:
                excl_note = (
                    " Skipping " + ", ".join(intent.exclude_artists) + " as requested."
                )

            openers = [
                f"Something in the {style} territory — {name0}"
                + (f" by {artist0}" if artist0 else "") + " sets the tone perfectly.",
                f"Right in the {era} {style} sweet spot: {name0}"
                + (f" by {artist0}" if artist0 else "") + " leads the way.",
                f"{artist0 or name0} captures exactly that vibe —"
                + (f" {name0} in particular" if artist0 else "") + " is a strong pick here.",
                f"That {style} mood you're after? {name0}"
                + (f" by {artist0}" if artist0 else "") + " is where I'd start.",
                f"If {artist0 or name0} resonates,"
                + (f" {artist1}" if artist1 and artist1 != artist0 else " the rest of this list")
                + " should too — real {style} energy throughout.".format(style=style),
            ]
            opener_sentence = _random.choice(openers)
        else:
            excl_note = ""
            opener_sentence = f"Couldn't find a strong match this time — try broadening the request a little."

        return (
            f"{opener_sentence}{excl_note} "
            "Let me know if you'd like to shift the mood or explore a different direction!"
        )


def predict_blind_a_sql(
    pipeline: "MusicRecPipelineSQL",
    blind_df: pd.DataFrame,
    output_path: Path,
    top_k: int = 20,
    generate_response: bool = True,
    verbose: bool = True,
) -> List[Dict]:
    """
    SQL 增强版 predict_blind_a。

    与 baseline 版完全相同的流程，唯一区别：
      - 调用 _generate_response_sql（传入 intent，含 SQL 约束信息）
      - 输出文件名默认带 _sql 后缀

    可以直接替代 baseline 的 predict_blind_a 使用。
    """
    from music_rec_baseline import (
        _build_conversation_history,
        _extract_positive_ids_from_history,
    )

    try:
        from tqdm import tqdm  # type: ignore
        _tqdm_available = True
    except ImportError:
        _tqdm_available = False

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    predictions: List[Dict] = []
    rows      = list(blind_df.iterrows())
    iterator  = (
        tqdm(rows, desc="Predicting (SQL)", unit="session", dynamic_ncols=True)
        if _tqdm_available else rows
    )
    t_start = time.time()

    for idx, (_, row) in enumerate(iterator, 1):
        session_id = row["session_id"]
        user_id    = row["user_id"]
        turns      = row["conversations"]

        turn_number = len(turns) // 3 + 1

        if _tqdm_available:
            iterator.set_postfix(
                session=session_id[:8],
                turn=turn_number,
                cached=len(pipeline.cache),
                refresh=False,
            )
        elif verbose:
            n_sessions = len(rows)
            elapsed    = time.time() - t_start
            eta        = (elapsed / idx) * (n_sessions - idx) if idx > 1 else 0
            print(f"[{idx}/{n_sessions}] session={session_id[:8]}  "
                  f"turn_number={turn_number}  "
                  f"elapsed={elapsed:.0f}s  eta={eta:.0f}s")

        intent_history, response_history = _build_conversation_history(turns)

        if not intent_history:
            msg = f"[SKIP] No user turns in session {session_id}"
            (tqdm.write if _tqdm_available else print)(msg)
            continue

        played_ids = _extract_positive_ids_from_history(turns)

        # 解析 session 元数据
        user_profile = row.get("user_profile")
        if isinstance(user_profile, (list, np.ndarray)):
            user_profile = user_profile[0] if len(user_profile) > 0 else {}
        if not isinstance(user_profile, dict):
            try:    user_profile = json.loads(str(user_profile))
            except: user_profile = {}

        conv_goal = row.get("conversation_goal")
        if isinstance(conv_goal, (list, np.ndarray)):
            conv_goal = conv_goal[0] if len(conv_goal) > 0 else {}
        if not isinstance(conv_goal, dict):
            try:    conv_goal = json.loads(str(conv_goal))
            except: conv_goal = {}

        listener_goal = str(conv_goal.get("listener_goal") or "").strip()
        user_culture  = str(user_profile.get("preferred_musical_culture") or "").strip()

        # ── Recommend（MusicRecPipelineSQL.recommend，含 SQL 过滤）──────────
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
            (tqdm.write if _tqdm_available else print)(msg)
            ranked, intent = [], None

        # 去重
        seen: set           = set()
        predicted_ids: List[str] = []
        for st in ranked:
            if st.track_id not in seen:
                seen.add(st.track_id)
                predicted_ids.append(st.track_id)

        # ── 生成自然语言回复（SQL 增强版，含排除约束注入）──────────────────
        if generate_response and ranked:
            sql_intent = intent if isinstance(intent, ParsedIntentSQL) else None
            response_text = _generate_response_sql(
                response_history, ranked,
                intent=sql_intent,
                api_key=pipeline._api_key,
            )
        else:
            response_text = "Here are some songs you might enjoy."

        predictions.append({
            "session_id":          session_id,
            "user_id":             user_id,
            "turn_number":         turn_number,
            "predicted_track_ids": predicted_ids,
            "predicted_response":  response_text,
        })

        if verbose:
            detail = (f"  → {len(predicted_ids)} tracks  "
                      f"response={response_text[:55]!r}...")
            (tqdm.write if _tqdm_available else print)(detail)

    # 原子写入
    tmp = output_path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    tmp.replace(output_path)

    print(f"\n[DONE] {len(predictions)} predictions saved to {output_path}")
    return predictions


# ─────────────────────────────────────────────────────────────────────────────
# 5. CLI  （兼容 baseline 所有参数）
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="TalkPlay Music Rec — SQL-Enhanced Baseline"
    )
    parser.add_argument(
        "--data-root", type=Path, default=DEFAULT_DATA_ROOT,
        help="Root directory containing the five data folders",
    )
    parser.add_argument(
        "--cache-file", type=Path, default=DEFAULT_CACHE_FILE,
        help="Path to intent JSON cache file",
    )
    parser.add_argument("--demo",            action="store_true")
    parser.add_argument("--eval",            action="store_true")
    parser.add_argument("--eval-k",          type=int, default=10)
    parser.add_argument("--max-sessions",    type=int, default=100)
    parser.add_argument("--query",           type=str, default=None)
    parser.add_argument("--user-id",         type=str, default=None)
    parser.add_argument("--predict-blind",   action="store_true")
    parser.add_argument("--predict-top-k",   type=int, default=20)
    parser.add_argument(
        "--predict-output", type=Path,
        default=Path("./predictions/blind_a_predictions_sql.json"),
    )
    parser.add_argument("--no-response",     action="store_true")
    parser.add_argument("--train-reranker",  action="store_true")
    parser.add_argument("--reranker-sessions", type=int, default=2000)
    parser.add_argument("--reranker-pool",   type=int, default=100)
    # SQL 专属调试参数
    parser.add_argument(
        "--sql-debug", action="store_true",
        help="Print SQL filter debug info for each query turn",
    )
    args = parser.parse_args()

    if args.demo:
        _base._demo_synthetic()
        return

    pipeline = MusicRecPipelineSQL.from_local(
        data_root=args.data_root,
        cache_file=args.cache_file,
        deepseek_api_key=DEEPSEEK_API_KEY,
    )

    if args.train_reranker:
        dfs = load_all_data(args.data_root)
        if "sessions" not in dfs:
            print("[ERROR] sessions data not found; cannot train reranker.")
            return
        pipeline.train_reranker(
            dfs["sessions"],
            max_sessions=args.reranker_sessions,
            candidate_pool=args.reranker_pool,
        )
        print("[train-reranker] Done.")
        return

    if args.query:
        uid  = args.user_id or "unknown-user"
        conv = [{"role": "user", "content": args.query}]
        ranked, intent = pipeline.recommend(uid, conv, verbose=True)
        if args.sql_debug:
            dbg = pipeline.sql_filter.debug_query(intent)
            print("\n[SQL DEBUG]", json.dumps(dbg, ensure_ascii=False, indent=2))
        print(pipeline.format_results(ranked))
        # 生成自然语言回复（SQL 增强版，含排除约束注入）
        if ranked and not args.no_response:
            sql_intent = intent if isinstance(intent, ParsedIntentSQL) else None
            response_text = _generate_response_sql(
                conv, ranked, intent=sql_intent, api_key=DEEPSEEK_API_KEY
            )
            print(f"\n[Response]\n{response_text}")

    if args.eval:
        dfs = load_all_data(args.data_root)
        if "sessions" not in dfs:
            print("[ERROR] sessions data not found.")
            return
        sess_df = dfs["sessions"]
        if "split" in sess_df.columns:
            test_df = sess_df[sess_df["split"].str.startswith("test")]
        else:
            test_df = sess_df[
                sess_df["user_profile"].apply(
                    lambda x: "test" in str(x.get("user_split", ""))
                    if isinstance(x, dict) else False
                )
            ]
        print(f"Evaluating on {len(test_df)} test sessions …")
        metrics = evaluate(pipeline, test_df, k=args.eval_k,
                           max_sessions=args.max_sessions)
        print(json.dumps(metrics, indent=2))

    if args.predict_blind:
        dfs = load_all_data(args.data_root)
        if "blind" not in dfs:
            print("[ERROR] Challenge-Blind-A not found under", args.data_root)
            return
        blind_df = dfs["blind"]
        print(f"Generating SQL-enhanced predictions for {len(blind_df)} blind sessions …")
        predict_blind_a_sql(
            pipeline=pipeline,
            blind_df=blind_df,
            output_path=args.predict_output,
            top_k=args.predict_top_k,
            generate_response=not args.no_response,
            verbose=True,
        )


if __name__ == "__main__":
    main()