"""板块标的池治理：成分数门槛与完全重复板块去重。

东财细分板块中既有仅 1--几只成分股的极小截面，也有成分完全一致、仅名称层级
不同的重复板块。把它们与大板块同权用于横截面打分，会同时放大噪声和重复暴露。
本模块只处理当前可观测的成分快照；历史成分变更并不在缓存中，因此不会宣称已
消除幸存者偏差。
"""

from __future__ import annotations

import re

import pandas as pd


_TIER_SUFFIX = re.compile(r"([ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+)$")
_TIER_ORDER = {ch: i for i, ch in enumerate("ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ", 1)}


def constituent_counts(board_map: pd.DataFrame) -> pd.Series:
    """按 ``(board_code, code)`` 去重后的板块成分股数。"""
    required = {"board_code", "code"}
    missing = required - set(board_map.columns)
    if missing:
        raise ValueError(f"board_map 缺少列：{sorted(missing)}")
    return (board_map.dropna(subset=["board_code", "code"])
            .drop_duplicates(["board_code", "code"])
            .groupby("board_code")["code"].nunique().sort_index())


def _tier_rank(name: str) -> int:
    """同成分的 Ⅱ/Ⅲ 等层级优先保留更上层；无层级后缀优先级最高。"""
    match = _TIER_SUFFIX.search(str(name))
    if not match:
        return 0
    return _TIER_ORDER.get(match.group(1)[0], 99)


def select_board_universe(board_map: pd.DataFrame, boards: pd.DataFrame, *,
                          min_constituents: int = 1,
                          duplicate_jaccard: float = 1.0,
                          enabled: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """选择用于打分和回测的板块池，并返回可序列化的审计信息。

    ``duplicate_jaccard=1.0`` 只合并成分完全一致的板块，是不改变经济暴露的
    保守默认值；若要合并近似概念板块，应经过单独研究后再显式放宽阈值。
    """
    if min_constituents < 1:
        raise ValueError("min_constituents 必须 >= 1")
    if not 0 < duplicate_jaccard <= 1:
        raise ValueError("duplicate_jaccard 必须在 (0, 1] 内")
    if not {"board_code", "board_name"}.issubset(boards.columns):
        raise ValueError("boards 需包含 board_code 和 board_name")

    raw_codes = pd.Index(boards["board_code"].dropna().astype(str).drop_duplicates())
    named = boards.drop_duplicates("board_code").set_index("board_code")["board_name"]
    counts = constituent_counts(board_map).reindex(raw_codes, fill_value=0)
    raw_members = board_map.drop_duplicates(["board_code", "code"])

    if not enabled:
        diag = {
            "enabled": False,
            "raw_board_count": int(len(raw_codes)),
            "eligible_board_count": int(len(raw_codes)),
            "raw_constituent_pairs": int(len(raw_members)),
            "min_constituents": min_constituents,
            "duplicate_jaccard": duplicate_jaccard,
            "dropped_small": [],
            "dropped_duplicate": [],
            "historical_constituents": "current_snapshot_only",
        }
        return board_map.copy(), boards.copy(), diag

    eligible = counts[counts >= min_constituents].index.tolist()
    dropped_small = [
        {"code": code, "name": str(named.get(code, code)), "constituents": int(counts[code])}
        for code in raw_codes if code not in set(eligible)
    ]

    # N≈100，平方枚举比引入近似索引更透明。以并查集处理可能的传递重复组。
    members = {
        code: set(raw_members.loc[raw_members["board_code"].astype(str) == code, "code"].astype(str))
        for code in eligible
    }
    parent = {code: code for code in eligible}

    def find(code: str) -> str:
        while parent[code] != code:
            parent[code] = parent[parent[code]]
            code = parent[code]
        return code

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    pairs: list[tuple[str, str, float]] = []
    for i, left in enumerate(eligible):
        left_set = members[left]
        for right in eligible[i + 1:]:
            right_set = members[right]
            union_size = len(left_set | right_set)
            if not union_size:
                continue
            jaccard = len(left_set & right_set) / union_size
            if jaccard >= duplicate_jaccard:
                union(left, right)
                pairs.append((left, right, jaccard))

    groups: dict[str, list[str]] = {}
    for code in eligible:
        groups.setdefault(find(code), []).append(code)
    keep_codes: set[str] = set()
    dropped_duplicate: list[dict] = []
    pair_jaccard = {frozenset((a, b)): j for a, b, j in pairs}
    for codes in groups.values():
        canonical = min(
            codes,
            key=lambda code: (
                _tier_rank(str(named.get(code, code))),
                -int(counts[code]),
                str(named.get(code, code)),
                code,
            ),
        )
        keep_codes.add(canonical)
        for code in codes:
            if code == canonical:
                continue
            # 在传递组中，代表与该代码不一定直接相连，记录最保守的可知相似度。
            related = [j for pair, j in pair_jaccard.items()
                       if code in pair and canonical in pair]
            dropped_duplicate.append({
                "code": code,
                "name": str(named.get(code, code)),
                "kept_code": canonical,
                "kept_name": str(named.get(canonical, canonical)),
                "constituents": int(counts[code]),
                "jaccard": float(max(related) if related else duplicate_jaccard),
            })

    selected = [code for code in raw_codes if code in keep_codes]
    filtered_map = board_map[board_map["board_code"].astype(str).isin(selected)].copy()
    filtered_boards = boards[boards["board_code"].astype(str).isin(selected)].copy()
    diag = {
        "enabled": True,
        "raw_board_count": int(len(raw_codes)),
        "eligible_board_count": int(len(selected)),
        "raw_constituent_pairs": int(len(raw_members)),
        "eligible_constituent_pairs": int(
            len(filtered_map.drop_duplicates(["board_code", "code"]))),
        "min_constituents": min_constituents,
        "duplicate_jaccard": duplicate_jaccard,
        "dropped_small": dropped_small,
        "dropped_duplicate": dropped_duplicate,
        "historical_constituents": "current_snapshot_only",
    }
    return filtered_map, filtered_boards, diag
