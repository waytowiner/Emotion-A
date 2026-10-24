"""政策（时政）支柱：新闻联播文本 -> 板块政策热度 -> 双因交叉信号。

与情绪支柱完全平行、完全解耦：
- 不进入情绪合成（不改 FACTOR_NAMES、不进 combine_factors）；
- 不修改 config.yaml（词典独立于其 SHA256 冻结）；
- 观察模式纪律：全部输出仅描述状态，不生成买卖建议。

口径约定：
- 数据源为央视新闻联播文字稿（emotion.data.news_cctv 缓存）；
- 映射为纯关键词字符串匹配（config/policy_keywords.yaml）；
- 一条新闻命中某板块任意关键词，该板块当日"命中条数"+1
  （按新闻条数去重，不按关键词出现次数累计）——即提及强度而非利好方向；
- 非交易日（周末/假期）的命中累计并入下一个交易日（周末联播的
  政策信息在下一交易日才可被市场消化，避免天然丢数）。
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import yaml

from .score import percentile_rank, smooth_score, zscore_cross_section

POLICY_FACTOR_NAMES = ("policy_intensity", "policy_breadth", "policy_heat")

# 双因状态（基于情绪分位 × 政策分位的四象限 + 中间带）
DUAL_STATE_CN = {
    "resonance": "政策共振（情绪热+政策热）",
    "lurking": "政策潜伏（情绪冷+政策热）",
    "funds_first": "资金先行（情绪热+政策冷）",
    "dual_cold": "双冷（情绪冷+政策冷）",
    "neutral": "中性",
}

ROOT = Path(__file__).resolve().parents[1]


def load_keyword_map(path: str | Path | None = None) -> dict[str, list[str]]:
    """读取关键词词典 -> {board_code: [关键词, ...]}。

    词条 terms 为空或缺失的板块被剔除；不做任何模糊归一化。
    """
    p = Path(path) if path else ROOT / "config" / "policy_keywords.yaml"
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    out: dict[str, list[str]] = {}
    for code, entry in (data.get("keywords") or {}).items():
        terms = [str(t) for t in (entry.get("terms") or []) if str(t)]
        if terms:
            out[str(code)] = terms
    return out


def align_hits_to_calendar(hit: pd.DataFrame,
                           trading_calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """把命中矩阵对齐到 A 股交易日：非交易日命中并入下一交易日。

    联播每天都播（含周末），但情绪分只存在于交易日；直接按日期
    内连接会丢掉周末命中，分位窗口口径也会与情绪侧错位。约定：
    周末/假期的命中累计到下一个交易日（信息在下一个交易日才可交易）。
    交易日无任何命中记 0（不 NaN），缺失的日期范围不外插。
    """
    if hit is None or hit.empty:
        return pd.DataFrame()
    cal = pd.DatetimeIndex(trading_calendar)
    # 有效交易日窗口：日历 ∩ [命中首日, 命中末日]，不外插；
    # 但窗口前的交易日不参与滚动并入（其无对应命中期）。
    lo, hi = hit.index.min(), hit.index.max()
    cal_before = cal[cal < lo]
    cal_in = cal[(cal >= lo) & (cal <= hi)]
    out = hit.reindex(cal_in).fillna(0.0)
    # 非交易日（周末/假期）命中并入下一个交易日
    extra = hit.loc[~hit.index.isin(cal_in)]
    if not extra.empty:
        for dt, row in extra.iterrows():
            nxt = cal_in[cal_in >= dt]
            if len(nxt):
                out.loc[nxt[0]] = out.loc[nxt[0]].add(row, fill_value=0.0)
    # 窗口前的交易日保留为 0 行（供调用方决定是否截用）
    if len(cal_before):
        zero = pd.DataFrame(0.0, index=cal_before, columns=hit.columns)
        out = pd.concat([zero, out])
    return out


def cal_dates(hit: pd.DataFrame) -> pd.DatetimeIndex:
    """辅助：返回命中矩阵自身的日期索引（保留供测试与诊断使用）。"""
    return pd.DatetimeIndex(hit.index)


def map_news_to_boards(news: pd.DataFrame, keyword_map: dict[str, list[str]],
                       board_names: dict[str, str] | None = None
                       ) -> pd.DataFrame:
    """逐日逐条新闻匹配关键词，返回 date × board 的命中条数矩阵。

    只保留板块池内代码（board_names 传入时按其键过滤）。
    一条新闻命中某板块任意关键词计 1 条；同日多条新闻逐条累计。
    匹配用预编译正则（多关键词 alternation），纯文本包含判定。
    """
    if news is None or news.empty or not keyword_map:
        return pd.DataFrame()
    d = news.dropna(subset=["date", "title"]).copy()
    if "content" in d.columns:
        d["text"] = d["title"].astype(str) + " " + d["content"].astype(str)
    else:
        d["text"] = d["title"].astype(str)

    valid_boards = set(board_names) if board_names else set(keyword_map)
    rx_by_board: dict[str, re.Pattern] = {}
    for code, terms in keyword_map.items():
        if code not in valid_boards or not terms:
            continue
        rx_by_board[code] = re.compile("|".join(re.escape(t) for t in terms))
    if not rx_by_board:
        return pd.DataFrame()

    # 逐条新闻：命中的板块集合 -> (date, board) 计数
    counts: dict[tuple, int] = {}
    boards = list(rx_by_board)
    texts = d["text"].tolist()
    dates = pd.to_datetime(d["date"]).tolist()
    for text, dt in zip(texts, dates):
        for code, rx in rx_by_board.items():
            if rx.search(text):
                key = (dt, code)
                counts[key] = counts.get(key, 0) + 1
    if not counts:
        # 全程无命中：返回 0 矩阵（保持日期/板块结构）
        idx = pd.DatetimeIndex(sorted(set(dates)))
        return pd.DataFrame(0.0, index=idx, columns=boards)
    s = pd.Series(counts)
    s.index = pd.MultiIndex.from_tuples(s.index, names=["date", "board_code"])
    mat = s.unstack("board_code")
    mat = mat.reindex(columns=boards).fillna(0.0)
    # 补上新闻存在但完全无命中的日期，保持连续
    mat = mat.reindex(pd.DatetimeIndex(sorted(set(dates) | set(mat.index))),
                      fill_value=0.0)
    mat.index.name = "date"
    return mat.sort_index()


def compute_policy_factors(hit: pd.DataFrame, *,
                           heat_window: int = 5,
                           breadth_window: int = 10) -> dict[str, pd.DataFrame]:
    """政策三口径（输入为 map_news_to_boards/align 后的命中矩阵）。

    - policy_intensity：当日命中条数（原始提及强度）
    - policy_breadth：近 breadth_window 日内出现过命中的天数占比（0~1）
    - policy_heat：heat_window 日滚动命中条数（强度平滑）
    """
    if hit is None or hit.empty:
        return {k: pd.DataFrame() for k in POLICY_FACTOR_NAMES}
    intensity = hit.fillna(0.0).astype(float)
    breadth = (hit > 0).rolling(breadth_window, min_periods=1).mean()
    heat = intensity.rolling(heat_window, min_periods=1).sum()
    return {
        "policy_intensity": intensity,
        "policy_breadth": breadth,
        "policy_heat": heat,
    }


def build_policy_scores(hit: pd.DataFrame, *, smooth_window: int = 5,
                        lookback: int = 120
                        ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """政策分合成：三口径横截面 z（±3 截断）-> 等权逐格均值 -> 平滑 -> 分位。

    与情绪侧 build_scores 平行：同平滑窗口、同分位 lookback，口径可平行解读。
    命中为 0 的板块 z 记为全市场命中均值的相对位置（横截面内标准化，
    全员 0 的日期 std=0 -> 整行 NaN，不参与分位）。
    返回 (日度原始分, 平滑分, 时序分位)。
    """
    if hit is None or hit.empty:
        empty = pd.DataFrame()
        return empty, empty.copy(), empty.copy()
    factors = compute_policy_factors(hit)
    zs = {k: zscore_cross_section(v).clip(-3, 3) for k, v in factors.items()
          if v is not None and not v.empty}
    if not zs:
        empty = pd.DataFrame()
        return empty, empty.copy(), empty.copy()
    idx = sorted(set().union(*[z.index for z in zs.values()]))
    cols = sorted(set().union(*[z.columns for z in zs.values()]))
    # 三口径等权：堆叠后按日期分组取逐格均值（mean 跳过 NaN，
    # 与情绪侧"只聚合实际存在的因子"同思路）
    stacked = pd.concat([z.reindex(index=idx, columns=cols) for z in zs.values()])
    raw = stacked.groupby(level=0).mean()
    raw.index = pd.DatetimeIndex(raw.index)
    smoothed = smooth_score(raw, smooth_window)
    smoothed = smoothed.loc[raw.dropna(how="all").index]
    pct = percentile_rank(smoothed, lookback=lookback)
    return raw, smoothed, pct


def dual_factor_states(emotion_pct: pd.DataFrame, policy_pct: pd.DataFrame, *,
                       hot_q: float = 0.8, cold_q: float = 0.2
                       ) -> pd.DataFrame:
    """双因状态划分：每板块基于情绪/政策各自的时序分位归入五状态之一。

    - resonance   情绪热+政策热：题材获资金与政策双重加持，观察重点
    - lurking     情绪冷+政策热：政策已至、盘面未动，政策驱动型观察
    - funds_first 情绪热+政策冷：纯资金/情绪驱动，无政策背书
    - dual_cold   情绪冷+政策冷
    - neutral     其余中间带（含任一侧分位缺失）
    返回 date × board 的状态字符串矩阵。
    """
    if (policy_pct is None or policy_pct.empty
            or emotion_pct is None or emotion_pct.empty):
        return pd.DataFrame()
    cols = sorted(set(emotion_pct.columns) & set(policy_pct.columns))
    if not cols:
        return pd.DataFrame()
    idx = emotion_pct.index.intersection(policy_pct.index)
    e = emotion_pct.reindex(index=idx, columns=cols)
    p = policy_pct.reindex(index=idx, columns=cols)

    def _state(er: float, pr: float) -> str:
        if pd.isna(er) or pd.isna(pr):
            return "neutral"
        e_hot, e_cold = er >= hot_q, er <= cold_q
        p_hot, p_cold = pr >= hot_q, pr <= cold_q
        if e_hot and p_hot:
            return "resonance"
        if e_cold and p_hot:
            return "lurking"
        if e_hot and p_cold:
            return "funds_first"
        if e_cold and p_cold:
            return "dual_cold"
        return "neutral"

    out = pd.DataFrame(index=idx, columns=cols, dtype=object)
    for c in cols:
        out[c] = [_state(er, pr) for er, pr in zip(e[c].tolist(), p[c].tolist())]
    return out


def detect_policy_signals(emotion_pct: pd.DataFrame,
                          policy_pct: pd.DataFrame, *,
                          hot_q: float = 0.8, cold_q: float = 0.2
                          ) -> dict[str, list[str]]:
    """双因交叉信号（观察口径）：返回最后一日的板块代码列表。

    - policy_resonance：情绪分位 >= hot_q 且政策分位 >= hot_q
    - policy_lurking：情绪分位 <= cold_q 且政策分位 >= hot_q
    与 detect_signals 输出结构同型；独立函数，不改其签名。
    """
    out: dict[str, list[str]] = {"policy_resonance": [], "policy_lurking": []}
    if (policy_pct is None or policy_pct.empty
            or emotion_pct is None or emotion_pct.empty):
        return out
    last_e = emotion_pct.iloc[-1]
    last_p = policy_pct.iloc[-1]
    cols = last_e.index.intersection(last_p.index)
    for c in cols:
        er, pr = last_e.get(c), last_p.get(c)
        if pd.isna(er) or pd.isna(pr):
            continue
        if er >= hot_q and pr >= hot_q:
            out["policy_resonance"].append(c)
        elif er <= cold_q and pr >= hot_q:
            out["policy_lurking"].append(c)
    # 按政策分降序展示
    out["policy_resonance"] = sorted(out["policy_resonance"],
                                     key=lambda c: last_p[c], reverse=True)
    out["policy_lurking"] = sorted(out["policy_lurking"],
                                   key=lambda c: last_p[c], reverse=True)
    return out
