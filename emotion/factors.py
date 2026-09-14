"""板块情绪因子层。

输入均为"长表"DataFrame，输出统一的 date × board 因子矩阵（pivot）。
全部纯计算、无网络、无 LLM，可复现。

口径约定（与 README 一致）：
- 涨停判定：当日最高价涨幅触及涨停阈值 且 收盘价封板（close >= high*0.998）
- 涨停阈值：创业板(30)/科创板(68) 19.8%，其余 9.8%；ST 股 4.8%（按名称含 ST）
- 炸板：最高涨幅触板但未封住
- 北交所不在 baostock 数据内，相关个股自动缺席（占比很小）
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SEAL_TOL = 0.002  # 封板容忍：收盘距最高价 0.2% 以内视为封板
FACTOR_NAMES = (
    "limit_up_strength", "money_effect", "breadth", "rps",
    "turnover_heat", "promotion_rate",
)


def limit_threshold(code: str, name: str = "", *, is_st: bool | None = None) -> float:
    """个股涨停阈值（%）。

    ST 仅对主板生效（5%）；创业板/科创板 ST 涨跌停仍为 20%（与正常股相同）。
    is_st 显式传入时优先于名称判定，用于按日 isST 字段还原历史 ST 状态。
    """
    if is_st is None:
        is_st = "ST" in str(name).upper()
    if str(code).startswith(("30", "68")):
        return 19.8
    if is_st:
        return 4.8
    return 9.8


def flag_limit_up(stock_daily: pd.DataFrame) -> pd.DataFrame:
    """给个股日线打标记：is_limit_up / is_touch（炸板）/ high_pct。

    输入需含 [date, code, close, high, pct_chg]，可选 [name, preclose, isST]。
    - 前收优先用 preclose 字段（baostock 原值，精确）；缺省时用 pct_chg 反推
      （baostock pctChg 两位小数，阈值边缘有舍入误差）。
    - ST 优先用按日 isST 字段（还原历史戴帽/摘帽）；缺省回退名称含 ST。
    """
    df = stock_daily.copy()
    pct = pd.to_numeric(df["pct_chg"], errors="coerce")
    if "preclose" in df.columns:
        pc = pd.to_numeric(df["preclose"], errors="coerce")
        derived = df["close"] / (1.0 + pct / 100.0)
        prev_close = pc.where(pc.notna() & (pc > 0), derived)
    else:
        prev_close = df["close"] / (1.0 + pct / 100.0)
    df["prev_close"] = prev_close
    df["high_pct"] = (df["high"] / prev_close - 1.0) * 100.0

    if "isST" in df.columns:
        is_st = pd.to_numeric(df["isST"], errors="coerce").fillna(0).astype(int) == 1
    elif "name" in df.columns:
        is_st = df["name"].astype(str).str.upper().str.contains("ST", na=False)
    else:
        is_st = pd.Series(False, index=df.index)

    code = df["code"].astype(str)
    chinext_star = code.str.startswith(("30", "68"))
    thr = np.where(chinext_star, 19.8, np.where(is_st, 4.8, 9.8))
    sealed = df["close"] >= df["high"] * (1 - SEAL_TOL)
    hit = df["high_pct"].to_numpy() >= thr
    df["is_limit_up"] = hit & sealed.to_numpy()
    df["is_touch"] = hit & ~sealed.to_numpy()
    return df


def _with_board(df: pd.DataFrame, board_map: pd.DataFrame) -> pd.DataFrame:
    # 一股多板：东财行业口径下个股同时属于一/二/三级板块（如某芯片股同属
    # 电子、半导体、数字芯片设计）。只按 (code, board_code) 对去重，不按 code
    # 去重——否则每只股票只剩行序最靠前的一个板块，三级细分板块会整板缺失
    # （曾导致 35 个细分板块无任何个股因子、情绪分全 NaN）。
    m = board_map[["code", "board_code"]].drop_duplicates()
    return df.merge(m, on="code", how="inner")


def factor_limit_up_strength(flagged: pd.DataFrame, board_map: pd.DataFrame) -> pd.DataFrame:
    """涨停强度 = 板块内涨停家数 / 板块内当日交易家数（剔除停牌，与宽度同口径）。"""
    d = _with_board(flagged, board_map)
    d = d[d["trade_status"].astype(str) == "1"]
    lu = d.groupby(["date", "board_code"])["is_limit_up"].sum()
    n = d.groupby(["date", "board_code"])["code"].count()
    out = (lu / n.replace(0, np.nan)).rename("limit_up_strength").reset_index()
    return out.pivot(index="date", columns="board_code", values="limit_up_strength")


def factor_money_effect(flagged: pd.DataFrame, board_map: pd.DataFrame) -> pd.DataFrame:
    """赚钱效应 = 昨日涨停股今日涨幅的中位数（按板块）。

    以中位数而非均值：少数大涨会拉高均值，中位数更接近"多数人的体感"。
    """
    d = _with_board(flagged, board_map)
    d = d.sort_values(["code", "date"])
    d["prev_limit_up"] = d.groupby("code")["is_limit_up"].shift(1, fill_value=False)
    hit = d[d["prev_limit_up"] & (d["trade_status"].astype(str) == "1")]
    out = hit.groupby(["date", "board_code"])["pct_chg"].median().rename("money_effect")
    return out.reset_index().pivot(index="date", columns="board_code", values="money_effect")


def factor_breadth(flagged: pd.DataFrame, board_map: pd.DataFrame) -> pd.DataFrame:
    """涨跌宽度 = (上涨家数 - 下跌家数) / 交易家数。"""
    d = _with_board(flagged, board_map)
    d = d[d["trade_status"].astype(str) == "1"]
    d["up"] = d["pct_chg"] > 0
    d["down"] = d["pct_chg"] < 0
    g = d.groupby(["date", "board_code"]).agg(up=("up", "sum"), down=("down", "sum"),
                                              n=("code", "count"))
    g["breadth"] = (g["up"] - g["down"]) / g["n"].replace(0, np.nan)
    return g["breadth"].reset_index().pivot(index="date", columns="board_code", values="breadth")


def factor_rps(board_kline: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """相对强度：板块 window 日涨幅在全板块中的分位（0~1）。"""
    k = board_kline.sort_values(["board_code", "date"])
    k["ret"] = k.groupby("board_code")["close"].pct_change(window)
    p = k.pivot(index="date", columns="board_code", values="ret")
    return p.rank(axis=1, pct=True)


def factor_turnover_heat(flagged: pd.DataFrame, board_map: pd.DataFrame,
                         window: int = 5, market_total: pd.Series | None = None) -> pd.DataFrame:
    """热度变化 = 板块成交额占全市场比例的 window 日变化量（百分点）。

    market_total：全市场日成交额（date 索引的 Series），由 pipeline 在板块
    归属过滤前计算；缺省时退化为板块池内成交额合计（旧口径）。
    """
    d = _with_board(flagged, board_map)
    amt = d.groupby(["date", "board_code"])["amount"].sum()
    total = market_total if market_total is not None else amt.groupby("date").sum()
    # 用 level 显式广播：amt 为 (date, board_code) MultiIndex Series，total 为 date 索引
    # Series。直接相除依赖 total 索引名恰为 "date"（pandas 2.x 行为），任意一方索引
    # 无名即抛 ValueError；改用 div(level="date") 对无名索引同样健壮。
    share = amt.div(total, level="date").rename("share").reset_index()
    p = share.pivot(index="date", columns="board_code", values="share")
    return p.diff(window) * 100.0


def factor_promotion_rate(flagged: pd.DataFrame, board_map: pd.DataFrame) -> pd.DataFrame:
    """晋级率 = 昨日涨停股今日继续涨停的比例（按板块）。

    衡量资金对板块的承接意愿；样本不足 2 家的板块当日记 NaN。
    """
    d = _with_board(flagged, board_map)
    d = d.sort_values(["code", "date"])
    d["prev_limit_up"] = d.groupby("code")["is_limit_up"].shift(1, fill_value=False)
    cand = d[d["prev_limit_up"]]
    # 与赚钱效应同口径：今日停牌股不计入（既不能继续涨停也不进分母）
    cand = cand[cand["trade_status"].astype(str) == "1"]
    g = cand.groupby(["date", "board_code"]).agg(
        promoted=("is_limit_up", "sum"), total=("code", "count"))
    g["promotion_rate"] = np.where(g["total"] >= 2, g["promoted"] / g["total"], np.nan)
    return g["promotion_rate"].reset_index().pivot(index="date", columns="board_code",
                                                   values="promotion_rate")


def compute_all_factors(stock_daily: pd.DataFrame, board_map: pd.DataFrame,
                        board_kline: pd.DataFrame,
                        *, market_total: pd.Series | None = None) -> dict[str, pd.DataFrame]:
    """一键计算全部因子，返回 {因子名: date×board 矩阵}。

    market_total：全市场日成交额（可选，pipeline.prepare 提供），
    作为热度变化因子的真实全市场分母。
    """
    flagged = flag_limit_up(stock_daily)
    return {
        "limit_up_strength": factor_limit_up_strength(flagged, board_map),
        "money_effect": factor_money_effect(flagged, board_map),
        "breadth": factor_breadth(flagged, board_map),
        "rps": factor_rps(board_kline),
        "turnover_heat": factor_turnover_heat(flagged, board_map,
                                              market_total=market_total),
        "promotion_rate": factor_promotion_rate(flagged, board_map),
    }
