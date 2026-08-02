"""板块龙头股计算：给没有对应 ETF 的板块提供可操作的个股标的。

口径：
- 流动性龙头：板块内近 5 个交易日日均成交额前 N 的成分股
  （剔除停牌 trade_status != 1 的记录）。
- 情绪龙头：近 10 个交易日内有涨停记录（复用 factors.flag_limit_up 口径）
  且窗口内最高连续涨停天数（连板数）最高的股票；无涨停记录则为空。

纯 pandas、无网络、无 LLM。输入为 pipeline.prepare 的长表。
"""

from __future__ import annotations

import pandas as pd

from .factors import flag_limit_up


def _last_dates(df: pd.DataFrame, n: int) -> list:
    """全表最近 n 个交易日。"""
    return sorted(df["date"].unique())[-n:]


def _latest_names(df: pd.DataFrame) -> dict:
    """每只股票最近的名称。"""
    d = df.sort_values("date").groupby("code")["name"].last()
    return d.to_dict()


def liquidity_leaders(stock_daily: pd.DataFrame, board_map: pd.DataFrame,
                      *, window: int = 5, top_n: int = 3) -> dict[str, list[dict]]:
    """流动性龙头：近 window 日日均成交额前 top_n。

    返回 {board_code: [{code, name, avg_amount}, ...]}（按成交额降序）。
    """
    dates = _last_dates(stock_daily, window)
    d = stock_daily[stock_daily["date"].isin(dates)]
    d = d[d["trade_status"].astype(str) == "1"]  # 剔除停牌
    m = board_map[["code", "board_code"]].drop_duplicates("code")
    d = d.merge(m, on="code", how="inner")
    d["amount"] = pd.to_numeric(d["amount"], errors="coerce")
    g = d.groupby(["board_code", "code"])["amount"].mean().dropna().reset_index()

    names = _latest_names(stock_daily)
    out: dict[str, list[dict]] = {}
    for board, sub in g.groupby("board_code"):
        top = sub.nlargest(top_n, "amount")
        out[board] = [{
            "code": r.code,
            "name": names.get(r.code, r.code),
            "avg_amount": float(r.amount),
        } for r in top.itertuples()]
    return out


def _max_streak(flags: pd.Series) -> int:
    """布尔序列（按日期升序）中最长连续 True 段长度。"""
    best = cur = 0
    for v in flags:
        if v:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def momentum_leaders(stock_daily: pd.DataFrame, board_map: pd.DataFrame,
                     *, window: int = 10, top_n: int = 2) -> dict[str, list[dict]]:
    """情绪龙头：近 window 日内有涨停且最高连板数最高的前 top_n 只。

    返回 {board_code: [{code, name, max_streak}, ...]}（按连板数降序，
    同数时最近一次涨停更近的优先）。无涨停记录的板块不在返回中。
    """
    dates = _last_dates(stock_daily, window)
    flagged = flag_limit_up(stock_daily[stock_daily["date"].isin(dates)].copy())
    m = board_map[["code", "board_code"]].drop_duplicates("code")
    flagged = flagged.merge(m, on="code", how="inner")

    rows = []
    for (board, code), sub in flagged.groupby(["board_code", "code"]):
        sub = sub.sort_values("date")
        streak = _max_streak(sub["is_limit_up"].tolist())
        if streak == 0:
            continue
        last_lu = sub.loc[sub["is_limit_up"], "date"].max()
        rows.append({"board_code": board, "code": code, "max_streak": streak,
                     "last_limit_up": last_lu})

    names = _latest_names(stock_daily)
    out: dict[str, list[dict]] = {}
    df = pd.DataFrame(rows)
    if df.empty:
        return out
    for board, sub in df.groupby("board_code"):
        top = sub.sort_values(["max_streak", "last_limit_up"],
                              ascending=[False, False]).head(top_n)
        out[board] = [{
            "code": r.code,
            "name": names.get(r.code, r.code),
            "max_streak": int(r.max_streak),
        } for r in top.itertuples()]
    return out


def compute_leaders(stock_daily: pd.DataFrame, board_map: pd.DataFrame,
                    *, liquidity_window: int = 5, momentum_window: int = 10,
                    liquidity_top_n: int = 3, momentum_top_n: int = 2
                    ) -> dict[str, dict]:
    """一键计算全部板块龙头股。

    返回 {board_code: {"liquidity": [...], "momentum": [...]}}，
    覆盖 board_map 中出现的全部板块（无数据则为空列表）。
    """
    liq = liquidity_leaders(stock_daily, board_map,
                            window=liquidity_window, top_n=liquidity_top_n)
    mom = momentum_leaders(stock_daily, board_map,
                           window=momentum_window, top_n=momentum_top_n)
    boards = board_map["board_code"].unique()
    return {b: {"liquidity": liq.get(b, []), "momentum": mom.get(b, [])}
            for b in boards}
