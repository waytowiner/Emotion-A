"""信号层：从情绪分矩阵生成板块切换信号。规则透明，非预测黑盒。"""

from __future__ import annotations

import pandas as pd


def _consecutive_trend(s: pd.Series, days: int, direction: str) -> bool:
    """序列尾部是否连续 days 天单调上升/下降（NaN 视为中断）。"""
    tail = s.dropna().tail(days + 1)
    if len(tail) < days + 1:
        return False
    diff = tail.diff().dropna()
    if direction == "up":
        return bool((diff > 0).all())
    return bool((diff < 0).all())


def detect_signals(smoothed: pd.DataFrame, pct: pd.DataFrame,
                   *, consecutive_days: int = 3, hot_quantile: float = 0.9,
                   cold_quantile: float = 0.1, recent_top_days: int = 20,
                   recent_top_k: int = 10) -> dict[str, list[str]]:
    """生成信号。

    返回:
      heating   升温榜：情绪分连续 N 日上升
      cooling   降温榜：情绪分连续 N 日下降
      hot       亢奋：时序分位 > hot_quantile（不宜追高）
      cold      冰点：时序分位 < cold_quantile（观察反转机会）
      switching 切换预警：近期曾居前 K 的板块正在降温 且 有低位板块升温
    """
    if smoothed.empty:
        return {"heating": [], "cooling": [], "hot": [], "cold": [], "switching": []}

    last_pct = pct.iloc[-1]
    heating, cooling = [], []
    for board in smoothed.columns:
        s = smoothed[board]
        if _consecutive_trend(s, consecutive_days, "up"):
            heating.append(board)
        if _consecutive_trend(s, consecutive_days, "down"):
            cooling.append(board)

    hot = last_pct[last_pct > hot_quantile].index.tolist()
    cold = last_pct[last_pct < cold_quantile].index.tolist()

    # 近期（recent_top_days 内任一交易日）情绪分曾进入全市场前 K
    recent = smoothed.tail(recent_top_days)
    was_top = set()
    for _, row in recent.iterrows():
        was_top.update(row.dropna().nlargest(recent_top_k).index)
    former_leaders_cooling = sorted(was_top & set(cooling))
    low_heating = sorted(
        b for b in heating
        if pd.notna(last_pct.get(b)) and last_pct[b] < 0.5
    )
    switching = []
    if former_leaders_cooling and low_heating:
        switching = [f"{b}↓ → {','.join(low_heating[:5])}↑" for b in former_leaders_cooling]

    def _by_score(boards: list[str]) -> list[str]:
        last = smoothed.iloc[-1]
        return sorted(boards, key=lambda b: last.get(b, float("nan")),
                      reverse=True)

    return {
        "heating": _by_score(heating),
        "cooling": _by_score(cooling),
        "hot": _by_score(hot),
        "cold": _by_score(cold),
        "switching": switching,
    }
