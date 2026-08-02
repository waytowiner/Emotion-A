"""情绪分合成层：因子 z-score -> 加权合成 -> 平滑。

fund_flow 等可选因子缺失时，权重自动在可用因子间归一化。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def zscore_cross_section(factor: pd.DataFrame) -> pd.DataFrame:
    """横截面 z-score：每天在各板块间标准化。"""
    mean = factor.mean(axis=1)
    std = factor.std(axis=1).replace(0, np.nan)
    return factor.sub(mean, axis=0).div(std, axis=0)


def combine_factors(factors: dict[str, pd.DataFrame],
                    weights: dict[str, float],
                    min_coverage: float = 0.5) -> pd.DataFrame:
    """加权合成情绪分（原始日度）。

    逐单元格计算：只聚合该日期-板块上实际存在的因子（按权重归一），
    可用因子占比低于 min_coverage 的单元格不予评分。稀疏因子
    （如赚钱效应、晋级率）不会拖垮整行。
    """
    available = {k: v for k, v in factors.items() if v is not None and not v.empty}
    if not available:
        raise ValueError("no factors available")
    w = {k: weights.get(k, 1.0) for k in available}
    if sum(w.values()) <= 0:
        raise ValueError("sum of weights must be positive")

    zscores = {k: zscore_cross_section(v) for k, v in available.items()}
    # 对齐到统一索引/列
    idx = sorted(set().union(*[z.index for z in zscores.values()]))
    cols = sorted(set().union(*[z.columns for z in zscores.values()]))
    mats = np.stack([
        z.reindex(index=idx, columns=cols).to_numpy(dtype=float)
        for z in zscores.values()
    ])  # (n_factors, n_dates, n_boards)
    w_arr = np.array([w[k] for k in zscores]).reshape(-1, 1, 1)

    valid = ~np.isnan(mats)
    wsum = (valid * w_arr).sum(axis=0)
    total_w = sum(w.values())
    coverage = wsum / total_w  # 按权重计的可用因子占比
    numer = np.nansum(mats * w_arr, axis=0)
    combined_arr = np.where(coverage >= min_coverage, numer / np.maximum(wsum, 1e-12), np.nan)

    return pd.DataFrame(combined_arr, index=pd.DatetimeIndex(idx), columns=cols)


def smooth_score(raw: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """周度情绪分：window 日均值（决策用）；日度原始分仅作观察。"""
    return raw.rolling(window, min_periods=max(2, window // 2)).mean()


def percentile_rank(score: pd.DataFrame, lookback: int = 250) -> pd.DataFrame:
    """各板块情绪分的时序分位（0~1）：相对自身过去 lookback 日的位置。"""
    return score.rolling(lookback, min_periods=min(60, lookback)).apply(
        lambda s: (s.iloc[-1] >= s).mean(), raw=False)


def build_scores(factors: dict[str, pd.DataFrame], weights: dict[str, float],
                 smooth_window: int = 5,
                 lookback: int = 250) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """返回 (日度原始分, 周度平滑分, 平滑分时序分位)。

    平滑分只保留原始分实际存在的日期——避免某数据源多出一个
    交易日（如盘中未收盘的板块指数）导致报告日期虚标。
    lookback 为时序分位窗口，随市场 regime 缩短（如 120 天）。
    """
    raw = combine_factors(factors, weights)
    smoothed = smooth_score(raw, smooth_window)
    smoothed = smoothed.loc[raw.dropna(how="all").index]
    pct = percentile_rank(smoothed, lookback=lookback)
    return raw, smoothed, pct
