# -*- coding: utf-8 -*-
"""临时诊断：市场等权指数走势 + 回撤 + 关键宽度指标，判断是否进入准熊市。
只用板块 K 线（已补齐至最新交易日），不依赖个股数据。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd

CACHE = ROOT / "data" / "cache"

def main() -> None:
    frames = []
    for f in CACHE.glob("board_kline_etf_*.parquet"):
        k = pd.read_parquet(f, columns=["date", "close"])
        k["board_code"] = f.stem.replace("board_kline_etf_", "")
        frames.append(k)
    bk = pd.concat(frames, ignore_index=True)
    close = bk.pivot(index="date", columns="board_code", values="close").sort_index()
    close = close[close.columns[close.notna().sum() >= 20]]
    pct = close.pct_change()

    # 市场等权日收益
    eq_ret = pct.mean(axis=1)
    eq_nav = (1 + eq_ret).cumprod()

    # 从最近高点回撤
    peak = eq_nav.cummax()
    dd = eq_nav / peak - 1
    cur_dd = dd.iloc[-1]
    peak_date = eq_nav[eq_nav == peak.iloc[-1]].index[-1]

    print(f"数据截止: {close.index.max():%Y-%m-%d}, 板块数: {close.shape[1]}")
    print(f"等权净值(窗口起点=1): {eq_nav.iloc[-1]:.3f}")
    print(f"最近高点日: {peak_date:%Y-%m-%d}, 当前自高点回撤: {cur_dd:+.2%}")
    # 高点之前的历史最大回撤
    print(f"窗口内最大回撤: {dd.min():+.2%} @ {dd.idxmin():%Y-%m-%d}")

    # 区间涨跌幅
    def rng(n):
        if len(eq_nav) <= n:
            return float("nan")
        return eq_nav.iloc[-1] / eq_nav.iloc[-1 - n] - 1

    print(f"\n近 5 日: {rng(5):+.2%}  近 10 日: {rng(10):+.2%}  "
          f"近 20 日: {rng(20):+.2%}")
    print(f"近 60 日: {rng(60):+.2%}  近 120 日: {rng(120):+.2%}")

    # 上涨家数占比（板块口径宽度）
    up_ratio = (pct > 0).mean(axis=1)
    print(f"\n近 5 日平均上涨板块占比: {up_ratio.tail(5).mean():.1%}")
    print(f"近 20 日平均上涨板块占比: {up_ratio.tail(20).mean():.1%}")
    print(f"最新一日上涨板块占比: {up_ratio.iloc[-1]:.1%}")

    # 60 日新高/新低板块数
    hi60 = (close == close.rolling(60).max()).sum(axis=1)
    lo60 = (close == close.rolling(60).min()).sum(axis=1)
    print(f"\n60日新高板块数(最新): {hi60.iloc[-1]}, 60日新低板块数(最新): {lo60.iloc[-1]}")
    print(f"近10日平均: 新高 {hi60.tail(10).mean():.0f} / 新低 {lo60.tail(10).mean():.0f}")

    # 最近 30 个交易日逐日等权收益
    print("\n近 15 交易日逐日等权收益:")
    for dt, r in eq_ret.tail(15).items():
        print(f"  {dt:%m-%d}: {r:+.2%}")

    # 月度收益
    m_ret = eq_nav.resample("ME").last().pct_change().dropna()
    print("\n近 6 个月等权月收益:")
    for dt, r in m_ret.tail(6).items():
        print(f"  {dt:%Y-%m}: {r:+.2%}")

if __name__ == "__main__":
    main()
