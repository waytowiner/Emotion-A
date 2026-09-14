# -*- coding: utf-8 -*-
"""验证 regime 回撤护栏：市场等权近 10 日累计收益是否低于 -3%。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
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
    eq_ret = close.pct_change().mean(axis=1)

    print("回撤护栏检查（guard_days=10, threshold=-3%）:")
    for n in [5, 10]:
        cum = (1 + eq_ret.tail(n)).prod() - 1
        print(f"  近 {n} 日累计: {cum:+.2%}" + ("  → 触发强制观望!" if n == 10 and cum < -0.03 else ""))
    # 每日滚动10日
    roll10 = (1 + eq_ret).rolling(10).apply(lambda x: x.prod() - 1, raw=False)
    print("\n近 10 个交易日逐日滚动 10 日累计:")
    for dt, v in roll10.tail(10).items():
        flag = " ← 触发" if v < -0.03 else ""
        print(f"  {dt:%m-%d}: {v:+.2%}{flag}")

if __name__ == "__main__":
    main()
