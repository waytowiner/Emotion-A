# -*- coding: utf-8 -*-
"""临时诊断：用最近一期情绪分 + 最新板块价格，验证近 3 日 IC 实现情况。
绕开个股数据（更新中），只依赖 board_kline（已补齐至 09-11）+ 历史 scores csv。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

CACHE = ROOT / "data" / "cache"
REPORT = ROOT / "report"

def main() -> None:
    frames = []
    for f in CACHE.glob("board_kline_etf_*.parquet"):
        k = pd.read_parquet(f, columns=["date", "close"])
        k["board_code"] = f.stem.replace("board_kline_etf_", "")
        frames.append(k)
    bk = pd.concat(frames, ignore_index=True)
    close = bk.pivot(index="date", columns="board_code", values="close").sort_index()

    # 最近 8 期 scores，看每期 Top5/Bottom5 与 Q1/Q5 组合其后表现
    csvs = sorted(REPORT.glob("scores_*.csv"))[-8:]
    print("情绪分时点 → 之后累计等权收益（Top5 追涨 vs Bottom5 vs 全市场等权）")
    for c in csvs:
        dt = c.stem.replace("scores_", "")
        sc = pd.read_csv(c, index_col=0).iloc[:, 0] if len(pd.read_csv(c, index_col=0).columns) else None
        sc = pd.read_csv(c)
        # 列名可能是 board/name + score，自适应
        cols = sc.columns.tolist()
        date_idx = pd.Timestamp(dt)
        # 从该日期起的后 3 日 / 5 日收益
        if date_idx not in close.index:
            # 用最近一个 <= date_idx 的交易日
            valid = close.index[close.index <= date_idx]
            if len(valid) == 0:
                continue
            date_idx = valid[-1]
        pos = close.index.get_loc(date_idx)
        for fwd, label in [(3, "3日"), (5, "5日")]:
            if pos + fwd >= len(close.index):
                continue
            fwd_ret = close.iloc[pos + fwd] / close.iloc[pos] - 1
            mkt = fwd_ret.mean()
            board_col = "board_code" if "board_code" in cols else cols[0]
            score_col = [c2 for c2 in cols if "score" in c2.lower()]
            if not score_col:
                continue
            score_col = score_col[0]
            codes = sc[board_col].astype(str).str.zfill(6)
            vals = pd.Series(sc[score_col].values, index=codes).dropna()
            vals = vals[vals.index.isin(close.columns)]
            if len(vals) < 50:
                continue
            top5 = vals.nlargest(5).index
            bot5 = vals.nsmallest(5).index
            q = pd.qcut(vals.rank(method="first"), 5, labels=False)
            q1 = vals[q == 0].index
            q5 = vals[q == 4].index
            print(f"  {dt} → 后{label}: Top5 {fwd_ret[top5].mean():+.2%} | "
                  f"Bottom5 {fwd_ret[bot5].mean():+.2%} | 市场 {mkt:+.2%} | "
                  f"Q5 {fwd_ret[q5].mean():+.2%} | Q1 {fwd_ret[q1].mean():+.2%}")

    # 周度 IC 序列：用每期 scores 计算与后 5 日收益的 Spearman
    print("\n各期情绪分 vs 后5日收益 Spearman IC:")
    for c in csvs:
        dt = c.stem.replace("scores_", "")
        sc = pd.read_csv(c)
        cols = sc.columns.tolist()
        date_idx = pd.Timestamp(dt)
        valid = close.index[close.index <= date_idx]
        if len(valid) == 0:
            continue
        date_idx = valid[-1]
        pos = close.index.get_loc(date_idx)
        if pos + 5 >= len(close.index):
            continue
        fwd_ret = close.iloc[pos + 5] / close.iloc[pos] - 1
        board_col = "board_code" if "board_code" in cols else cols[0]
        score_col = [c2 for c2 in cols if "score" in c2.lower()][0]
        codes = sc[board_col].astype(str).str.zfill(6)
        vals = pd.Series(sc[score_col].values, index=codes)
        vals = vals[vals.index.isin(close.columns)].dropna()
        if len(vals) < 50:
            continue
        ic = vals.corr(fwd_ret[vals.index], method="spearman")
        print(f"  {dt}: IC5 = {ic:+.3f}")

if __name__ == "__main__":
    main()
