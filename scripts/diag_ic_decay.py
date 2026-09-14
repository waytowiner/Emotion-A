"""临时诊断：把最近 180 个交易日切成 30 天一段，逐段计算 IC/分层，
定位信号衰减从哪一段开始。诊断脚本，不进主流程。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from emotion import backtest, pipeline, score
from emotion.factors import compute_all_factors

BUCKET = 30
TOTAL = 180


def main() -> None:
    cfg = pipeline.load_config()
    data = pipeline.prepare(cfg, end_override=pipeline.cache_end_date(cfg))
    factors = compute_all_factors(data["stock_daily"], data["board_map"],
                                  data["board_kline"],
                                  market_total=data.get("market_total"))
    _, smoothed, _ = score.build_scores(factors, cfg["weights"],
                                        cfg.get("smooth_window", 5))
    board_close = data["board_kline"].pivot(
        index="date", columns="board_code", values="close").sort_index()

    recent = smoothed.tail(TOTAL)
    print(f"窗口: {recent.index.min():%Y-%m-%d} ~ {recent.index.max():%Y-%m-%d}\n")

    # 市场背景：等权日收益的 20 日波动与累计收益，判断 regime
    eq_ret = board_close.pct_change().mean(axis=1).tail(TOTAL)
    for i in range(0, len(recent), BUCKET):
        seg = recent.iloc[i:i + BUCKET]
        if len(seg) < 20:
            continue
        d0, d1 = seg.index.min(), seg.index.max()
        ic = backtest.ic_analysis(seg, board_close, [5, 10])
        q = backtest.quantile_test(seg, board_close, horizon=5)
        ic5 = ic[ic["horizon"] == 5].iloc[0]
        mkt = eq_ret.loc[d0:d1]
        q_str = ""
        if not q.empty:
            q_str = "  ".join(f"Q{int(g)+1}={r['mean_fwd_ret']:+.2%}"
                              for g, r in q.iterrows())
        print(f"[{d0:%m-%d} ~ {d1:%m-%d}]  n={len(seg)}")
        print(f"  IC5={ic5['IC_mean']:+.4f}  胜率={ic5['IC_win_rate']:.0%}"
              f"  |  IC10={ic[ic['horizon']==10].iloc[0]['IC_mean']:+.4f}"
              f"  |  等权累计={mkt.sum():+.1%}  日波动={mkt.std():.2%}")
        print(f"  分层: {q_str}\n")

    # 逐因子诊断：最近60日 vs 之前的60~120日，各单因子 IC5
    print("=" * 70)
    print("单因子 IC5 对比（近60日 vs 60~120日前）：")
    fwd = backtest.forward_returns(board_close, 5)

    def seg_ic(f: pd.DataFrame, idx: pd.Index) -> float:
        ics = []
        for dt in idx.intersection(fwd.index):
            pair = pd.concat([f.loc[dt], fwd.loc[dt]],
                             axis=1, keys=["s", "f"]).dropna()
            if len(pair) < 10:
                continue
            ics.append(pair["s"].corr(pair["f"], method="spearman"))
        return pd.Series(ics).mean() if ics else float("nan")

    for fname, f in factors.items():
        if not isinstance(f, pd.DataFrame) or f.empty:
            continue
        new = seg_ic(f, smoothed.tail(60).index)
        old = seg_ic(f, smoothed.iloc[-120:-60].index)
        print(f"  {fname:22s}  近60日: {new:+.4f}   60~120日前: {old:+.4f}")


if __name__ == "__main__":
    main()
