"""临时诊断：区分"涨停生态退潮"与"动量信号反转"。
按 30 日分段输出：市场涨停家数、打板生态原始读数（赚钱效应/晋级率均值）、
高RPS组 vs 低RPS组的 5 日前瞻收益差（纯价格动量的多空价差）。不进主流程。"""
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
    fwd5 = backtest.forward_returns(board_close, 5)

    sd = data["stock_daily"]
    from emotion.factors import flag_limit_up
    flagged = flag_limit_up(sd)
    lu_daily = flagged[flagged["is_limit_up"]].groupby("date").size()

    recent_idx = smoothed.tail(TOTAL).index
    rps = factors["rps"]

    print(f"\n{'段':24s} {'涨停/日':>8s} {'赚钱效应%':>9s} {'晋级率%':>8s} "
          f"{'高RPS-低RPS 5日价差':>18s}")
    for i in range(0, len(recent_idx), BUCKET):
        idx = recent_idx[i:i + BUCKET]
        if len(idx) < 20:
            continue
        d0, d1 = idx.min(), idx.max()
        n_lu = lu_daily.reindex(idx).mean()
        # money_effect 原始口径即 pct_chg（百分数单位），不再 ×100
        me = factors["money_effect"].loc[idx].stack().mean()
        pr = factors["promotion_rate"].loc[idx].stack().mean() * 100
        # 高RPS vs 低RPS 5日前瞻价差
        diffs = []
        for dt in idx.intersection(fwd5.index):
            pair = pd.concat([rps.loc[dt], fwd5.loc[dt]], axis=1,
                             keys=["r", "f"]).dropna()
            if len(pair) < 20:
                continue
            hi = pair.nlargest(max(3, len(pair) // 10), "r")["f"].mean()
            lo = pair.nsmallest(max(3, len(pair) // 10), "r")["f"].mean()
            diffs.append(hi - lo)
        diff = pd.Series(diffs).mean() * 100 if diffs else float("nan")
        print(f"{d0:%m-%d} ~ {d1:%m-%d}   n={len(idx):3d} {n_lu:8.0f} "
              f"{me:9.2f} {pr:8.2f} {diff:17.2f}%")


if __name__ == "__main__":
    main()
