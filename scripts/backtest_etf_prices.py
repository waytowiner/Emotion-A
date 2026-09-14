"""用真实 ETF 价格复跑 ETF 池的轮动回测（可执行口径）。

对比：同一套情绪分下，轮动收益分别用 指数官方收盘 与 对应 ETF 前复权收盘 计算。
前者是理论值，后者是真实可买到的价格（含跟踪误差/折溢价/上市日晚于指数）。

用法：python scripts/backtest_etf_prices.py [--window 180] [--end 2026-08-25]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from emotion import backtest, pipeline, score  # noqa: E402
from emotion.factors import compute_all_factors  # noqa: E402

CACHE = ROOT / "data" / "cache"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=180)
    ap.add_argument("--end", default="2026-08-25")
    args = ap.parse_args()

    cfg = dict(pipeline.load_config())
    cfg["board_source"] = "etf"
    data = pipeline.prepare(cfg, end_override=args.end)
    factors = compute_all_factors(
        data["stock_daily"], data["board_map"], data["board_kline"],
        market_total=data.get("market_total"))
    _, smoothed, _ = score.build_scores(
        factors, cfg["weights"], cfg.get("smooth_window", 5),
        lookback=cfg.get("score_lookback", 250))
    smoothed = smoothed.tail(args.window)

    # 指数价格（理论）
    idx_close = data["board_kline"].pivot(
        index="date", columns="board_code", values="close").sort_index()
    idx_close = idx_close.loc[smoothed.index[0]:]

    # ETF 价格（可执行）：指数代码 -> 规模最大 ETF 的前复权收盘
    tmap = pd.read_parquet(CACHE / "etf_trading_map.parquet")
    etf_close = pd.DataFrame(index=idx_close.index)
    mapped = 0
    for _, r in tmap.iterrows():
        p = CACHE / f"etf_kline_{r['etf_code']}.parquet"
        if not p.exists():
            continue
        k = pd.read_parquet(p)
        s = k.set_index("date")["close"]
        etf_close[r["board_code"]] = s
        mapped += 1
    etf_close = etf_close.sort_index()
    print(f"ETF 价格面板：{mapped} 只标的", flush=True)

    bt_cfg = cfg["backtest"]
    ic_df = backtest.ic_analysis(smoothed, idx_close, bt_cfg["forward_days"])
    ic_row = ic_df[ic_df["horizon"] == bt_cfg["forward_days"][0]].iloc[0]
    print(f"IC(指数价): mean={ic_row['IC_mean']:.4f} IR={ic_row['IR']:.3f} "
          f"胜率={ic_row['IC_win_rate']:.1%}")

    for label, close in [("指数官方价", idx_close), ("ETF 实际价", etf_close)]:
        # ETF 口径下只有映射到交易标的的板块才可买，选股池同步收窄
        sc = smoothed
        if label == "ETF 实际价":
            keep = [c for c in smoothed.columns if c in close.columns]
            sc = smoothed[keep]
        rot = backtest.rotation_backtest(
            sc, close,
            top_n=cfg["signal"].get("top_n", 3),
            cost_bps=bt_cfg.get("cost_bps", 5), direction="top")
        s = rot["strategy"]
        print(f"{label}: 年化={s['annual_ret']:+.1%} 夏普={s['sharpe']:.2f} "
              f"最大回撤={s['max_dd']:.1%} 期末净值={rot['final_nav']:.3f}",
              flush=True)


if __name__ == "__main__":
    main()
