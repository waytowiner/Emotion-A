"""回测入口：python backtest.py"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from emotion import backtest, pipeline, score
from emotion.factors import compute_all_factors


def main() -> None:
    cfg = pipeline.load_config()
    print("[1/4] 准备数据 ...")
    data = pipeline.prepare(cfg, end_override=pipeline.cache_end_date(cfg))

    print("[2/4] 计算情绪因子 ...")
    factors = compute_all_factors(data["stock_daily"], data["board_map"],
                                  data["board_kline"],
                                  market_total=data.get("market_total"))

    print("[3/4] 合成情绪分 ...")
    _, smoothed, _ = score.build_scores(
        factors, cfg["weights"], cfg.get("smooth_window", 5),
        lookback=cfg.get("score_lookback", 250),
        z_clip=cfg.get("score_z_clip", 3.0))

    # 回测窗口：多档验证（backtest_windows），兼容旧的单档 backtest_window
    windows = cfg.get("backtest_windows")
    if not windows:
        single = cfg.get("backtest_window")
        windows = [int(single)] if single else [None]
    windows = [int(w) if w else None for w in windows]

    print(f"[4/4] 回测（窗口：{['全部' if w is None else w for w in windows]}）...")
    board_close = data["board_kline"].pivot(
        index="date", columns="board_code", values="close").sort_index()
    bt_cfg = cfg["backtest"]
    top_n = cfg["signal"].get("top_n", 3)
    cost_bps = bt_cfg.get("cost_bps", 5)
    results = {}
    for w in windows:
        sm = smoothed.tail(w) if w else smoothed
        bc = board_close.loc[sm.index[0]:] if w else board_close
        results[w] = {
            "ic": backtest.ic_analysis(sm, bc, bt_cfg["forward_days"]),
            "quant": backtest.quantile_test(sm, bc,
                                            horizon=bt_cfg["forward_days"][0]),
            "top": backtest.rotation_backtest(
                sm, bc, top_n=top_n, cost_bps=cost_bps, direction="top"),
            "bottom": backtest.rotation_backtest(
                sm, bc, top_n=top_n, cost_bps=cost_bps, direction="bottom"),
            "avoid": backtest.rotation_backtest(
                sm, bc, top_n=top_n, cost_bps=cost_bps, direction="avoid_top"),
        }
    if len(results) == 1:
        (w, res), = results.items()
        md = backtest.format_backtest_report(
            res["ic"], res["quant"], res["top"], res["bottom"], res["avoid"],
            window_days=w)
    else:
        md = backtest.format_multi_window_report(results, bt_cfg["forward_days"])

    out_dir = ROOT / cfg["paths"]["report_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"backtest_{pd.Timestamp.today():%Y%m%d}.md"
    path.write_text(md, encoding="utf-8")
    print(md)
    print(f"\n已保存：{path}")


if __name__ == "__main__":
    main()
