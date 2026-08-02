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
        lookback=cfg.get("score_lookback", 250))

    # 回测窗口：默认最近 N 个交易日（适应当前市场风格）
    window = cfg.get("backtest_window")
    if window:
        smoothed = smoothed.tail(int(window))

    print("[4/4] 回测 ...")
    board_close = data["board_kline"].pivot(
        index="date", columns="board_code", values="close").sort_index()
    if window:
        board_close = board_close.loc[smoothed.index[0]:]
    bt_cfg = cfg["backtest"]
    ic_df = backtest.ic_analysis(smoothed, board_close, bt_cfg["forward_days"])
    quant_df = backtest.quantile_test(smoothed, board_close,
                                      horizon=bt_cfg["forward_days"][0])
    rot_top = backtest.rotation_backtest(
        smoothed, board_close,
        top_n=cfg["signal"].get("top_n", 3),
        cost_bps=bt_cfg.get("cost_bps", 5),
        direction="top",
    )
    rot_bottom = backtest.rotation_backtest(
        smoothed, board_close,
        top_n=cfg["signal"].get("top_n", 3),
        cost_bps=bt_cfg.get("cost_bps", 5),
        direction="bottom",
    )
    rot_avoid = backtest.rotation_backtest(
        smoothed, board_close,
        top_n=cfg["signal"].get("top_n", 3),
        cost_bps=bt_cfg.get("cost_bps", 5),
        direction="avoid_top",
    )
    md = backtest.format_backtest_report(ic_df, quant_df, rot_top, rot_bottom,
                                         rot_avoid)

    out_dir = ROOT / cfg["paths"]["report_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"backtest_{pd.Timestamp.today():%Y%m%d}.md"
    path.write_text(md, encoding="utf-8")
    print(md)
    print(f"\n已保存：{path}")


if __name__ == "__main__":
    main()
