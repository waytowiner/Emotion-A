"""每日板块情绪报告入口：python main.py [--refresh-boards]"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from emotion import pipeline, report, score, signal
from emotion.factors import compute_all_factors


def main() -> None:
    ap = argparse.ArgumentParser(description="A股板块情绪日报")
    ap.add_argument("--refresh-boards", action="store_true", help="强制刷新板块列表与成分")
    ap.add_argument("--offline", action="store_true", help="仅使用现有缓存复算日报")
    args = ap.parse_args()

    cfg = pipeline.load_config()
    print("[1/4] 准备数据 ...")
    data = pipeline.prepare(cfg, refresh_boards=args.refresh_boards,
                            end_override=pipeline.cache_end_date(cfg), offline=args.offline)
    print(f"      板块 {data['board_map']['board_code'].nunique()} 个，"
          f"个股 {data['stock_daily']['code'].nunique()} 只，"
          f"日线 {len(data['stock_daily'])} 行")

    print("[2/4] 计算情绪因子 ...")
    factors = compute_all_factors(data["stock_daily"], data["board_map"],
                                  data["board_kline"],
                                  market_total=data.get("market_total"))
    # P0 穿透指标：成交额×涨跌幅代理（信号层用，不进合成情绪分）。
    # 它不是主动买卖盘意义的真实资金净流入。
    flow_proxy = signal.compute_board_flow_proxy(data["stock_daily"], data["board_map"])

    print("[3/4] 合成情绪分 ...")
    raw, smoothed, pct = score.build_scores(
        factors, cfg["weights"], cfg.get("smooth_window", 5),
        lookback=cfg.get("score_lookback", 250),
        z_clip=cfg.get("score_z_clip", 3.0))

    print("[4/4] 生成信号与报告 ...")
    sig_cfg = cfg["signal"]
    signals = signal.detect_signals(
        smoothed, pct,
        consecutive_days=sig_cfg.get("consecutive_days", 3),
        hot_quantile=sig_cfg.get("hot_quantile", 0.9),
        cold_quantile=sig_cfg.get("cold_quantile", 0.1),
        factors=factors, flow_proxy=flow_proxy,
        breakthrough_cfg=sig_cfg.get("breakthrough"),
    )
    # 信号健康度：最近 60 日 IC 监控（风格切换感知）
    from emotion import backtest as _bt
    board_close = data["board_kline"].pivot(
        index="date", columns="board_code", values="close").sort_index()
    health = _bt.recent_signal_health(smoothed, board_close, days=60)
    # 风格判定（regime 自适应）：滚动已实现 IC + 回撤护栏 -> 方向建议
    regime_cfg = cfg.get("regime", {})
    regime = _bt.regime_snapshot(smoothed, board_close, regime_cfg,
                                 top_n=sig_cfg.get("top_n", 3))
    # 因子层监控：逐因子 IC + 方向可信度（区分"可信为负"与"风格切换"）
    factor_ic = _bt.factor_ic_snapshot(factors, board_close)
    date = smoothed.index[-1]
    md = report.build_report(date, smoothed, pct, signals, data["board_names"],
                              health=health, regime=regime,
                              breakthrough=sig_cfg.get("breakthrough"),
                              universe=data.get("universe"),
                              factor_ic=factor_ic)
    md_path, csv_path = report.save_report(
        ROOT / cfg["paths"]["report_dir"], date, md, smoothed, data["board_names"])
    # 前瞻预注册台账：事前判定留痕（同日重跑幂等跳过）
    report.append_prospective_log(
        ROOT / cfg["paths"]["report_dir"], date, regime, health)
    print(f"完成：{md_path}")
    print(f"      {csv_path}")


if __name__ == "__main__":
    main()
