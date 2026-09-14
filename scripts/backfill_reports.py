"""补生成历史日报：python scripts/backfill_reports.py <start> [end]

适用场景：自动任务断档若干交易日后，用已补齐的缓存按日截断数据，
逐日复现 main.py 的报告流程（因子 -> 情绪分 -> 信号 -> 日报 + CSV）。

无前视保证：每日只用 <= 当日 的个股日线与板块指数（stock_daily /
market_total / board_kline / smoothed 全部按日截断），时序分位窗口
lookback 亦只回看，健康度与 regime 按当日时点计算。

用法：
    python scripts/backfill_reports.py 2026-09-01 2026-09-07
    python scripts/backfill_reports.py 2026-09-01          # 单日
已存在的 report_YYYYMMDD.md 默认跳过（--force 覆盖）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from emotion import backtest as _bt
from emotion import pipeline, report, score, signal
from emotion.factors import compute_all_factors


def build_one(cfg: dict, day: pd.Timestamp) -> Path:
    """生成 day 当日的日报，返回 md 路径。数据按 <= day 截断。"""
    end = day.strftime("%Y-%m-%d")
    data = pipeline.prepare(cfg, end_override=end)

    d_ts = day
    stock = data["stock_daily"]
    stock = stock[stock["date"] <= d_ts]
    market_total = data["market_total"]
    market_total = market_total[market_total.index <= d_ts]
    board_kline = data["board_kline"]
    board_kline = board_kline[pd.to_datetime(board_kline["date"]) <= d_ts]

    factors = compute_all_factors(stock, data["board_map"], board_kline,
                                  market_total=market_total)
    flow_proxy = signal.compute_board_flow_proxy(stock, data["board_map"])

    raw, smoothed, pct = score.build_scores(
        factors, cfg["weights"], cfg.get("smooth_window", 5),
        lookback=cfg.get("score_lookback", 250),
        z_clip=cfg.get("score_z_clip", 3.0))
    if smoothed.empty or smoothed.index[-1] != day:
        raise RuntimeError(f"{end}: 截断后情绪分末日为 "
                           f"{None if smoothed.empty else smoothed.index[-1]}，"
                           "与目标日不一致（缺行情数据？）")

    sig_cfg = cfg["signal"]
    signals = signal.detect_signals(
        smoothed, pct,
        consecutive_days=sig_cfg.get("consecutive_days", 3),
        hot_quantile=sig_cfg.get("hot_quantile", 0.9),
        cold_quantile=sig_cfg.get("cold_quantile", 0.1),
        factors=factors, flow_proxy=flow_proxy,
        breakthrough_cfg=sig_cfg.get("breakthrough"),
    )
    board_close = board_kline.pivot(
        index="date", columns="board_code", values="close").sort_index()
    health = _bt.recent_signal_health(smoothed, board_close, days=60)
    regime_cfg = cfg.get("regime", {})
    regime = _bt.regime_state(
        smoothed, board_close,
        ic_window=regime_cfg.get("ic_window", 15),
        threshold=regime_cfg.get("threshold", 0.02),
        horizon=regime_cfg.get("horizon", 5),
        min_obs=regime_cfg.get("min_obs", 10),
        guard_days=regime_cfg.get("guard_days", 10),
        guard_threshold=regime_cfg.get("guard_threshold", -0.03),
        vol_window=regime_cfg.get("vol_window", 20),
        vol_threshold=regime_cfg.get("vol_threshold"),
        require_robust_signal=regime_cfg.get("require_robust_signal", False),
        robust_ic_window=regime_cfg.get("robust_ic_window", 60),
        robust_bootstrap_samples=regime_cfg.get("robust_bootstrap_samples", 500),
        mono_days=regime_cfg.get("mono_days"),
    )
    regime["ic_window"] = regime_cfg.get("ic_window", 15)
    regime["guard_days"] = regime_cfg.get("guard_days", 10)
    regime["mono_days"] = regime_cfg.get("mono_days")

    md = report.build_report(day, smoothed, pct, signals, data["board_names"],
                             health=health, regime=regime,
                             breakthrough=sig_cfg.get("breakthrough"),
                             universe=data.get("universe"))
    md_path, csv_path = report.save_report(
        ROOT / cfg["paths"]["report_dir"], day, md, smoothed,
        data["board_names"])
    return md_path


def main() -> None:
    ap = argparse.ArgumentParser(description="补生成历史日报（无前视按日截断）")
    ap.add_argument("start", help="起始日期 YYYY-MM-DD")
    ap.add_argument("end", nargs="?", default=None, help="结束日期（默认=起始）")
    ap.add_argument("--force", action="store_true", help="覆盖已存在的报告")
    args = ap.parse_args()

    cfg = pipeline.load_config()
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end or args.start)

    # 交易日近似：跳过周末；节假日会因"截断后末日不一致"报错跳过
    days = pd.date_range(start, end, freq="B")
    for day in days:
        md_path = ROOT / cfg["paths"]["report_dir"] / \
            f"report_{day:%Y%m%d}.md"
        if md_path.exists() and not args.force:
            print(f"[skip] {day:%Y-%m-%d} 已存在 {md_path.name}", flush=True)
            continue
        try:
            path = build_one(cfg, day)
            print(f"[ok] {day:%Y-%m-%d} -> {path.name}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[fail] {day:%Y-%m-%d}: {exc}", flush=True)


if __name__ == "__main__":
    main()
