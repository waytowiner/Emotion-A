"""验证重新导出的日报/看板与自适应回测，不搜索策略参数。

运行：python scripts/validate_consistency.py [--expected-config-sha256 HASH]
先运行 main.py --offline 与 scripts/dump_dashboard_data.py 更新分析产物。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from emotion import backtest, pipeline
from emotion.project_status import OBSERVATION_NOTICE, project_status
from scripts.backtest_regime_adaptive import adaptive_parameters, adaptive_rotation, seg_ret


def main() -> None:
    ap = argparse.ArgumentParser(description="交易日和日报/回测规则一致性验证")
    ap.add_argument("--expected-config-sha256")
    args = ap.parse_args()
    digest = hashlib.sha256((ROOT / "config.yaml").read_bytes()).hexdigest()
    if args.expected_config_sha256:
        assert digest.lower() == args.expected_config_sha256.lower(), "策略配置发生变化"
    cfg = pipeline.load_config()
    folder = ROOT / "data" / "dashboard"
    latest = json.loads((folder / "latest.json").read_text(encoding="utf-8"))
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    scores = pd.read_parquet(folder / "scores.parquet").set_index("date")
    close = pd.read_parquet(folder / "board_kline.parquet").pivot(
        index="date", columns="board_code", values="close").sort_index()
    pd.testing.assert_index_equal(scores.index, close.index)
    assert not (close.index.dayofweek >= 5).any(), "价格面板仍含周末"
    assert meta["score_end"] == latest["date"] == scores.index[-1].strftime("%Y-%m-%d")

    parameters = backtest.regime_parameters(cfg["regime"])
    known = backtest.realized_series(backtest.daily_ic_series(scores, close, parameters["horizon"]),
                                     close.index, lag=parameters["horizon"])
    rot = adaptive_rotation(scores, close, known, top_n=cfg["signal"]["top_n"],
                             cost_bps=cfg["backtest"]["cost_bps"], **adaptive_parameters(cfg["regime"]))
    log = rot["regime_log"]
    numeric = ["ic_mean", "ic_win_rate", "mkt_recent", "vol_now", "spread_last",
               "robust_ic_low", "robust_ic_high", "n_obs"]
    gates = ["guard_hit", "vol_gate", "robust_gate", "mono_gate"]
    for _, row in log.iterrows():
        dt = row["date"]
        # 从截断至当日的价格重新计算 IC，独立检查全样本预计算没有混入未来。
        daily = backtest.regime_state(scores.loc[:dt], close.loc[:dt], **parameters)
        assert daily["direction"] == row["state_direction"], f"单日方向不一致：{dt}"
        assert daily["raw_direction"] == row["raw_direction"], f"原始方向不一致：{dt}"
        for key in gates:
            assert daily[key] == bool(row[key]), f"闸门不一致：{dt} {key}"
        for key in numeric:
            np.testing.assert_allclose(daily[key], row[key], atol=1e-12, rtol=1e-12,
                                       equal_nan=True, err_msg=f"{dt} {key}")
    snapshot = backtest.regime_snapshot(scores, close, cfg["regime"], top_n=cfg["signal"]["top_n"])
    for key in ["direction", "raw_direction", "rebalance_direction", "rebalance_date", *gates]:
        assert snapshot[key] == latest["regime"][key], f"看板判定不一致：{key}"
    for key in numeric:
        actual = latest["regime"][key]
        np.testing.assert_allclose(snapshot[key], np.nan if actual is None else actual,
                                   atol=1e-12, rtol=1e-12, equal_nan=True)
    report_day = scores.index[-1].strftime("%Y%m%d")
    daily_report = (ROOT / "report" / f"report_{report_day}.md").read_text(encoding="utf-8")
    direction_cn = {"top": "正向", "bottom": "反向", "neutral": "中性"}
    assert f"原模型读数：{direction_cn[snapshot['direction']]}" in daily_report
    assert f"：{snapshot['ic_mean']:+.3f}（胜率" in daily_report
    assert f"最近研究信号日：{snapshot['rebalance_date']}" in daily_report
    assert OBSERVATION_NOTICE in daily_report
    assert latest["project_status"] == project_status()

    health = backtest.recent_signal_health(scores, close, days=60)
    audit = meta["calendar_audit"]
    lines = ["# 交易日与日报/回测一致性验证", "",
             f"数据区间：{close.index[0]:%Y-%m-%d} ~ {close.index[-1]:%Y-%m-%d}；"
             f"{len(close)} 个 A 股交易日。", "",
             f"- 配置 SHA256：`{digest}`；" + ("与修复前完全一致。" if args.expected_config_sha256 else "仅记录当前配置。"),
             f"- 过滤 {len(audit['removed_dates'])} 个非 A 股交易日、{audit['removed_board_rows']} 行板块数据。",
             f"- {len(log)} 个调仓日：重新截断价格计算的日报读数、四道闸门与回测全部一致（数值容差 1e-12）。",
             "- 最新日报和看板的当日风格、IC、闸门及最近周度确认方向一致。",
             "- 连续两次周度确认、四道闸门和原有换仓成本口径均沿用当前配置。",
             f"- 最新单日风格：{direction_cn[snapshot['direction']]}；最近调仓日 {snapshot['rebalance_date']}"
             f" 的确认方向：{direction_cn[snapshot['rebalance_direction']]}。",
             f"- 近60个已实现 IC：{health['ic_mean']:+.4f}；Bootstrap 95% CI "
             f"[{health['IC_bootstrap_low']:+.4f}, {health['IC_bootstrap_high']:+.4f}]。",
             "", "## 修复后的完整固定配置回测", "",
             "| 指标 | 自适应策略 | 板块等权参考（不可交易） |", "|---|---|---|"]
    for key, name in [("total_ret", "累计收益"), ("max_dd", "最大回撤")]:
        lines.append(f"| {name} | {rot['strategy'][key]:+.2%} | {rot['benchmark'][key]:+.2%} |")
    lines += [f"| 夏普 | {rot['strategy']['sharpe']:.2f} | {rot['benchmark']['sharpe']:.2f} |",
              f"| 2026年7月以来 | {seg_ret(rot['nav'], '2026-07-01'):+.2%} | "
              f"{seg_ret(rot['benchmark_nav'], '2026-07-01'):+.2%} |", "",
              "本次验证修复计算口径，不证明策略盈利或 ETF 可交易性；未搜索参数。"
              "历史成分快照及指数模拟的原有边界仍在。", ""]
    out = ROOT / "report" / f"consistency_validation_{report_day}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    log.to_csv(out.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    print("\n".join(lines))
    print(f"验证通过：{out}")


if __name__ == "__main__":
    main()
