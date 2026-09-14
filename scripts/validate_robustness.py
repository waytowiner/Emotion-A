"""P2 稳健性验证：IC 重叠校正 + 时间顺序验证块。

用法：python scripts/validate_robustness.py

这不是参数搜索器。它固定当前 config.yaml 的因子、标的池和预先声明的窗口，
检查该读数在时间维度上是否稳定；若结论不稳，应降低仓位/观望，而非继续在同一
段历史上调阈值。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from emotion import backtest, pipeline, score
from emotion.factors import compute_all_factors


def _fmt(v: float | None, fmt: str = ".4f") -> str:
    return "—" if v is None or pd.isna(v) else format(v, fmt)


def main() -> None:
    cfg = pipeline.load_config()
    ev = cfg.get("evaluation", {})
    samples = int(ev.get("bootstrap_samples", 1_000))
    block_days = int(ev.get("rolling_block_days", 20))

    print("[1/3] 准备缓存数据与治理后的板块池 ...")
    data = pipeline.prepare(cfg, end_override=pipeline.cache_end_date(cfg))
    print("[2/3] 计算当前固定配置的情绪分 ...")
    factors = compute_all_factors(data["stock_daily"], data["board_map"],
                                  data["board_kline"],
                                  market_total=data.get("market_total"))
    _, smoothed, _ = score.build_scores(
        factors, cfg["weights"], cfg.get("smooth_window", 5),
        lookback=cfg.get("score_lookback", 120),
        z_clip=cfg.get("score_z_clip", 3.0))
    close = data["board_kline"].pivot(
        index="date", columns="board_code", values="close").sort_index()

    print("[3/3] 计算重叠校正与滚动验证 ...")
    lines = ["# 情绪分稳健性验证（P2）", "",
             f"评估日期：{smoothed.index[-1]:%Y-%m-%d}；IC bootstrap：{samples} 次；"
             f"时间验证块：{block_days} 个已实现 IC。", "",
             "## 固定配置与数据边界", "",
             f"- 入选板块：{data['universe']['eligible_board_count']}/"
             f"{data['universe']['raw_board_count']}（最小成分数 "
             f"{data['universe']['min_constituents']}；完全重复阈值 "
             f"{data['universe']['duplicate_jaccard']:.2f}）。",
             "- 因子、权重和 regime 参数直接读取 config.yaml；本脚本不扫描或选择参数。",
             "- 成分股使用当前快照回填历史，仍有幸存者偏差；结果只用于监控，不构成可交易证明。",
             "", "## IC 重叠校正", "",
             "| 前瞻 | IC均值 | 朴素IR | 非重叠IC/样本 | HAC t（p） | 区块Bootstrap 95% CI |", 
             "|---|---|---|---|---|---|"]

    h5_blocks = pd.DataFrame()
    for horizon in cfg["backtest"]["forward_days"]:
        ic = backtest.daily_ic_series(smoothed, close, horizon)
        robust = backtest.ic_robustness(ic, horizon=horizon,
                                        bootstrap_samples=samples)
        ir = ic.mean() / ic.std() if ic.std() else float("nan")
        lines.append(
            f"| {horizon}日 | {_fmt(ic.mean())} | {_fmt(ir, '.3f')} | "
            f"{_fmt(robust['IC_nonoverlap_mean'])}/{robust['IC_nonoverlap_n']} | "
            f"{_fmt(robust['IC_HAC_t'], '.2f')}（{_fmt(robust['IC_HAC_p'], '.3f')}） | "
            f"[{_fmt(robust['IC_bootstrap_low'])}, {_fmt(robust['IC_bootstrap_high'])}] |")
        if horizon == cfg["backtest"]["forward_days"][0]:
            h5_blocks = backtest.rolling_ic_validation(ic, block_days=block_days)

    lines += ["", "> HAC 与移动区块 bootstrap 会校正前瞻收益重叠造成的时间相关性。"
              "若区间覆盖 0，不能将该窗口视为已经证实的有效信号。", "",
              "## 时间顺序验证（首个前瞻窗口）", "",
              "| 已实现区间 | 样本 | 本块IC | 胜率 | 前一块IC |", "|---|---|---|---|---|"]
    for _, row in h5_blocks.iterrows():
        lines.append(
            f"| {row['start']:%Y-%m-%d} ~ {row['end']:%Y-%m-%d} | {int(row['n_obs'])} | "
            f"{row['IC_mean']:+.4f} | {row['IC_win_rate']:.0%} | "
            f"{_fmt(row['previous_IC_mean'], '+.4f')} |")
    lines += ["", "## 使用纪律", "",
              "1. 不用本报告中表现最好的窗口/阈值直接改生产参数；新的参数必须先写入配置，"
              "再在后续未参与选择的数据上验证。",
              "2. 最近两个时间验证块方向相反，或 bootstrap 区间覆盖 0 时，维持观望/降低风险，"
              "不要靠增加调仓频率弥补。",
              "3. 板块等权参考不是可交易基准；ETF 映射仅用于落地候选，实际交易还需单独验证"
              "流动性、跟踪误差和费用。", ""]

    out_dir = ROOT / cfg["paths"]["report_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"robustness_{smoothed.index[-1]:%Y%m%d}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n已保存：{out}")


if __name__ == "__main__":
    main()
