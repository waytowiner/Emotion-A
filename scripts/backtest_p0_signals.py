"""P0 突变穿透信号回测：最近 90 交易日的量化验证。

目的：验证 P0 代理信号（成交额价格代理 z>2.0、|单因子z|>2σ、代理连续为正）
量化三条穿透信号在最近 90 个交易日的前瞻收益表现，判断阈值是否需要调整。

三条规则均为原始日度读数、不经平滑（平滑会抹平突变），与经平滑的合成情绪分互补。
前瞻收益口径：T 收盘买入 → T+h 收盘（与 emotion.backtest 一致）。

输出：report/p0_signal_backtest_YYYYMMDD.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from emotion import pipeline, score, signal
from emotion.factors import compute_all_factors
from emotion.score import zscore_cross_section
from emotion.backtest import forward_returns

DAYS = 90
HORIZONS = [5, 10, 20]


def _stats(arr: np.ndarray, mask: np.ndarray) -> dict:
    """arr: 前瞻收益矩阵；mask: 同形状布尔。返回触发子集统计。"""
    finite = np.isfinite(arr)
    m = mask & finite
    vals = arr[m]
    n = len(vals)
    if n == 0:
        return {"n": 0}
    return {
        "n": n,
        "mean": vals.mean(),
        "win": (vals > 0).mean(),
        "median": np.median(vals),
        "days_with_trig": int((m.any(axis=1)).sum()),
    }


def main() -> None:
    cfg = pipeline.load_config()
    print("[1/5] 准备数据 ...")
    data = pipeline.prepare(cfg, end_override=pipeline.cache_end_date(cfg))

    print("[2/5] 计算因子与成交额价格代理 ...")
    factors = compute_all_factors(data["stock_daily"], data["board_map"],
                                  data["board_kline"],
                                  market_total=data.get("market_total"))
    flow_proxy = signal.compute_board_flow_proxy(data["stock_daily"], data["board_map"])

    print("[3/5] 合成情绪分 ...")
    _, smoothed, pct = score.build_scores(
        factors, cfg["weights"], cfg.get("smooth_window", 5),
        lookback=cfg.get("score_lookback", 250),
        z_clip=cfg.get("score_z_clip", 3.0))

    sig_cfg = cfg["signal"]
    bt_cfg = sig_cfg.get("breakthrough", {})
    thr1 = float(bt_cfg.get("flow_proxy_z_threshold",
                            bt_cfg.get("fund_flow_z_threshold", 2.0)))
    sigma = float(bt_cfg.get("extreme_sigma", 2.0))
    sigma_hot = float(bt_cfg.get("extreme_hot_sigma", sigma))
    sigma_cold = float(bt_cfg.get("extreme_cold_sigma", sigma))
    ndays = int(bt_cfg.get("flow_positive_days", bt_cfg.get("fund_positive_days", 3)))
    cold_q = sig_cfg.get("cold_quantile", 0.1)
    hot_q = sig_cfg.get("hot_quantile", 0.9)

    board_close = (data["board_kline"]
                   .pivot(index="date", columns="board_code", values="close")
                   .sort_index())

    print("[4/5] 构建信号掩码 ...")
    common_boards = sorted(set(pct.columns) & set(board_close.columns))
    idx = pct.index
    pct_a = pct.reindex(index=idx, columns=common_boards)
    sm_a = smoothed.reindex(index=idx, columns=common_boards)
    ff_a = flow_proxy.reindex(index=idx, columns=common_boards)
    bc_a = board_close.reindex(index=idx, columns=common_boards)

    cold_mask = (pct_a < cold_q).to_numpy()
    hot_mask = (pct_a > hot_q).to_numpy()

    ffz = zscore_cross_section(ff_a).to_numpy()
    rule1 = cold_mask & (ffz > thr1)

    # 各因子横截面 |z| 的逐格最大值（NaN 安全）
    factor_z_abs = []
    for name, f in factors.items():
        if f is None or f.empty:
            continue
        z = zscore_cross_section(f).abs().reindex(index=idx, columns=common_boards)
        factor_z_abs.append(z.to_numpy())
    if factor_z_abs:
        stacked = np.stack(factor_z_abs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # 全 NaN 列取 max 无害
            max_abs = np.nanmax(stacked, axis=0)
    else:
        max_abs = np.full(cold_mask.shape, np.nan)
    # 规则2 拆分：热端/冷端（已有拆分回测证实热端胜率58%、冷端34%）
    rule2_hot = hot_mask & (max_abs > sigma_hot)
    rule2_cold = cold_mask & (max_abs > sigma_cold)

    ff_pos = (ff_a > 0).to_numpy()
    # 滑窗 ndays 内全部为正：用累计计数技巧（向量化，避免 Python 循环）
    # streak_n[t] = 最近 ndays 日全为正的布尔
    streak_sum = np.full(cold_mask.shape, np.nan)
    ff_pos_finite = ff_pos & np.isfinite(ff_a.to_numpy())
    for i in range(len(idx)):
        lo = max(0, i - ndays + 1)
        if i - lo + 1 == ndays:
            streak_sum[i] = ff_pos_finite[lo:i + 1].all(axis=0).astype(float)
    fund_streak = (streak_sum == 1.0)
    rule3 = fund_streak & (sm_a.to_numpy() < 0)

    masks = {
        f"底部反转 (冰点+价格代理z>{thr1})": rule1,
        f"顺势加速度 (亢奋+单因子>{sigma_hot}σ)": rule2_hot,
        f"极端冷区异动 (冰点+单因子>{sigma_cold}σ)": rule2_cold,
        f"代理-情绪背离 (价格代理连{ndays}日正+情绪<0)": rule3,
    }
    baselines = {
        f"底部反转 (冰点+价格代理z>{thr1})": cold_mask,
        f"顺势加速度 (亢奋+单因子>{sigma_hot}σ)": hot_mask,
        f"极端冷区异动 (冰点+单因子>{sigma_cold}σ)": cold_mask,
        f"代理-情绪背离 (价格代理连{ndays}日正+情绪<0)": (sm_a.to_numpy() < 0),
    }

    print("[5/5] 计算前瞻收益 ...")
    today = pd.Timestamp.today()
    lines = []
    lines.append(f"# P0 突变穿透信号回测（最近 {DAYS} 交易日）")
    lines.append("")
    lines.append(f"> 生成：{today:%Y-%m-%d %H:%M}  |  数据截至：{idx[-1]:%Y-%m-%d}  |  "
                 f"板块口径：{cfg.get('board_source')}（{len(common_boards)} 个）")
    lines.append(f"> 阈值：成交额价格代理 z > {thr1}、热端 σ > {sigma_hot}、冷端 σ > {sigma_cold}、代理连续 {ndays} 日为正  "
                 f"|  冰点 < {cold_q}、亢奋 > {hot_q}")
    lines.append("> “成交额价格代理”=成交额×涨跌幅，不是真实资金净流入；三条规则均为原始日度读数、不经平滑。")
    lines.append("> 超额(vs市场)=信号平均-全市场平均；超额(vs对照)=信号平均-对应分位组平均。")
    lines.append("> 每个前瞻窗口取最近 90 个**可观测**交易日（该窗口有未来收益的日期）。")
    lines.append("")

    for name, mask in masks.items():
        lines.append(f"## {name}")
        lines.append("")
        lines.append("| 前瞻 | 触发数 | 日均触发 | 平均收益 | 胜率 | 中位数 | 市场基准 | 超额(vs市场) | 对照基准 | 超额(vs对照) |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        base_mask = baselines[name]
        for h in HORIZONS:
            fwd = forward_returns(bc_a, h).to_numpy()
            # 该窗口下可观测触发日：该日至少一个板块有未来收益
            valid_row = np.isfinite(fwd).any(axis=1)
            valid_idx = np.where(valid_row)[0]
            if len(valid_idx) == 0:
                continue
            sel = valid_idx[-DAYS:]
            fwd_win = fwd[sel]
            mask_win = mask[sel]
            base_win = base_mask[sel]

            st = _stats(fwd_win, mask_win)
            if st["n"] == 0:
                lines.append(f"| {h}日 | 0 | — | — | — | — | — | — | — | — |")
                continue
            finite_all = np.isfinite(fwd_win)
            mkt = fwd_win[finite_all].mean()
            base_vals = fwd_win[base_win & finite_all]
            base_mean = base_vals.mean() if len(base_vals) else np.nan
            exc_mkt = st["mean"] - mkt
            exc_base = st["mean"] - base_mean
            avg_per_day = st["n"] / max(st["days_with_trig"], 1)
            lines.append(
                f"| {h}日 | {st['n']} | {avg_per_day:.1f} | {st['mean']:+.2%} | "
                f"{st['win']:.0%} | {st['median']:+.2%} | {mkt:+.2%} | {exc_mkt:+.2%} | "
                f"{base_mean:+.2%} | {exc_base:+.2%} |"
            )
        # 对照基准规模
        last_window = slice(max(0, len(idx) - DAYS), len(idx))
        base_size = int((base_mask[last_window] & np.isfinite(
            forward_returns(bc_a, HORIZONS[0]).to_numpy()[last_window])).sum())
        lines.append("")
        lines.append(f"> 对照基准（{name.split('(')[0].strip()}）规模：近 {DAYS} 日约 "
                     f"{base_size} 板块-日。超额(vs对照)>0 表示穿透规则在该子集上"
                     f"优于只看分位。")
        lines.append("")

    lines.append("## 后续验证纪律")
    lines.append("")
    lines.append("- 本回测为样本内统计；不要根据同一段数据的触发数直接放宽/收紧阈值。")
    lines.append("- 新阈值先作为候选写入独立研究记录，随后用未参与选择的新时间段验证。")
    lines.append("- 代理信号的正超额不等于真实资金流入；接入主动买卖盘数据前只作价格行为辅助读数。")
    lines.append("")

    out_dir = ROOT / cfg["paths"]["report_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"p0_signal_backtest_{today:%Y%m%d}.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\n已保存：{out_path}")


if __name__ == "__main__":
    main()
