"""方案5验证：regime 自适应轮动回测。

策略：每周最后交易日收盘后，用"截至当日已可知"的滚动 IC 判定市场风格——
  IC > +阈值  -> 动量市，持仓情绪 Top N（追涨）
  IC < -阈值  -> 反转市，持仓情绪 Bottom N（抄底）
  中间        -> 观望，等权持有全部板块（防御）
  波动闸门（可选）：市场 20 日日波动 > 阈值时强制观望

无前视：IC 的前瞻收益必须在决策日之前走完（d+5 ≤ T），IC 序列整体后移 5 个
交易日作为"可知时点"。对照：固定追涨 / 固定抄底 / 等权基准 + 窗口×阈值网格。

运行：python scripts/backtest_regime_adaptive.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from emotion import backtest, pipeline, score
from emotion.factors import compute_all_factors

IC_WINDOW_DEFAULT = 60      # 滚动 IC 窗口（交易日）
THRESHOLD_DEFAULT = 0.03    # 与项目 0.03 显著性阈值一致
VOL_WINDOW = 20             # 波动闸门：20 日滚动日波动
VOL_THRESHOLD = 0.022       # 近期翻转段日波动 2.84%、平稳期 1.1-1.8%


from emotion.backtest import daily_ic_series


def ic_known_series(ic_s: pd.Series, calendar: pd.DatetimeIndex,
                    lag: int = 5) -> pd.Series:
    """把 IC 序列的索引后移 lag 个交易日：IC(d) 在 d+5 收盘后才可知。"""
    pos = calendar.searchsorted(ic_s.index)
    known_pos = pos + lag
    ok = known_pos < len(calendar)
    return pd.Series(ic_s.values[ok], index=calendar[known_pos[ok]])


def adaptive_rotation(score_df: pd.DataFrame, board_close: pd.DataFrame,
                       ic_known: pd.Series, *, top_n: int = 3,
                       cost_bps: float = 5.0, ic_window: int = 60,
                       threshold: float = 0.03,
                       min_obs: int = 10,
                      vol_series: pd.Series | None = None,
                      vol_threshold: float | None = None,
                      ew_halflife: float | None = None,
                      mkt_ret: pd.Series | None = None,
                      guard: float | None = None,
                      confirm: int = 1) -> dict:
    """regime 自适应周度轮动。neutral=等权持有全部板块。

    ew_halflife：对窗口内 IC 做指数加权（半衰期，交易日），反应更快。
    guard：回撤护栏——市场近 10 日累计收益 < guard（如 -0.03）时强制观望。
    confirm：方向确认迟滞——非观望方向需连续 confirm 次调仓判定同向才生效，
    未确认期间观望；风控闸门（波动/回撤）不受迟滞影响，始终即时生效。
    """
    days_with_score = score_df.dropna(how="all").index
    rebal_dates = backtest.rebalance_dates(days_with_score, freq="W-FRI")
    daily_ret = board_close.pct_change()
    cost = cost_bps / 10000.0
    price_boards = daily_ret.columns

    # 决策序列：先在各调仓日算出 IC 原始方向，整体做确认迟滞；
    # 风控闸门在交易循环内即时生效（风险降级不做迟滞）。
    raw_dir, ic_by_date = {}, {}
    for dt in rebal_dates:
        # 只考虑有行情数据的板块：有分无价（如 K 线抓取失败）的板块不可交易
        s = score_df.loc[dt].dropna()
        s = s[s.index.isin(price_boards)]
        if len(s) < top_n:
            continue
        hist = ic_known.loc[:dt].tail(ic_window)
        if len(hist) >= min_obs:
            if ew_halflife:
                ic_mean = hist.ewm(halflife=ew_halflife).mean().iloc[-1]
            else:
                ic_mean = hist.mean()
        else:
            ic_mean = np.nan
        ic_by_date[dt] = ic_mean
        if np.isnan(ic_mean):
            raw_dir[dt] = "neutral"
        elif ic_mean > threshold:
            raw_dir[dt] = "top"
        elif ic_mean < -threshold:
            raw_dir[dt] = "bottom"
        else:
            raw_dir[dt] = "neutral"
    confirmed = backtest.apply_confirm_hysteresis(
        pd.Series(raw_dir, dtype=object), confirm=confirm)

    holdings: list[str] = []
    nav, bench_nav = 1.0, 1.0
    nav_series, bench_series = {}, {}
    turnover_cost_total = 0.0
    turnover_total = 0.0
    regime_log: list[dict] = []
    current_weights = pd.Series(dtype=float)

    all_days = daily_ret.index
    rebal_set = set(ic_by_date)

    for dt in all_days:
        if holdings:
            day_ret = daily_ret.loc[dt, holdings].mean()
            if np.isfinite(day_ret):
                nav *= 1 + day_ret
        bench_day = daily_ret.loc[dt].mean()
        if np.isfinite(bench_day):
            bench_nav *= 1 + bench_day

        if dt in rebal_set:
            s = score_df.loc[dt].dropna()
            s = s[s.index.isin(price_boards)]  # 与决策序列同口径：有分无价的剔除
            ic_mean = ic_by_date[dt]
            direction = confirmed.get(dt, "neutral")

            vol_gate = False
            if vol_series is not None and vol_threshold is not None:
                v = vol_series.get(dt, np.nan)
                vol_gate = pd.notna(v) and v > vol_threshold
            if vol_gate:
                direction = "neutral"

            guard_hit = False
            if mkt_ret is not None and guard is not None:
                m = mkt_ret.get(dt, np.nan)
                guard_hit = pd.notna(m) and m < guard
            if guard_hit:
                direction = "neutral"

            if direction == "top":
                pick = s.nlargest(top_n)
            elif direction == "bottom":
                pick = s.nsmallest(top_n)
            else:
                pick = s  # 等权全部
            new_holdings = pick.index.tolist()
            target_weights = backtest.equal_weight_portfolio(new_holdings)
            if holdings:
                turnover = backtest.portfolio_turnover(current_weights, target_weights)
                tc = cost * turnover
                nav *= 1 - tc
                turnover_cost_total += tc
                turnover_total += turnover
            holdings = new_holdings
            current_weights = target_weights
            regime_log.append({
                "date": dt, "ic": ic_mean, "direction": direction,
                "raw_direction": raw_dir[dt],
                "vol_gate": vol_gate, "guard_hit": guard_hit,
            })
        nav_series[dt] = nav
        bench_series[dt] = bench_nav

    nav_s = pd.Series(nav_series).sort_index()
    bench_s = pd.Series(bench_series).sort_index()
    return {
        "nav": nav_s, "benchmark_nav": bench_s,
        "strategy": backtest.performance_stats(nav_s),
        "benchmark": backtest.performance_stats(bench_s),
        "total_cost_drag": turnover_cost_total,
        "total_turnover": turnover_total,
        "final_nav": nav_s.iloc[-1] if len(nav_s) else np.nan,
        "final_benchmark": bench_s.iloc[-1] if len(bench_s) else np.nan,
        "regime_log": pd.DataFrame(regime_log),
    }


def seg_ret(nav: pd.Series, start: str) -> float:
    """start 之后的区间收益（找不到起点则从头算）。"""
    sub = nav.loc[start:]
    if len(sub) < 2:
        return np.nan
    return sub.iloc[-1] / sub.iloc[0] - 1


def main() -> None:
    ap = argparse.ArgumentParser(description="regime 自适应轮动的预注册配置验证")
    ap.add_argument("--research-grid", action="store_true",
                    help="额外输出研究网格；不会自动改变生产配置")
    args = ap.parse_args()
    cfg = pipeline.load_config()
    print("[1/4] 准备数据（缓存口径，无网络）...")
    data = pipeline.prepare(cfg, end_override=pipeline.cache_end_date(cfg))

    print("[2/4] 计算因子与情绪分 ...")
    factors = compute_all_factors(data["stock_daily"], data["board_map"],
                                  data["board_kline"],
                                  market_total=data.get("market_total"))
    _, smoothed, _ = score.build_scores(
        factors, cfg["weights"], cfg.get("smooth_window", 5),
        lookback=cfg.get("score_lookback", 250),
        z_clip=cfg.get("score_z_clip", 3.0))
    board_close = data["board_kline"].pivot(
        index="date", columns="board_code", values="close").sort_index()

    print("[3/4] 计算 IC 可知序列 ...")
    regime_cfg = cfg.get("regime", {})
    horizon = int(regime_cfg.get("horizon", 5))
    ic_s = daily_ic_series(smoothed, board_close, horizon=horizon)
    ic_known = ic_known_series(ic_s, board_close.index, lag=horizon)
    vol_window = int(regime_cfg.get("vol_window", VOL_WINDOW))
    vol = board_close.pct_change().mean(axis=1).rolling(vol_window).std()

    print("[4/4] 回测（网格 + 对照）...")
    top_n = cfg["signal"].get("top_n", 3)
    cost_bps = cfg["backtest"].get("cost_bps", 5)

    # 固定方向对照（复用主回测口径）
    fixed_top = backtest.rotation_backtest(smoothed, board_close,
                                           top_n=top_n, cost_bps=cost_bps,
                                           direction="top")
    fixed_bottom = backtest.rotation_backtest(smoothed, board_close,
                                              top_n=top_n, cost_bps=cost_bps,
                                              direction="bottom")

    eq_ret = board_close.pct_change().mean(axis=1)
    guard_days = int(regime_cfg.get("guard_days", 10))
    mkt_guard = eq_ret.rolling(guard_days).sum()

    # 默认只验证 config.yaml 中预先声明的策略，避免样本内“跑完后选最佳”。
    configured_kw = {
        "ic_window": int(regime_cfg.get("ic_window", IC_WINDOW_DEFAULT)),
        "threshold": float(regime_cfg.get("threshold", THRESHOLD_DEFAULT)),
        "min_obs": int(regime_cfg.get("min_obs", 10)),
        "mkt_ret": mkt_guard,
        "guard": regime_cfg.get("guard_threshold", -0.03),
        "confirm": int(regime_cfg.get("confirm_readings", 1)),
    }
    if regime_cfg.get("vol_threshold") is not None:
        configured_kw.update({"vol_series": vol,
                              "vol_threshold": float(regime_cfg["vol_threshold"])})
    variants = [("configured", configured_kw)]
    if args.research_grid:
        # 研究网格只生成下一轮候选假设，不能覆盖 production 配置。
        for window in (15, 20, 30, 60, 90):
            for thr in (0.02, 0.03):
                variants.append((f"research_w{window}_t{thr:.2f}",
                                 {"ic_window": window, "threshold": thr,
                                  "min_obs": int(regime_cfg.get("min_obs", 10))}))
        for hl in (5, 10, 20):
            variants.append((f"research_ew{hl}_t0.02",
                             {"ic_window": 90, "threshold": 0.02,
                              "min_obs": int(regime_cfg.get("min_obs", 10)),
                              "ew_halflife": hl}))
        # 确认迟滞候选：在预注册配置（w15/t0.02/护栏）上只改 confirm，
        # 隔离迟滞本身的效果，避免与窗口/阈值选择耦合。
        for cf in (2, 3):
            variants.append((f"research_configured_c{cf}",
                             {**configured_kw, "confirm": cf}))

    grid = []
    results = {}
    for tag, kw in variants:
        res = adaptive_rotation(smoothed, board_close, ic_known,
                                top_n=top_n, cost_bps=cost_bps, **kw)
        results[tag] = res
        grid.append({
            "tag": tag,
            **{f"adaptive_{k}": v for k, v in res["strategy"].items()},
            "adaptive_final": res["final_nav"],
            "flip_seg": seg_ret(res["nav"], "2026-07-01"),
        })

    base = results["configured"]

    # ---------- 输出 ----------
    lines = ["# Regime 自适应轮动回测（方案5验证）", ""]
    d0, d1 = base["nav"].index[0], base["nav"].index[-1]
    rule_txt = ("规则：滚动 IC（决策日仅用已走完前瞻窗口的样本）按 config.yaml `regime` "
                f"的窗口 {configured_kw['ic_window']}、阈值 ±{configured_kw['threshold']:.2f}、"
                f"最少样本 {configured_kw['min_obs']} 判定；中间等权观望。")
    if configured_kw["confirm"] > 1:
        rule_txt += (f"方向确认迟滞：非观望方向需连续 {configured_kw['confirm']} 次"
                     "调仓判定同向才生效，未确认期间观望；风控闸门即时生效。")
    rule_txt += f"周度调仓，单边 100% 换手成本 {cost_bps:g}bps。"
    lines += [f"回测区间：{d0:%Y-%m-%d} ~ {d1:%Y-%m-%d}（含 "
               f"{len(base['nav'])} 个交易日，前段为 IC 预热期，期间等权观望）",
              "", rule_txt, ""]

    heading = ("## 研究网格（仅生成候选假设，不可据此上线）"
               if args.research_grid else "## 预注册配置验证（未做参数选择）")
    lines += [heading, "", "| 组合 | 年化 | 夏普 | 最大回撤 | 期末净值 | 7月以来 |",
               "|---|---|---|---|---|---|"]
    for g in sorted(grid, key=lambda x: -(x["adaptive_sharpe"]
                                           if pd.notna(x["adaptive_sharpe"]) else -9)):
        lines.append(f"| {g['tag']} | {g['adaptive_annual_ret']:.2%} "
                     f"| {g['adaptive_sharpe']:.2f} | {g['adaptive_max_dd']:.2%} "
                     f"| {g['adaptive_final']:.3f} | {g['flip_seg']:+.1%} |")

    best_tag = max(grid, key=lambda x: x["adaptive_sharpe"])["tag"]
    best = results[best_tag]
    rec_tag = "configured"
    rec = base

    lines += ["", "## 对照（同期同口径）", "",
              "| 策略 | 年化 | 夏普 | 最大回撤 | 期末净值 | 7月以来 |",
              "|---|---|---|---|---|---|"]

    def row(label: str, perf: dict, final: float, nav: pd.Series) -> str:
        return (f"| {label} | {perf['annual_ret']:.2%} | {perf['sharpe']:.2f} "
                f"| {perf['max_dd']:.2%} | {final:.3f} | {seg_ret(nav, '2026-07-01'):+.1%} |")

    if args.research_grid:
        lines.append(row(f"研究网格样本内最佳（{best_tag}，不可上线）", best["strategy"],
                         best["final_nav"], best["nav"]))
    lines.append(row(f"预注册配置（{rec_tag}）", rec["strategy"],
                     rec["final_nav"], rec["nav"]))
    lines.append(row("固定追涨 Top3", fixed_top["strategy"], fixed_top["final_nav"], fixed_top["nav"]))
    lines.append(row("固定抄底 Bottom3", fixed_bottom["strategy"], fixed_bottom["final_nav"], fixed_bottom["nav"]))
    lines.append(row("等权基准", base["benchmark"], base["final_benchmark"], base["benchmark_nav"]))
    lines.append("")
    lines.append(f"换仓成本损耗——预注册配置：{rec['total_cost_drag']:.2%}；"
                 f"累计换手 {rec['total_turnover']:.2f}；"
                 f"固定追涨：{fixed_top['total_cost_drag']:.2%}；"
                 f"固定抄底：{fixed_bottom['total_cost_drag']:.2%}")

    # 风格切换只展示预注册配置；研究网格不产生上线建议。
    for tag, res in ((f"预注册配置（{rec_tag}）", rec),):
        log = res["regime_log"].copy()
        log["prev"] = log["direction"].shift()
        switches = log[log["direction"] != log["prev"]].dropna(subset=["prev"])
        lines += ["", f"## 风格切换时间线：{tag}", "",
                  "| 日期 | 滚动IC | 方向 | 闸门 |", "|---|---|---|---|"]
        for _, r in switches.iterrows():
            arrow = {"top": "追涨", "bottom": "抄底", "neutral": "观望"}[r["direction"]]
            ic_txt = f"{r['ic']:+.3f}" if pd.notna(r["ic"]) else "样本不足"
            gate = []
            if r.get("vol_gate"):
                gate.append("波动")
            if r.get("guard_hit"):
                gate.append("回撤")
            lines.append(f"| {r['date']:%Y-%m-%d} | {ic_txt} | {arrow} "
                         f"| {'+'.join(gate)} |")

    lines += ["", "## 结论与限制", "",
              f"1. 本报告只评估预注册配置；累计收益 {rec['strategy']['total_ret']:+.1%}、"
              f"夏普 {rec['strategy']['sharpe']:.2f}、最大回撤 {rec['strategy']['max_dd']:.1%}。"
              "样本期不足半年时不应以年化作为策略选择依据。",
              f"2. 2026-07 以来预注册配置 {seg_ret(rec['nav'], '2026-07-01'):+.1%}；"
              f"固定追涨 {seg_ret(fixed_top['nav'], '2026-07-01'):+.1%}，"
              f"固定抄底 {seg_ret(fixed_bottom['nav'], '2026-07-01'):+.1%}。"
              "这只是一个历史区间，不代表未来切换可预测。",
              "3. 使用 --research-grid 时，样本内最佳只能生成研究假设，不能覆盖 config.yaml；"
              "必须用 P2 稳健性报告和后续未参与选择的时段验证。",
              "4. 成分股使用当前快照、板块等权参考不可交易，ETF 实盘还要另计跟踪误差、"
              "流动性与费用。", ""]
    md = "\n".join(lines)
    print(md)

    out_dir = ROOT / cfg["paths"]["report_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"regime_adaptive_backtest_{pd.Timestamp.today():%Y%m%d}.md"
    path.write_text(md, encoding="utf-8")
    print(f"\n已保存：{path}")


if __name__ == "__main__":
    main()
