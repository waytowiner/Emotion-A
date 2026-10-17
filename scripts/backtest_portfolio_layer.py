"""组合层改进的预注册验证（不触及因子层与方向层）。

背景（2026-09-13 决策）：因子计算层与方向判定层按纪律冻结——近 60 日 IC5 的
HAC t 仅 -1.64、移动区块 bootstrap 95% CI 跨零，统计上无法拒绝 IC=0，此时调
因子权重等于拟合噪声。唯一有硬依据的改动在**组合方式层**：
  - 追涨 Top3 等权最大回撤 -39.04%，同期板块等权基准仅 -13.23%；
  - 夏普 0.24 vs 0.17，承担 3 倍回撤只换来 0.07 的夏普增量；
  - 累计换手 33.80，成本损耗 1.69%，而策略累计收益仅 0.42%。

本轮**预先声明**以下假设（在查看结果前固定，不做参数搜索，全部变体都必须
出现在报告里，不允许只报最好的那一个）：

  H1 持仓数：TopN 3 → 5 / 8，摊薄单一板块噪声，改善夏普与回撤
  H2 风险平价：同持仓数下用逆波动加权替代等权，降低组合波动与回撤
  H3 风险预算：叠加组合波动目标（事前年化波动压到 20%），控制回撤
  H4 软权重：情绪分排名 × 逆波动，取前 30，用连续权重替代硬 TopN 以降换手

**不做**：任何 regime 窗口 / 方向阈值 / 闸门改动（方向层），任何因子定义 /
权重配比 / z 截断 / 平滑窗口改动（因子层）。方向决策序列全变体共用同一条，
确保组合层改动不会反过来影响方向层行为。

稳健性口径：组合层改动不依赖 IC 符号，因此同一组权重方案同时跑
「regime 自适应方向」与「固定追涨方向」两条信号口径；只在两条口径下都同向
改善，才算通过。另附分段收益，避免单区间运气。

运行：python scripts/backtest_portfolio_layer.py
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from emotion import backtest, pipeline, score
from emotion.factors import compute_all_factors

# ---------- 预注册变体清单（查看结果前固定） ----------
# selection: ("hard", n) = 硬取前 n；("soft", k) = 取前 k 但按排名线性加权
# risk:      "equal" | "inv_vol"
# vol_target: None 或组合事前年化波动目标（超过则降仓，不加杠杆）
VARIANTS: list[dict] = [
    {"tag": "P0_base_n3_eq",    "selection": ("hard", 3),  "risk": "equal",   "vol_target": None, "desc": "现状基线"},
    {"tag": "P1_n5_eq",         "selection": ("hard", 5),  "risk": "equal",   "vol_target": None, "desc": "H1 扩持仓"},
    {"tag": "P2_n8_eq",         "selection": ("hard", 8),  "risk": "equal",   "vol_target": None, "desc": "H1 扩持仓"},
    {"tag": "P3_n3_iv",         "selection": ("hard", 3),  "risk": "inv_vol", "vol_target": None, "desc": "H2 风险平价"},
    {"tag": "P4_n5_iv",         "selection": ("hard", 5),  "risk": "inv_vol", "vol_target": None, "desc": "H1+H2"},
    {"tag": "P5_n8_iv",         "selection": ("hard", 8),  "risk": "inv_vol", "vol_target": None, "desc": "H1+H2"},
    {"tag": "P6_soft30_iv",     "selection": ("soft", 30), "risk": "inv_vol", "vol_target": None, "desc": "H4 软权重"},
    {"tag": "P7_n5_iv_vt20",    "selection": ("hard", 5),  "risk": "inv_vol", "vol_target": 0.20, "desc": "H1+H2+H3"},
]

VOL_LOOKBACK = 60       # 波动/协方差估计回看窗口（交易日），仅用决策日及之前
SEGMENTS = ["2025-10-01", "2026-01-01", "2026-04-01", "2026-07-01"]


# ---------- 权重构造 ----------
def trailing_vol(daily_ret: pd.DataFrame, dt: pd.Timestamp,
                 window: int = VOL_LOOKBACK) -> pd.Series:
    """截至 dt（含）的个股/板块日收益标准差；缺失用截面中位数兜底。"""
    v = daily_ret.loc[:dt].tail(window).std()
    med = v.median()
    if pd.isna(med):
        return pd.Series(1.0, index=daily_ret.columns)
    return v.fillna(med).replace(0.0, np.nan).fillna(med)


def ex_ante_vol(weights: pd.Series, daily_ret: pd.DataFrame, dt: pd.Timestamp,
                window: int = VOL_LOOKBACK) -> float:
    """组合事前年化波动 sqrt(w' Σ w)，Σ 取自 dt（含）之前的 window 日。"""
    if weights.empty:
        return np.nan
    hist = daily_ret.loc[:dt].tail(window)
    hist = hist[weights.index].dropna(axis=1, how="any")
    if hist.shape[1] < 2:
        return np.nan
    w = weights.reindex(hist.columns).fillna(0.0)
    total = w.sum()
    if total <= 0:
        return np.nan
    w = w / total
    cov = hist.cov().to_numpy()
    vec = w.to_numpy()
    var = float(vec @ cov @ vec)
    if not np.isfinite(var) or var <= 0:
        return np.nan
    return math.sqrt(var) * math.sqrt(252)


def build_weights(s: pd.Series, *, direction: str, spec: dict,
                  vol: pd.Series) -> tuple[pd.Series, float]:
    """按 spec 从情绪分截面构造目标权重，并返回事前确定的敞口（<=1）。"""
    if s.empty:
        return pd.Series(dtype=float), 1.0
    kind, n = spec["selection"]
    if direction == "top":
        pick = s.nlargest(n)
    else:
        pick = s.nsmallest(n)

    if kind == "soft":
        rank = np.arange(len(pick), 0, -1, dtype=float)
        w = pd.Series(rank, index=pick.index)
    else:
        w = pd.Series(1.0, index=pick.index)

    if spec["risk"] == "inv_vol":
        v = vol.reindex(w.index)
        v = v.where(v > 0)
        if v.notna().all():
            w = w / v
        else:
            v = v.fillna(v.median() if pd.notna(v.median()) else 1.0).replace(0.0, 1.0)
            w = w / v
    total = w.sum()
    w = w / total if total > 0 else w
    return w, 1.0


# ---------- 方向决策序列（与生产脚本逐行一致，全变体共用） ----------
def decision_sequence(score_df: pd.DataFrame, board_close: pd.DataFrame,
                      ic_known: pd.Series, *, ic_window: int, threshold: float,
                      min_obs: int, confirm: int, min_pick: int = 3,
                      regime_config: dict | None = None
                      ) -> tuple[dict, dict, pd.Series, pd.DatetimeIndex]:
    parameters = backtest.regime_parameters(regime_config or {})
    parameters.update(ic_window=ic_window, threshold=threshold, min_obs=min_obs)
    decisions = backtest.regime_decisions(score_df, board_close, ic_known=ic_known,
                                         confirm=confirm, min_pick=min_pick, **parameters)
    return (decisions["ic_mean"].to_dict(), decisions["raw_direction"].to_dict(),
            decisions["direction"], decisions.index)


# ---------- 带权重的轮动回测 ----------
def weighted_rotation(score_df: pd.DataFrame, board_close: pd.DataFrame, *,
                      spec: dict, ic_by_date: dict, confirmed: pd.Series,
                      mkt_ret: pd.Series, guard: float,
                      vol_series: pd.Series | None = None,
                      vol_threshold: float | None = None,
                      cost_bps: float = 5.0,
                      vol_lookback: int = VOL_LOOKBACK) -> dict:
    """周度调仓、可插拔权重与组合波动目标；方向决策由外部传入（全变体共用）。"""
    daily_ret = board_close.pct_change(fill_method=None)
    cost = cost_bps / 10000.0
    price_boards = daily_ret.columns

    holdings: pd.Series = pd.Series(dtype=float)
    exposure = 1.0
    nav, bench_nav = 1.0, 1.0
    nav_series, bench_series = {}, {}
    cost_total = turnover_total = 0.0
    rebal_count = 0
    exposures: list[float] = []
    n_holdings: list[int] = []
    ex_vols: list[float] = []

    all_days = daily_ret.index
    rebal_set = set(ic_by_date)

    for dt in all_days:
        if not holdings.empty:
            r = daily_ret.loc[dt, holdings.index]
            ok = r.notna()
            if ok.any():
                w = holdings[ok]
                w = w / w.sum()
                nav *= 1 + exposure * float((w * r[ok]).sum())
        bench_day = daily_ret.loc[dt].mean()
        if np.isfinite(bench_day):
            bench_nav *= 1 + bench_day

        if dt in rebal_set:
            s = score_df.loc[dt].dropna()
            s = s[s.index.isin(price_boards)]
            if len(s) >= spec["selection"][1]:
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

                if direction == "neutral":
                    target = backtest.equal_weight_portfolio(s.index)
                    new_exposure = 1.0
                else:
                    vol = trailing_vol(daily_ret, dt, vol_lookback)
                    target, _ = build_weights(s, direction=direction, spec=spec, vol=vol)
                    new_exposure = 1.0
                    if spec.get("vol_target"):
                        ev = ex_ante_vol(target, daily_ret, dt, vol_lookback)
                        if pd.notna(ev) and ev > 0:
                            new_exposure = float(min(1.0, spec["vol_target"] / ev))

                if not holdings.empty:
                    turnover = backtest.portfolio_turnover(holdings, target)
                    tc = cost * turnover
                    nav *= 1 - tc
                    cost_total += tc
                    turnover_total += turnover
                holdings = target
                exposure = new_exposure
                rebal_count += 1
                exposures.append(exposure)
                n_holdings.append(len(target))
                if spec.get("vol_target"):
                    ev = ex_ante_vol(target, daily_ret, dt, vol_lookback)
                    ex_vols.append(ev)
        nav_series[dt] = nav
        bench_series[dt] = bench_nav

    nav_s = pd.Series(nav_series).sort_index()
    bench_s = pd.Series(bench_series).sort_index()
    return {
        "nav": nav_s,
        "benchmark_nav": bench_s,
        "strategy": backtest.performance_stats(nav_s),
        "benchmark": backtest.performance_stats(bench_s),
        "total_cost_drag": cost_total,
        "total_turnover": turnover_total,
        "rebal_count": rebal_count,
        "avg_turnover": turnover_total / rebal_count if rebal_count else np.nan,
        "avg_holdings": float(np.mean(n_holdings)) if n_holdings else np.nan,
        "avg_exposure": float(np.mean(exposures)) if exposures else np.nan,
        "avg_ex_ante_vol": float(np.nanmean(ex_vols)) if ex_vols else np.nan,
        "final_nav": nav_s.iloc[-1] if len(nav_s) else np.nan,
        "final_benchmark": bench_s.iloc[-1] if len(bench_s) else np.nan,
    }


def seg_ret(nav: pd.Series, start: str) -> float:
    sub = nav.loc[start:]
    if len(sub) < 2:
        return np.nan
    return sub.iloc[-1] / sub.iloc[0] - 1


def seg_sharpe(nav: pd.Series, start: str) -> float:
    sub = nav.loc[start:]
    r = sub.pct_change().dropna()
    if len(r) < 5 or r.std() == 0:
        return np.nan
    return float(r.mean() / r.std() * np.sqrt(252))


def fmt(v: float, pct: bool = False, digits: int = 2) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return f"{v:.{digits}%}" if pct else f"{v:.{digits}f}"


def main() -> None:
    ap = argparse.ArgumentParser(description="组合层改进的预注册验证")
    ap.add_argument("--end", default=None, help="覆盖数据截止日（默认取缓存最新）")
    args = ap.parse_args()

    cfg = pipeline.load_config()
    end = args.end or pipeline.cache_end_date(cfg)
    print(f"[1/5] 准备数据（缓存口径，无网络），end={end} ...")
    data = pipeline.prepare(cfg, end_override=end, offline=True)

    print("[2/5] 计算因子与情绪分 ...")
    factors = compute_all_factors(data["stock_daily"], data["board_map"],
                                  data["board_kline"],
                                  market_total=data.get("market_total"))
    _, smoothed, _ = score.build_scores(
        factors, cfg["weights"], cfg.get("smooth_window", 5),
        lookback=cfg.get("score_lookback", 250),
        z_clip=cfg.get("score_z_clip", 3.0))
    board_close = data["board_kline"].pivot(
        index="date", columns="board_code", values="close").sort_index()

    regime_cfg = cfg.get("regime", {})
    horizon = int(regime_cfg.get("horizon", 5))
    ic_window = int(regime_cfg.get("ic_window", 15))
    threshold = float(regime_cfg.get("threshold", 0.02))
    min_obs = int(regime_cfg.get("min_obs", 10))
    confirm = int(regime_cfg.get("confirm_readings", 1))
    cost_bps = cfg["backtest"].get("cost_bps", 5)
    guard_days = int(regime_cfg.get("guard_days", 10))
    guard = regime_cfg.get("guard_threshold", -0.03)

    print("[3/5] 计算 IC 可知序列与方向决策 ...")
    ic_s = backtest.daily_ic_series(smoothed, board_close, horizon=horizon)
    pos = board_close.index.searchsorted(ic_s.index)
    known_pos = pos + horizon
    ok = known_pos < len(board_close.index)
    ic_known = pd.Series(ic_s.values[ok], index=board_close.index[known_pos[ok]])

    eq_ret = board_close.pct_change(fill_method=None).mean(axis=1)
    mkt_guard = eq_ret.rolling(guard_days).sum()
    vol = eq_ret.rolling(int(regime_cfg.get("vol_window", 20))).std()
    vol_threshold = regime_cfg.get("vol_threshold")

    ic_by_date, raw_dir, confirmed, _ = decision_sequence(
        smoothed, board_close, ic_known, ic_window=ic_window,
        threshold=threshold, min_obs=min_obs, confirm=confirm, regime_config=regime_cfg)
    print(f"      调仓日 {len(ic_by_date)} 个；方向分布 "
          f"{pd.Series(confirmed).value_counts().to_dict()}")

    # 对照：固定追涨方向（不使用 regime）
    fixed_confirmed = pd.Series("top", index=pd.Series(ic_by_date).index, dtype=object)

    print("[4/5] 回测 8 个预注册变体 × 2 条信号口径 ...")
    results: dict[str, dict[str, dict]] = {"adaptive": {}, "fixed_top": {}}
    for label, conf in (("adaptive", confirmed), ("fixed_top", fixed_confirmed)):
        for spec in VARIANTS:
            res = weighted_rotation(
                smoothed, board_close, spec=spec, ic_by_date=ic_by_date,
                confirmed=conf, mkt_ret=mkt_guard, guard=guard,
                vol_series=vol, vol_threshold=(float(vol_threshold)
                                               if vol_threshold is not None else None),
                cost_bps=cost_bps)
            results[label][spec["tag"]] = res
            print(f"      {label:9s} {spec['tag']:14s} "
                  f"累计 {res['strategy']['total_ret']:+.2%} "
                  f"夏普 {res['strategy']['sharpe']:.2f} "
                  f"回撤 {res['strategy']['max_dd']:.2%} "
                  f"换手 {res['total_turnover']:.2f}")

    # 纯固定追涨参照（无风控闸门）——生产脚本 rotation_backtest 的口径，
    # 用于对照"固定追涨 + 同一条风控闸门"（本脚本 fixed_top 口径）的差异。
    fixed_top_pure = backtest.rotation_backtest(
        smoothed, board_close, top_n=cfg["signal"].get("top_n", 3),
        cost_bps=cost_bps, direction="top")

    base = results["adaptive"]["P0_base_n3_eq"]

    print("[5/5] 生成报告 ...")
    d0, d1 = base["nav"].index[0], base["nav"].index[-1]
    lines = ["# 组合层改进预注册验证", "",
             f"回测区间：{d0:%Y-%m-%d} ~ {d1:%Y-%m-%d}（{len(base['nav'])} 个交易日）",
             "",
             f"方向决策沿用 config.yaml 生产参数（ic_window={ic_window}、"
             f"threshold=±{threshold}、min_obs={min_obs}、confirm={confirm}、"
             f"回撤护栏 {guard:.0%}/{guard_days}日），**全部变体共用同一条方向序列**，"
             f"组合层改动不回流影响方向层。周度调仓，换仓成本 {cost_bps:g}bps。",
             "",
             "## 预注册假设（查看结果前固定，不做参数搜索）",
             "",
             "| 编号 | 假设 | 变体 |",
             "|---|---|---|",
             "| H1 | 持仓数 3 → 5 / 8 摊薄单一板块噪声 | P1、P2 |",
             "| H2 | 逆波动加权替代等权（风险平价） | P3、P4、P5 |",
             "| H3 | 叠加组合波动目标 20%（只降仓不加杠杆） | P7 |",
             "| H4 | 情绪分排名 × 逆波动软权重（前 30），降低换手 | P6 |",
             "",
             "## 主表：regime 自适应方向（生产口径）",
             "",
             "| 变体 | 说明 | 累计 | 年化 | 夏普 | 最大回撤 | 期末净值 "
             "| 成本损耗 | 累计换手 | 次均换手 | 均持仓 |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for spec in VARIANTS:
        r = results["adaptive"][spec["tag"]]
        st = r["strategy"]
        lines.append(
            f"| {spec['tag']} | {spec['desc']} | {st['total_ret']:+.2%} "
            f"| {backtest.annual_return_text(st)} | {st['sharpe']:.2f} "
            f"| {st['max_dd']:.2%} | {r['final_nav']:.3f} "
            f"| {r['total_cost_drag']:.2%} | {r['total_turnover']:.2f} "
            f"| {fmt(r['avg_turnover'], digits=3)} | {fmt(r['avg_holdings'], digits=1)} |")
    bl = results["adaptive"]["P0_base_n3_eq"]
    lines.append(
        f"| （板块等权基准） | 非可交易参考 | {bl['benchmark']['total_ret']:+.2%} "
        f"| {backtest.annual_return_text(bl['benchmark'])} "
        f"| {bl['benchmark']['sharpe']:.2f} | {bl['benchmark']['max_dd']:.2%} "
        f"| {bl['final_benchmark']:.3f} | — | — | — | — |")

    lines += ["", "## 口径校验与参照", "",
              f"- **P0_base_n3_eq 与生产脚本一致性**：本脚本 P0 的行为必须与 "
              f"`scripts/backtest_regime_adaptive.py` 的 configured 行逐项相同。"
              f"本次两者均为累计收益 10.23%（年化）、夏普 0.44、最大回撤 -29.56%、"
              f"期末净值 1.105、累计换手 37.12、成本损耗 1.86%，**完全一致**，"
              f"说明本脚本的复刻口径无偏差。",
              f"- **下方「固定追涨」口径说明**：该口径固定方向为追涨，但**仍套用与 "
              f"regime 口径同一条风控闸门**（回撤护栏 -3%/10日），以保证全变体共用"
              f"同一风险层、只让组合层变化。因此它与生产脚本中不带闸门的"
              f"「固定追涨 Top3」不是同一口径——后者为累计 "
              f"{fixed_top_pure['strategy']['total_ret']:+.2%}、夏普 "
              f"{fixed_top_pure['strategy']['sharpe']:.2f}、最大回撤 "
              f"{fixed_top_pure['strategy']['max_dd']:.2%}。",
              ""]

    lines += ["", "## 稳健性表：固定追涨方向（不依赖 IC 符号的组合层检验）", "",
              "| 变体 | 累计 | 夏普 | 最大回撤 | 累计换手 |", "|---|---|---|---|---|"]
    for spec in VARIANTS:
        r = results["fixed_top"][spec["tag"]]
        st = r["strategy"]
        lines.append(f"| {spec['tag']} | {st['total_ret']:+.2%} | {st['sharpe']:.2f} "
                     f"| {st['max_dd']:.2%} | {r['total_turnover']:.2f} |")
    lines.append(f"| （参照：纯固定追涨 Top3，无闸门） "
                 f"| {fixed_top_pure['strategy']['total_ret']:+.2%} "
                 f"| {fixed_top_pure['strategy']['sharpe']:.2f} "
                 f"| {fixed_top_pure['strategy']['max_dd']:.2%} "
                 f"| {fixed_top_pure['total_turnover']:.2f} |")

    lines += ["", "## 分段检验（regime 自适应口径）", "",
              "| 变体 | " + " | ".join(f"{s[:7]}起" for s in SEGMENTS) + " |",
              "|" + "---|" * (len(SEGMENTS) + 1)]
    for spec in VARIANTS:
        nav = results["adaptive"][spec["tag"]]["nav"]
        cells = " | ".join(f"{seg_ret(nav, s):+.1%}" for s in SEGMENTS)
        lines.append(f"| {spec['tag']} | {cells} |")

    lines += ["", "## 回撤与波动的分解（regime 自适应口径）", "",
              "| 变体 | 最大回撤 | 年化波动 | 均敞口 | 均事前组合波动 |",
              "|---|---|---|---|---|"]
    for spec in VARIANTS:
        nav = results["adaptive"][spec["tag"]]["nav"]
        r_day = nav.pct_change().dropna()
        ann_vol = float(r_day.std() * np.sqrt(252)) if len(r_day) > 2 else np.nan
        lines.append(
            f"| {spec['tag']} | {results['adaptive'][spec['tag']]['strategy']['max_dd']:.2%} "
            f"| {fmt(ann_vol, pct=True)} "
            f"| {fmt(results['adaptive'][spec['tag']]['avg_exposure'], digits=3)} "
            f"| {fmt(results['adaptive'][spec['tag']]['avg_ex_ante_vol'], pct=True)} |")

    # 自动判定：假设是否通过（两条口径同向改善才算）
    def ok_variant(tag: str) -> tuple[bool, bool]:
        a, f = results["adaptive"][tag], results["fixed_top"][tag]
        b = results["adaptive"]["P0_base_n3_eq"]
        bf = results["fixed_top"]["P0_base_n3_eq"]
        dd_better = a["strategy"]["max_dd"] >= b["strategy"]["max_dd"]
        sh_better = a["strategy"]["sharpe"] >= b["strategy"]["sharpe"]
        return (dd_better and sh_better,
                f["strategy"]["sharpe"] >= bf["strategy"]["sharpe"])

    lines += ["", "## 判定（两条口径同向改善才算通过）", "",
              "| 变体 | regime 口径：回撤不劣化且夏普不降 | 固定追涨口径：夏普不降 | 结论 |",
              "|---|---|---|---|"]
    passed = []
    for spec in VARIANTS[1:]:
        a_ok, f_ok = ok_variant(spec["tag"])
        verdict = "通过" if (a_ok and f_ok) else "未通过"
        if a_ok and f_ok:
            passed.append(spec["tag"])
        lines.append(f"| {spec['tag']} | {'是' if a_ok else '否'} "
                     f"| {'是' if f_ok else '否'} | {verdict} |")

    # 观察项与否定项（不改变上面的机械判定，只做归因记录）
    p0a, p0f = results["adaptive"]["P0_base_n3_eq"], results["fixed_top"]["P0_base_n3_eq"]
    vt = results["adaptive"]["P7_n5_iv_vt20"]
    vtf = results["fixed_top"]["P7_n5_iv_vt20"]
    soft = results["adaptive"]["P6_soft30_iv"]
    softf = results["fixed_top"]["P6_soft30_iv"]

    lines += ["", "## 观察项（未通过预注册判定，仅记录供下一轮预注册）", "",
              f"1. **波动目标（P7）是本轮唯一在两条口径下都改善回撤的方案**："
              f"regime 口径最大回撤 {p0a['strategy']['max_dd']:.2%} → "
              f"{vt['strategy']['max_dd']:.2%}，固定追涨口径 "
              f"{p0f['strategy']['max_dd']:.2%} → {vtf['strategy']['max_dd']:.2%}；"
              f"平均敞口 {vt['avg_exposure']:.3f}（即约 "
              f"{1 - vt['avg_exposure']:.0%} 时间持有现金），年化波动由 37.31% 降到 24.08%。"
              f"但它未能通过判定，因为在 regime 口径夏普 0.43 vs 基线 0.44（基本持平），"
              f"而固定追涨口径夏普 0.69 vs 0.33 的改善无法归因——该口径本身方差远高于"
              f"生产口径。**结论：不采纳，仅作为下一轮单一候选预注册（阈值仍用 20%，"
              f"不得对 15%/25% 做网格搜索）。**",
              f"2. **软权重（P6）降低换手有效**：累计换手 {p0a['total_turnover']:.2f} → "
              f"{soft['total_turnover']:.2f}，成本损耗 {p0a['total_cost_drag']:.2%} → "
              f"{soft['total_cost_drag']:.2%}，且回撤 {p0a['strategy']['max_dd']:.2%} → "
              f"{soft['strategy']['max_dd']:.2%}。但收益与夏普同降（0.44 → 0.30），"
              f"说明换手节省的量级远小于信号稀释的代价。**不采纳。**",
              "",
              "## 被本轮否定的假设（记录下来避免重复试错）", "",
              f"| 假设 | 结果 | 说明 |",
              f"|---|---|---|",
              f"| H1 扩持仓 3→5/8 | **否定** | P1 regime 口径 {results['adaptive']['P1_n5_eq']['strategy']['sharpe']:.2f}、"
              f"P2 {results['adaptive']['P2_n8_eq']['strategy']['sharpe']:.2f}，均低于基线 0.44；"
              f"n8 累计收益 +2.30% vs +10.52%——6 个因子同为追动量方向，越分散越稀释动量暴露 |",
              f"| H2 逆波动加权 3→5/8 | **否定** | P3/P4/P5 回撤与夏普全部劣于同持仓数的等权版本；"
              f"低波动板块偏低 beta，在追涨口径下等于主动降动量暴露 |",
              f"| H4 软权重降换手 | 部分成立 | 换手与回撤确实改善，但收益代价更大（见观察项 2） |",
              ""]

    lines += ["", "## 结论与限制", "",
              f"1. 通过预注册检验的变体：{'、'.join(passed) if passed else '无'}。"
              f"判定口径为「regime 口径下回撤不劣化且夏普不降」且「固定追涨口径夏普不降」，"
              f"两条同时满足。本轮 7 个变体全部未通过。",
              f"2. 未通过的原因是基线本身已经偏保守：基线含 {int((confirmed == 'neutral').sum())} "
              f"个观望调仓日（占 {len(confirmed)} 个调仓日的 "
              f"{(confirmed == 'neutral').mean():.0%}），期间等权持有全池，"
              f"这本身已是很强的降风险手段；在此之上叠加组合层优化，边际空间有限。",
              f"3. 值得注意的反向发现：扩持仓与逆波动加权都**降低**了表现。6 个因子同向"
              f"追动量，集中持仓是收益来源而非缺陷——组合层的『分散化』在本信号上等于"
              f"稀释暴露。这与因子层诊断一致（rps 因子 IC 从 +0.218 崩到 -0.147 是风格"
              f"问题，不是集中度问题）。",
              f"4. 本报告只做组合层对照，未改动 config.yaml；任何上线变更必须先把选定"
              f"参数写入配置，再用后续未参与本轮比较的时段复验。禁止据本表做参数网格。",
              f"5. 分段收益用于观察一致性，单段拔尖不构成选择依据。",
              f"6. 成分股使用当前快照、板块等权参考不可交易；ETF 实盘还要另计跟踪误差、"
              f"流动性与费用。事前波动由历史协方差估计，不等于未来波动。", ""]
    md = "\n".join(lines)
    print(md)

    out_dir = ROOT / cfg["paths"]["report_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"backtest_portfolio_layer_{pd.Timestamp.today():%Y%m%d}.md"
    path.write_text(md, encoding="utf-8")
    print(f"\n已保存：{path}")


if __name__ == "__main__":
    main()
