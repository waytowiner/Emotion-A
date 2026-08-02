"""回测层：IC 分析、分层测试、Top-N 板块轮动回测。

标的为板块指数本身（复权口径同输入数据）。ETF 与板块的映射关系
留给使用者在报告层自行对照，不在本模块内耦合。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def forward_returns(board_close: pd.DataFrame, days: int) -> pd.DataFrame:
    """板块收盘价的前瞻 days 日收益（T 收盘买入 -> T+days 收盘）。"""
    return board_close.pct_change(days).shift(-days)


def ic_analysis(score: pd.DataFrame, board_close: pd.DataFrame,
                horizons: list[int]) -> pd.DataFrame:
    """各前瞻窗口的 Spearman IC 序列统计：IC 均值、IR、胜率。

    IC = 当日情绪分截面 与 前瞻收益截面 的秩相关。
    """
    rows = []
    for h in horizons:
        fwd = forward_returns(board_close, h)
        common_dates = score.index.intersection(fwd.index)
        ics = []
        for dt in common_dates:
            s, f = score.loc[dt], fwd.loc[dt]
            pair = pd.concat([s, f], axis=1, keys=["s", "f"]).dropna()
            if len(pair) < 10:
                continue
            ics.append(pair["s"].corr(pair["f"], method="spearman"))
        ic = pd.Series(ics)
        rows.append({
            "horizon": h,
            "IC_mean": ic.mean(),
            "IC_std": ic.std(),
            "IR": ic.mean() / ic.std() if ic.std() > 0 else np.nan,
            "IC_win_rate": (ic > 0).mean(),
            "n_obs": len(ic),
        })
    return pd.DataFrame(rows)


def quantile_test(score: pd.DataFrame, board_close: pd.DataFrame,
                  horizon: int = 5, n_groups: int = 5) -> pd.DataFrame:
    """分层测试：按情绪分分 n_groups 组，比较各组前瞻 horizon 日均收益。

    单调递增的分组收益说明情绪分具有区分度。
    """
    fwd = forward_returns(board_close, horizon)
    records = []
    for dt in score.index.intersection(fwd.index):
        pair = pd.concat([score.loc[dt], fwd.loc[dt]], axis=1, keys=["s", "f"]).dropna()
        if len(pair) < n_groups * 3:
            continue
        pair["g"] = pd.qcut(pair["s"], n_groups, labels=False, duplicates="drop")
        records.append(pair.groupby("g")["f"].mean())
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    out = df.mean().rename("mean_fwd_ret").to_frame()
    out["n_obs"] = df.count()
    out.index.name = "quantile"
    return out


def rotation_backtest(score: pd.DataFrame, board_close: pd.DataFrame,
                      *, top_n: int = 3, cost_bps: float = 5.0,
                      freq: str = "W-FRI", direction: str = "top") -> dict:
    """周度轮动：每周最后交易日收盘调仓，等权持有 N 个板块。

    执行口径（T+1）：情绪分收盘后才可得，新持仓从次一交易日起享受收益，
    调仓当日收益仍归旧持仓——不用收盘后的分数赚当天的钱（前视）。
    调仓日取每周真实的最后交易日（周五遇节假日自动提前到周四），
    不会像 resample("W-FRI") 的标签那样落在非交易日导致该周静默不调仓。

    direction: "top"=买入情绪分最高 N 个（追涨），"bottom"=最低 N 个（抄底），
              "avoid_top"=等权持有除最高 N 个外的全部板块（回避亢奋）。
    成本：换仓部分按 cost_bps（万分之）计一次往返成本（卖+买合并计一次，
    口径为"每次换仓成本"而非"单边"）。
    """
    days_with_score = score.dropna(how="all").index
    rebal_dates = (days_with_score.to_series()
                   .groupby(pd.Grouper(freq=freq)).max().dropna())
    daily_ret = board_close.pct_change()
    cost = cost_bps / 10000.0

    holdings: list[str] = []
    nav, bench_nav = 1.0, 1.0
    nav_series, bench_series = {}, {}
    turnover_cost_total = 0.0

    all_days = daily_ret.index
    rebal_set = set(rebal_dates)

    for dt in all_days:
        # 当日收益先按调仓前的持仓结算
        if holdings:
            day_ret = daily_ret.loc[dt, holdings].mean()
            if np.isfinite(day_ret):
                nav *= 1 + day_ret
        bench_day = daily_ret.loc[dt].mean()
        if np.isfinite(bench_day):
            bench_nav *= 1 + bench_day
        # 收盘后用当日情绪分调仓，新持仓次一交易日起生效
        if dt in rebal_set:
            s = score.loc[dt].dropna()
            if len(s) >= top_n:
                if direction == "top":
                    pick = s.nlargest(top_n)
                elif direction == "bottom":
                    pick = s.nsmallest(top_n)
                else:  # avoid_top
                    pick = s.drop(s.nlargest(top_n).index)
                new_holdings = pick.index.tolist()
                changed = len(set(new_holdings) ^ set(holdings)) / 2  # 单边换仓只数
                if holdings:
                    tc = cost * changed / max(len(new_holdings), 1)
                    nav *= 1 - tc
                    turnover_cost_total += tc
                holdings = new_holdings
        nav_series[dt] = nav
        bench_series[dt] = bench_nav

    nav_s = pd.Series(nav_series).sort_index()
    bench_s = pd.Series(bench_series).sort_index()

    def perf(s: pd.Series) -> dict:
        r = s.pct_change().dropna()
        if r.empty:
            return {"annual_ret": np.nan, "sharpe": np.nan, "max_dd": np.nan}
        years = (s.index[-1] - s.index[0]).days / 365.25
        annual = s.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
        sharpe = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan
        dd = (s / s.cummax() - 1).min()
        return {"annual_ret": annual, "sharpe": sharpe, "max_dd": dd}

    return {
        "nav": nav_s,
        "benchmark_nav": bench_s,
        "strategy": perf(nav_s),
        "benchmark": perf(bench_s),
        "total_cost_drag": turnover_cost_total,
        "final_nav": nav_s.iloc[-1] if len(nav_s) else np.nan,
        "final_benchmark": bench_s.iloc[-1] if len(bench_s) else np.nan,
    }


def recent_signal_health(score: pd.DataFrame, board_close: pd.DataFrame,
                         *, days: int = 60, horizon: int = 5) -> dict:
    """最近 days 个交易日的信号健康度：滚动 IC 均值与胜率。

    用于 regime 感知——IC 持续走弱/转负说明当前风格下信号正在失效。
    """
    fwd = forward_returns(board_close, horizon)
    common = score.index.intersection(fwd.index).sort_values()[-days:]
    ics = []
    for dt in common:
        pair = pd.concat([score.loc[dt], fwd.loc[dt]], axis=1,
                         keys=["s", "f"]).dropna()
        if len(pair) < 10:
            continue
        ics.append(pair["s"].corr(pair["f"], method="spearman"))
    ic = pd.Series(ics)
    if ic.empty:
        return {"ic_mean": float("nan"), "ic_win_rate": float("nan"), "n": 0}
    return {"ic_mean": ic.mean(), "ic_win_rate": (ic > 0).mean(), "n": len(ic)}


def format_backtest_report(ic_df: pd.DataFrame, quant_df: pd.DataFrame,
                           rot_top: dict, rot_bottom: dict,
                           rot_avoid: dict) -> str:
    """回测结果 -> Markdown 文本（含追涨/抄底两个方向的轮动对照）。"""
    lines = ["# 板块情绪分回测报告", "", "## IC 分析（Spearman）", "",
             "| 前瞻窗口 | IC均值 | IR | IC胜率 | 样本数 |", "|---|---|---|---|---|"]
    for _, r in ic_df.iterrows():
        lines.append(f"| {int(r['horizon'])}日 | {r['IC_mean']:.4f} | {r['IR']:.3f} "
                     f"| {r['IC_win_rate']:.1%} | {int(r['n_obs'])} |")
    lines += ["", "## 分层测试（5日前瞻收益，按情绪分分组）", "",
              "| 分组 | 平均前瞻收益 | 样本数 |", "|---|---|---|"]
    for g, r in quant_df.iterrows():
        lines.append(f"| Q{int(g) + 1} | {r['mean_fwd_ret']:.3%} | {int(r['n_obs'])} |")

    def rot_rows(rot: dict) -> list[str]:
        s, b = rot["strategy"], rot["benchmark"]
        return [
            f"| 年化收益 | {s['annual_ret']:.2%} | {b['annual_ret']:.2%} |",
            f"| 夏普 | {s['sharpe']:.2f} | {b['sharpe']:.2f} |",
            f"| 最大回撤 | {s['max_dd']:.2%} | {b['max_dd']:.2%} |",
            f"| 期末净值 | {rot['final_nav']:.3f} | {rot['final_benchmark']:.3f} |",
        ]

    lines += ["", "## 周度轮动：买入情绪 Top N（追涨）", "",
              "| 指标 | 策略 | 等权基准 |", "|---|---|---|"] + rot_rows(rot_top)
    lines += ["", "## 周度轮动：买入情绪 Bottom N（抄底）", "",
              "| 指标 | 策略 | 等权基准 |", "|---|---|---|"] + rot_rows(rot_bottom)
    lines += ["", "## 周度轮动：回避情绪 Top N（持有其余全部）", "",
              "| 指标 | 策略 | 等权基准 |", "|---|---|---|"] + rot_rows(rot_avoid)
    lines += ["", f"换仓成本损耗——追涨：{rot_top['total_cost_drag']:.2%}；"
              f"抄底：{rot_bottom['total_cost_drag']:.2%}；"
              f"回避：{rot_avoid['total_cost_drag']:.2%}",
              "",
              "> 回测有效 ≠ 未来有效；IC 均值 |0.03| 以上、分层单调才说明信号有区分度。"]
    return "\n".join(lines)
