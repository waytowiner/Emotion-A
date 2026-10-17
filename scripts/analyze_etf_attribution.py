"""固定上一轮ETF输入，进行同池对照、执行期归因和决策层诊断；不优化参数。"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from emotion import attribution, backtest, etf_backtest  # noqa: E402
from scripts.backtest_etf_prices import price_panel, result_stats  # noqa: E402

LABELS = {
    "equal_weight": "同池等权", "fixed_top": "固定Top3", "fixed_bottom": "固定Bottom3",
    "adaptive": "当前自适应（映射池）", "original": "当前自适应（原排名缺口现金）",
    "raw_ic": "诊断：原始IC方向", "confirmed_no_gates": "诊断：确认后、未应用闸门",
    "gated_no_confirm": "诊断：闸门后、未应用确认",
    "balanced_active": "诊断：方向期Top/Bottom各半", "cash_matched": "诊断：等权匹配原排名仓位",
    "csi300": "沪深300ETF买入持有",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(frame, output, name, index=True):
    frame.to_csv(output / f"{name}.csv", encoding="utf-8-sig", index=index)


def rank_diagnostics(scores, mapping, panels, periods):
    mapping = mapping.set_index("board_code").etf_code
    known = panels["close"].ffill()
    records = []
    for date, period in periods.iterrows():
        values = scores.loc[date].dropna()
        values = values[values.index.isin(mapping.index)]
        codes = mapping.reindex(values.index)
        valid = [board for board in values.index if panels["close"].at[date, codes[board]] > 0
                 and panels["volume"].at[date, codes[board]] > 0]
        values = values.loc[valid]
        codes = codes.loc[valid]
        opening = panels["open"].loc[period.entry_date, codes].to_numpy()
        endpoint = (known.loc[period.exit_date] if period.exit_mark == "close" else
                    panels["open"].loc[period.exit_date].combine_first(known.shift().loc[period.exit_date]))
        returns = pd.Series(endpoint.loc[codes].to_numpy() / opening - 1, index=values.index)
        if returns.isna().any():
            raise ValueError("排名诊断缺少真实执行价格")
        buckets = pd.qcut(values, 5, labels=False, duplicates="drop")
        groups = returns.groupby(buckets).mean()
        records.append({"signal_date": date, "n_etfs": len(values),
                        "spearman_ic": values.corr(returns, method="spearman"),
                        **{f"Q{i+1}": groups.get(i, np.nan) for i in range(5)},
                        "Q5_minus_Q1": groups.get(4, np.nan) - groups.get(0, np.nan)})
    return pd.DataFrame(records).set_index("signal_date")


def holding_contributions(result, panels, periods, account, mapping):
    """逐ETF盈亏以初始本金为单位，可与逐笔成本直接相加核对总盈亏。"""
    known = panels["close"].ffill()
    names = mapping.set_index("etf_code").etf_name.to_dict()
    records = []
    for signal, period in periods.iterrows():
        entry, ending = period.entry_date, period.exit_date
        closing_value = result["holdings"].loc[entry] * result["daily"].at[entry, "nav"]
        codes = closing_value.index[closing_value > 1e-12]
        shares = closing_value[codes] / known.loc[entry, codes]
        start_price = panels["open"].loc[entry, codes]
        end_price = (known.loc[ending, codes] if period.exit_mark == "close" else
                     panels["open"].loc[ending, codes].combine_first(known.shift().loc[ending, codes]))
        pnl = shares * (end_price - start_price)
        costs = account.at[signal, "entry_fee"] + account.at[signal, "entry_slippage"]
        np.testing.assert_allclose(pnl.sum() - costs, account.at[signal, "pnl_initial_equity"], atol=1e-10)
        for code in codes:
            records.append({"signal_date": signal, "entry_date": entry, "exit_date": ending,
                            "etf_code": code, "etf_name": names.get(code, code),
                            "pnl_initial_equity": pnl[code],
                            "contribution_period_return": pnl[code] / account.at[signal, "opening_equity"]})
    return pd.DataFrame(records)


def concentration_diagnostics(selections, panels, periods, mapping):
    """方向期持仓间的历史收益相关性，仅使用信号日以前最多60日。"""
    selected = selections[(selections.strategy == "adaptive") & selections.direction.ne("neutral")
                          & selections.signal_date.isin(periods.index)]
    names = mapping.set_index("etf_code").etf_name.to_dict()
    returns = panels["close"].pct_change(fill_method=None)
    records = []
    for date, group in selected.groupby("signal_date"):
        for left, right in combinations(group.etf_code.unique(), 2):
            pair = returns.loc[:date, [left, right]].tail(60).dropna()
            records.append({"signal_date": date, "etf_a": left, "name_a": names[left],
                            "etf_b": right, "name_b": names[right], "known_observations": len(pair),
                            "past_correlation": pair[left].corr(pair[right]) if len(pair) >= 10 else np.nan})
    return pd.DataFrame(records)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=ROOT / "report/etf_adaptive_20261009")
    ap.add_argument("--output", type=Path, default=ROOT / "report/etf_attribution_20261009")
    args = ap.parse_args()
    source, output = args.source, args.output
    output.mkdir(parents=True, exist_ok=True)
    meta = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    config = meta["parameters"]
    config_path = ROOT / "config.yaml"
    assert digest(config_path) == meta["config_sha256"], "策略配置已变化"
    mapping_path = ROOT / "data/cache/etf_trading_map.parquet"
    assert digest(mapping_path) == meta["mapping_sha256"], "ETF映射已变化"
    for code, sha in meta["price_hashes"].items():
        assert digest(ROOT / "data/cache" / f"etf_kline_{code}.parquet") == sha, code
    scores = pd.read_parquet(source / "scores.parquet")
    decisions = pd.read_csv(source / "regime_decisions.csv", index_col="date", parse_dates=["date"])
    mapping = pd.read_parquet(mapping_path)
    panels, coverage = price_panel(mapping, scores.index)
    periods = attribution.execution_periods(scores.index, decisions.index)
    # 计算前固定实验清单，不依据结果改变窗口、阈值、N或门控定义。
    plan = {"source": str(source), "config_sha256": meta["config_sha256"],
            "source_scores_sha256": digest(source / "scores.parquet"),
            "source_decisions_sha256": digest(source / "regime_decisions.csv"),
            "strategies": LABELS, "cost_modes": ["gross", "fee_only", "net"],
            "fee_bps": meta["fee_bps_each_side"], "slippage_bps": meta["slippage_bps_each_side"],
            "same_pool": "signal-day nonmissing score AND positive ETF close/volume; frozen mapping",
            "decision_diagnostics": "saved raw/confirmed/state/final directions; no parameter search",
            "attribution": "next-session pre-trade open to next execution pre-trade open, final close",
            "timing_test": "all circular shifts of executed direction sequence; descriptive, not p-value",
            "bootstrap": "4 consecutive execution periods, 1000 samples, fixed seed",
            "parameters": config}
    (output / "experiment_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[1/5] 参数/行情/映射冻结；建立相同信号日可用池", flush=True)
    targets = {}
    selections = []
    for name, direction in [("equal_weight", "neutral"), ("fixed_top", "top"), ("fixed_bottom", "bottom"),
                            ("adaptive", "direction"), ("original", "direction"),
                            ("raw_ic", "raw_direction"), ("confirmed_no_gates", "confirmed_direction"),
                            ("gated_no_confirm", "state_direction")]:
        counterfactual = decisions.copy()
        counterfactual["direction"] = decisions[direction] if direction in decisions.columns else direction
        targets[name], selected = etf_backtest.map_targets(
            scores, counterfactual, mapping, panels["close"], panels["volume"],
            top_n=meta["top_n"], mapped_only=name != "original")
        selections.append(selected.assign(strategy=name))
    targets["balanced_active"] = targets["equal_weight"].copy()
    active = decisions.direction.ne("neutral")
    targets["balanced_active"].loc[active] = (targets["fixed_top"].loc[active] + targets["fixed_bottom"].loc[active]) / 2
    targets["cash_matched"] = targets["equal_weight"].mul(targets["original"].sum(axis=1), axis=0)
    targets["csi300"] = targets["equal_weight"].iloc[:1] * 0.0
    targets["csi300"]["510300"] = 1.0
    for name, weights in targets.items():
        save(weights, output, name + "_targets")
    save(pd.concat(selections, ignore_index=True), output, "selections", False)
    save(concentration_diagnostics(pd.concat(selections, ignore_index=True), panels, periods, mapping),
         output, "directional_concentration", False)
    save(decisions, output, "decisions")
    save(coverage, output, "coverage", False)
    costs = {"gross": (0, 0), "fee_only": (meta["fee_bps_each_side"], 0),
             "net": (meta["fee_bps_each_side"], meta["slippage_bps_each_side"])}
    results, accounts, records = {}, {}, []
    for name, weights in targets.items():
        print(f"[2/5] {name}: 无成本 / 费用 / 费用+滑点", flush=True)
        results[name], accounts[name] = {}, {}
        for mode, (fee, slip) in costs.items():
            result = etf_backtest.simulate(panels["open"], panels["close"], panels["volume"],
                                          weights, fee_bps=fee, slippage_bps=slip)
            results[name][mode] = result
            if name != "csi300":
                accounts[name][mode] = attribution.account_periods(result, panels, periods)
                compounded = (1 + accounts[name][mode].period_return).prod()
                np.testing.assert_allclose(compounded, result["nav"].iloc[-1], atol=1e-10)
            save(result["daily"], output, f"{name}_{mode}_daily")
            records.append({"strategy": name, "mode": mode, **result_stats(result),
                            "gross_turnover": result["daily"].gross_turnover.sum(),
                            "fee_paid": result["daily"].fee.sum(),
                            "slippage_paid": result["daily"].slippage.sum()})
        save(results[name]["net"]["trades"], output, f"{name}_net_trades", False)
    stats = pd.DataFrame(records)
    save(stats, output, "performance", False)
    summary = stats[stats["mode"].eq("net")].set_index("strategy").drop(columns="mode")
    gross_final = stats[stats["mode"].eq("gross")].set_index("strategy").total_ret
    fee_final = stats[stats["mode"].eq("fee_only")].set_index("strategy").total_ret
    net_final = summary.total_ret
    summary["gross_ret"] = gross_final
    summary["fee_drag"] = gross_final - fee_final
    summary["slippage_drag_after_fee"] = fee_final - net_final
    summary["cost_drag"] = gross_final - net_final
    summary["net_excess_vs_equal"] = net_final - net_final["equal_weight"]
    save(summary, output, "summary")
    print("[3/5] 周期/ETF盈亏归因、方向时段对照、排名检验", flush=True)
    period_table = periods.copy()
    period_table["direction"] = decisions.loc[periods.index, "direction"]
    for name in (key for key, value in accounts.items() if value):
        direct = attribution.gross_period_returns(targets[name], panels, periods)
        np.testing.assert_allclose(direct, accounts[name]["gross"].period_return, atol=1e-10)
        period_table[name + "_gross"] = direct
        period_table[name + "_net"] = accounts[name]["net"].period_return
    for name in ("fixed_top", "fixed_bottom", "adaptive", "original"):
        period_table[name + "_active_log"] = np.log1p(period_table[name + "_net"]) - np.log1p(period_table.equal_weight_net)
        np.testing.assert_allclose(period_table[name + "_active_log"].sum(),
            np.log((1 + net_final[name]) / (1 + net_final["equal_weight"])), atol=1e-10)
    period_table["rank_concentration_log"] = np.log1p(period_table.balanced_active_gross) - np.log1p(period_table.equal_weight_gross)
    period_table["side_choice_log"] = np.log1p(period_table.adaptive_gross) - np.log1p(period_table.balanced_active_gross)
    period_table["incremental_cost_log"] = (np.log1p(period_table.adaptive_net) - np.log1p(period_table.adaptive_gross)
        - np.log1p(period_table.equal_weight_net) + np.log1p(period_table.equal_weight_gross))
    np.testing.assert_allclose(period_table[["rank_concentration_log", "side_choice_log", "incremental_cost_log"]].sum(axis=1),
                               period_table.adaptive_active_log, atol=1e-10)
    save(period_table, output, "period_attribution")
    save(period_table.groupby("direction")[["adaptive_active_log", "rank_concentration_log", "side_choice_log", "incremental_cost_log"]].agg(["count", "sum"]),
         output, "direction_contributions")
    trades_pnl = holding_contributions(results["adaptive"]["net"], panels, periods, accounts["adaptive"]["net"], mapping)
    save(trades_pnl, output, "adaptive_etf_period_pnl", False)
    np.testing.assert_allclose(trades_pnl.pnl_initial_equity.sum() - summary.at["adaptive", "fee_paid"]
        - summary.at["adaptive", "slippage_paid"], net_final["adaptive"], atol=1e-10)
    save(trades_pnl.groupby(["etf_code", "etf_name"]).pnl_initial_equity.sum().sort_values().to_frame(), output, "adaptive_etf_pnl")
    bridges = [attribution.active_return_bridge(gross_final, net_final, strategy="adaptive", benchmark="equal_weight", anchor=anchor)
               for anchor in ("fixed_top", "fixed_bottom", "balanced_active")]
    save(pd.DataFrame(bridges), output, "return_bridges", False)
    cash_bridge = {"selection_and_direction_pp": gross_final["original"] - gross_final["cash_matched"],
                   "cash_exposure_pp": gross_final["cash_matched"] - gross_final["equal_weight"],
                   "incremental_cost_pp": (net_final["original"] - gross_final["original"])
                                          - (net_final["equal_weight"] - gross_final["equal_weight"])}
    np.testing.assert_allclose(sum(cash_bridge.values()), net_final["original"] - net_final["equal_weight"], atol=1e-10)
    save(pd.Series(cash_bridge).to_frame("contribution"), output, "original_cash_bridge")
    rank = rank_diagnostics(scores, mapping, panels, periods)
    save(rank, output, "rank_diagnostics")
    rank_summary = []
    for column in ["spearman_ic", "Q1", "Q2", "Q3", "Q4", "Q5", "Q5_minus_Q1"]:
        low, high = backtest._moving_block_bootstrap_interval(rank[column], block_size=4, samples=1000)
        rank_summary.append({"metric": column, "mean_per_period": rank[column].mean(),
                             "positive_rate": rank[column].gt(0).mean(), "ci_low": low, "ci_high": high,
                             "periods": len(rank)})
    save(pd.DataFrame(rank_summary), output, "rank_summary", False)
    timing_input = period_table[["fixed_top_gross", "fixed_bottom_gross", "equal_weight_gross"]].rename(columns=lambda c: c.removesuffix("_gross"))
    timing = attribution.circular_timing_test(timing_input, period_table.direction)
    np.testing.assert_allclose(timing.iloc[0].gross_return, gross_final["adaptive"], atol=1e-10)
    save(timing, output, "circular_timing", False)
    windows = []
    for label, since in [("2025段", None), ("2026段", "2026-01-01"), ("2026-07以来", "2026-07-01")]:
        for name in ("equal_weight", "fixed_top", "fixed_bottom", "adaptive", "original", "csi300"):
            result = results[name]["net"]
            if since is None:
                result = {"nav": result["nav"].loc[:"2025-12-31"]}
            windows.append({"window": label, "strategy": name, **result_stats(result, since=since)})
    save(pd.DataFrame(windows), output, "subperiods", False)
    bridge_rows = []
    old_nav = pd.read_csv(source / "nav.csv", index_col=0, parse_dates=True)
    for current, prior in [("adaptive", "adaptive_mapped_pool"), ("original", "adaptive_original"), ("csi300", "csi300_buy_hold")]:
        error = float((results[current]["net"]["nav"] - old_nav[prior]).abs().max())
        assert error < 1e-10, (current, error)
        bridge_rows.append({"strategy": current, "source_max_nav_error": error})
    prior_eq = old_nav.etf_equal_weight.iloc[-1] - 1
    bridge_rows.append({"strategy": "same_pool_benchmark", "prior_total_ret": prior_eq,
                       "new_total_ret": net_final["equal_weight"], "difference": net_final["equal_weight"] - prior_eq})
    save(pd.DataFrame(bridge_rows), output, "source_reconciliation", False)
    print("[4/5] 输出图表及可复核报告", flush=True)
    plot_results(output, results, period_table)
    write_report(output, meta, summary, period_table, rank_summary, timing, bridges, cash_bridge, windows, prior_eq)
    assert digest(config_path) == meta["config_sha256"]
    validation = {"passed": True, "executed_periods": len(periods),
                  "pending_signal": str(decisions.index[-1].date()), "config_unchanged": True,
                  "frozen_input_hashes_pass": True, "original_nav_reproduced": True,
                  "gross_period_compounding_pass": True, "next_open_overnight_attribution_pass": True,
                  "return_bridges_pass": True, "etf_pnl_plus_cost_reconciles": True}
    (output / "validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[5/5] 全部核对通过", flush=True)
    print(summary[["total_ret", "gross_ret", "cost_drag", "max_dd", "net_excess_vs_equal"]].to_string(), flush=True)
    print(output / "report.md", flush=True)


def plot_results(output, results, periods):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), gridspec_kw={"height_ratios": [2, 1]})
    for name, color in [("equal_weight", "#64748b"), ("fixed_top", "#d97706"),
                        ("fixed_bottom", "#0891b2"), ("adaptive", "#be123c")]:
        results[name]["net"]["nav"].plot(ax=axes[0], label=LABELS[name], color=color, linewidth=1.6)
    axes[0].axhline(1, color="#94a3b8", linewidth=.6)
    axes[0].set_title("相同 ETF 可用池与成交/成本口径：累计净值")
    axes[0].legend(ncol=2)
    axes[0].set_ylabel("净值")
    cumulative = pd.Series(periods.adaptive_active_log.cumsum().mul(100).to_numpy(),
                           index=pd.DatetimeIndex(periods.exit_date))
    cumulative.plot(ax=axes[1], color="#be123c", linewidth=1.6)
    axes[1].axhline(0, color="#94a3b8", linewidth=.6)
    axes[1].set_title("自适应相对同池等权：累计对数超额（按周期结束日记录）")
    axes[1].set_ylabel("对数百分点")
    for ax in axes:
        ax.grid(alpha=.2)
        ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)


def write_report(output, meta, summary, periods, rank_summary, timing, bridges, cash_bridge, windows, prior_eq):
    pct = lambda x: f"{x:+.2%}"
    lines = ["# ETF亏损归因与同口径对照", "",
             f"样本：{meta['start']}—{meta['end']}，{meta['sessions']}个交易日、{len(periods)}个已执行周期。",
             f"配置 SHA256：`{meta['config_sha256']}`；参数、分数、决策、ETF映射与行情均冻结。",
             "下一交易日开盘成交；买卖各5bps费用与5bps滑点；归一化复权份额。末次待执行信号不计作成交。", "",
             "## 主对照", "", "| 组合 | 净收益 | 无成本收益 | 成本损耗 | 最大回撤 | 相对等权净收益差 |",
             "|---|---:|---:|---:|---:|---:|"]
    for name in ("equal_weight", "fixed_top", "fixed_bottom", "adaptive", "original", "csi300"):
        r = summary.loc[name]
        lines.append(f"| {LABELS[name]} | {pct(r.total_ret)} | {pct(r.gross_ret)} | {r.cost_drag:.2%} | {r.max_dd:.2%} | {pct(r.net_excess_vs_equal)} |")
    lines += ["", f"主等权基准只持有信号日有情绪分且有ETF行情的标的，与Top/Bottom/自适应使用同一候选池。上轮ETF等权净收益{pct(prior_eq)}，本轮{pct(summary.at['equal_weight', 'total_ret'])}，差异来自该资格条件；三个既有账户净值均逐日复现。", "",
              f"![净值与超额]({(output / 'comparison.png').resolve().as_posix()})", "", "## 精确收益差桥接", "",
              "所有数值均为收益百分点；前两项随参照不同而改变，不能宣称存在唯一的‘排名/切换因果占比’。成本项为策略比基准多付的成本损耗。", "",
              "| 参照 | 排名/持仓无成本贡献 | 自适应切换/方向贡献 | 增量成本贡献 | 相对等权合计 |",
              "|---|---:|---:|---:|---:|"]
    for row in bridges:
        lines.append(f"| {LABELS[row['anchor']]} | {pct(row['ranking_pp'])} | {pct(row['switching_pp'])} | {pct(row['incremental_cost_pp'])} | {pct(row['total_excess_pp'])} |")
    lines += ["", "两端各半对照只在当前策略的方向期持有Top3/Bottom3各50%，观望期仍等权；用于区分‘何时集中持仓’与‘选高分还是低分’，并非新生产策略。", "",
              "## 原排名映射缺口", ""]
    lines += [f"- {key}：{pct(value)}（收益百分点）" for key, value in cash_bridge.items()]
    lines += ["", "将同池等权仓位按原排名可投资权重缩放，缺口留现金。该仓位匹配对照与原排名之差反映选择/方向；与满仓等权之差反映现金敞口时序。此为确定性对照，不等于因果识别。", "",
              "## 当前策略的已执行方向期", "",
              "周期在本次开盘成交前开始，在下次开盘成交前结束；隔夜收益属于旧持仓，成本属于新周期。", "",
              "| 信号日 | 成交日→退出估值 | 方向 | 自适应净收益 | 等权净收益 | Top3无成本 | Bottom3无成本 |",
              "|---|---|---|---:|---:|---:|---:|"]
    for signal, row in periods[periods.direction.ne("neutral")].iterrows():
        lines.append(f"| {signal:%Y-%m-%d} | {row.entry_date:%m-%d}→{row.exit_date:%m-%d}({row.exit_mark}) | {row.direction} | {pct(row.adaptive_net)} | {pct(row.equal_weight_net)} | {pct(row.fixed_top_gross)} | {pct(row.fixed_bottom_gross)} |")
    lines += ["", "## 决策层固定规则诊断", "",
              "直接使用既有决策记录中的原始IC方向、连续确认后方向、闸门后方向；未修改阈值/窗口，未选择最优参数。停用某一层仅为诊断，结果不能直接替换生产策略。", "",
              "| 决策层 | 净收益 | 无成本收益 | 成本损耗 | 最大回撤 |",
              "|---|---:|---:|---:|---:|"]
    for name in ("raw_ic", "confirmed_no_gates", "gated_no_confirm", "adaptive"):
        r = summary.loc[name]
        lines.append(f"| {LABELS[name]} | {pct(r.total_ret)} | {pct(r.gross_ret)} | {r.cost_drag:.2%} | {r.max_dd:.2%} |")
    actual = timing.iloc[0].gross_return
    other = timing.iloc[1:].gross_return
    lines += ["", f"方向序列循环错位诊断：原始无成本收益{pct(actual)}；其余{len(other)}个错位版本中位数{pct(other.median())}，低于原始收益的比例{other.lt(actual).mean():.1%}。方向次数与序列相邻结构尽量保留，循环边界会变化；这是描述性择时诊断，不是p值，错位版本使用未来序列只作反事实，不是可部署策略。", "",
              "## ETF执行周期横截面排名", "",
              "每次信号的情绪分与下一次开盘持有至下次调仓开盘的ETF收益比较；周期不重叠。Q1为最低分，Q5为最高分。均值区间使用连续4周期区块bootstrap，样本量有限。", "",
              "| 指标 | 每周期均值 | 正值比例 | 均值95%区间 |",
              "|---|---:|---:|---|"]
    for row in rank_summary:
        formatter = (lambda x: f"{x:+.4f}") if row["metric"] == "spearman_ic" else pct
        lines.append(f"| {row['metric']} | {formatter(row['mean_per_period'])} | {row['positive_rate']:.1%} | [{formatter(row['ci_low'])}, {formatter(row['ci_high'])}] |")
    lines += ["", "## 时间段稳定性", "", "| 时间段 | 同池等权 | Top3 | Bottom3 | 自适应映射池 | 原排名 | 沪深300ETF |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for label, group in pd.DataFrame(windows).groupby("window", sort=False):
        lines.append("| " + label + " | " + " | ".join(pct(v) for v in group.total_ret) + " |")
    lines += ["", "## 限制与核对", "",
              "- 当前成分与当前ETF映射有快照偏差；这轮是已观察样本内诊断，不是独立样本外证明。",
              "- 归一化复权份额未包含整手、最低佣金、容量、涨跌停排队；neutral仍是等权持仓，不表示退出市场风险。",
              "- 原排名与映射池不仅现金比例不同，也会选到不同基金；不可把两者收益差全部称为‘现金拖累’。",
              "- gross→fee_only→net的成本拆分按此固定顺序，包含复利与交互，不用成交费简单相加冒充期末净值损耗。",
              "- validation.json：输入冻结、既有账户逐日复现、执行周期复利、两条收益桥接、ETF逐项盈亏加成本均需核对通过。",
              "- period_attribution.csv保存每个周期的完整对照，adaptive_etf_period_pnl.csv及adaptive_etf_pnl.csv保存逐ETF盈亏，net_trades保存成交记录。"]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
