"""按事前锁定清单运行最后一轮组合去重/情绪增量对照，不搜索参数。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from emotion import attribution, backtest, etf_backtest, portfolio_experiment as pe
from scripts.backtest_etf_prices import price_panel, result_stats

PLAN = ROOT / "config/final_portfolio_experiment.json"
LABELS = {"A_monthly_equal": "A 同池月度等权", "B_dedup_equal": "B 去重月度等权",
          "C_dedup_emotion": "C 去重＋情绪倾斜", "reference_weekly_equal": "补充：原周度等权"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def save(frame, output, name, index=True):
    frame.to_csv(output / f"{name}.csv", index=index, encoding="utf-8-sig")


def freeze_inputs(plan, output):
    source = ROOT / plan["source"]
    meta = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    expected = plan["frozen_emotion_config_sha256"]
    assert digest(ROOT / "config.yaml") == expected == meta["config_sha256"], "情绪参数已变化"
    mapping_path = ROOT / "data/cache/etf_trading_map.parquet"
    assert digest(mapping_path) == meta["mapping_sha256"], "映射已变化"
    for code, sha in meta["price_hashes"].items():
        assert digest(ROOT / f"data/cache/etf_kline_{code}.parquet") == sha, code
    files = [PLAN, ROOT / "config.yaml", mapping_path, source / "scores.parquet",
             source / "regime_decisions.csv", source / "metadata.json"]
    manifest = {str(path.relative_to(ROOT)): digest(path) for path in files}
    manifest.update({f"data/cache/etf_kline_{code}.parquet": sha for code, sha in meta["price_hashes"].items()})
    path = output / "input_lock.json"
    if path.exists():
        assert json.loads(path.read_text(encoding="utf-8"))["hashes"] == manifest, "已锁定输入发生变化，拒绝覆盖实验"
    else:
        write_json(path, {"locked_before_computation_utc": datetime.now(timezone.utc).isoformat(),
                          "hashes": manifest})
        (output / "experiment_plan.json").write_bytes(PLAN.read_bytes())
    assert (output / "experiment_plan.json").read_bytes() == PLAN.read_bytes(), "实验规则已变化"
    return source, mapping_path, manifest


def run(output):
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    source, mapping_path, manifest = freeze_inputs(plan, output)
    print("[1/4] 规则及全部输入已锁定；建立相同月度候选池", flush=True)
    scores = pd.read_parquet(source / "scores.parquet")
    mapping = pd.read_parquet(mapping_path)
    decisions = pd.read_csv(source / "regime_decisions.csv", index_col="date", parse_dates=["date"])
    panels, coverage = price_panel(mapping, scores.index)
    dates = pe.monthly_signals(decisions.index)
    dcfg = plan["dedup"]
    kwargs = dict(lookback=dcfg["lookback_returns"], min_observations=dcfg["minimum_pair_observations"],
                  threshold=dcfg["correlation_threshold"], amplitude=plan["emotion_overlay"]["amplitude"])
    targets, audit, diagnostics = pe.build_targets(scores, mapping, panels["close"], panels["volume"], dates, **kwargs)
    neutral = decisions.assign(direction="neutral")
    targets["reference_weekly_equal"], _ = etf_backtest.map_targets(
        scores, neutral, mapping, panels["close"], panels["volume"], mapped_only=True)
    # 本数据映射无ETF别名；独立核对A与旧等权池完全一致。
    np.testing.assert_allclose(targets["A_monthly_equal"], targets["reference_weekly_equal"].loc[dates], atol=1e-12)
    save(audit, output, "groups", False)
    save(diagnostics, output, "dedup_diagnostics", False)
    save(coverage, output, "coverage", False)
    for name, frame in targets.items():
        save(frame, output, name + "_targets")
    results, records, accounts = {}, [], {}
    periods = attribution.execution_periods(scores.index, dates)
    for name, weights in targets.items():
        print(f"[2/4] {name}: 无成本、标准成本、双倍成本", flush=True)
        results[name] = {}
        for mode, cost in plan["costs"].items():
            result = etf_backtest.simulate(panels["open"], panels["close"], panels["volume"], weights, **cost)
            results[name][mode] = result
            records.append({"strategy": name, "mode": mode, **result_stats(result),
                            "gross_turnover": result["daily"].gross_turnover.sum(),
                            "fee_paid": result["daily"].fee.sum(), "slippage_paid": result["daily"].slippage.sum()})
            save(result["daily"], output, f"{name}_{mode}_daily")
            save(result["trades"], output, f"{name}_{mode}_trades", False)
            if name != "reference_weekly_equal":
                account = attribution.account_periods(result, panels, periods)
                np.testing.assert_allclose((1 + account.period_return).prod(), result["nav"].iloc[-1], atol=1e-10)
                save(account, output, f"{name}_{mode}_periods")
                if mode == "gross":
                    np.testing.assert_allclose(attribution.gross_period_returns(weights, panels, periods),
                                               account.period_return, atol=1e-10)
                if mode == "net":
                    accounts[name] = account
                    result["holdings"].to_parquet(output / f"{name}_net_holdings.parquet")
    stats = pd.DataFrame(records)
    save(stats, output, "performance", False)
    summary = stats[stats["mode"].eq("net")].set_index("strategy").drop(columns="mode")
    for mode in ("gross", "stress"):
        summary[mode + "_ret"] = stats[stats["mode"].eq(mode)].set_index("strategy").total_ret
    summary["cost_drag"] = summary.gross_ret - summary.total_ret
    summary["excess_vs_A"] = summary.total_ret - summary.at["A_monthly_equal", "total_ret"]
    save(summary, output, "summary")
    print("[3/4] 固定退出条件：增量、回撤、区间、分期和成本压力", flush=True)
    period_table = periods.copy()
    for name, account in accounts.items():
        period_table[name] = account.period_return
    period_table["C_minus_B"] = period_table.C_dedup_emotion - period_table.B_dedup_equal
    period_table["B_minus_A"] = period_table.B_dedup_equal - period_table.A_monthly_equal
    save(period_table, output, "period_comparisons")
    a, b, c = "A_monthly_equal", "B_dedup_equal", "C_dedup_emotion"
    active = (np.log1p(results[c]["net"]["daily"]["return"])
              - np.log1p(results[b]["net"]["daily"]["return"])).loc[periods.entry_date.iloc[0]:]
    infer = plan["inference"]
    lo, hi = backtest._moving_block_bootstrap_interval(active, block_size=infer["bootstrap_block_sessions"],
                samples=infer["bootstrap_samples"], seed=infer["bootstrap_seed"])
    completed = period_table.loc[period_table.exit_mark.eq("open"), "C_minus_B"]
    checks = pe.exit_checks(summary, lo, completed)
    decision = {"experiment_id": plan["experiment_id"], "plan_sha256": digest(PLAN),
                "all_pass": all(checks.values()), "checks": checks,
                "project_mode": "research_only" if all(checks.values()) else "observation_only",
                "independent_out_of_sample": False, "live_trading_eligible": False,
                "paired_daily_log_mean": float(active.mean()), "paired_daily_log_ci95": [lo, hi],
                "annualized_relative_ci95": [float(np.expm1(252 * lo)), float(np.expm1(252 * hi))],
                "completed_periods": len(completed), "positive_completed_periods": int(completed.gt(0).sum()),
                "right_censored_periods": int(period_table.exit_mark.eq("close").sum()),
                "executed_monthly_signals": len(periods),
                "pending_monthly_signals": [str(date.date()) for date in results[c]["net"]["pending_signals"]]}
    write_json(output / "decision.json", decision)
    # 精确桥接：降频 -> 去重 -> 情绪；只是描述性差值，不声称唯一因果。
    bridge = {"monthly_minus_weekly": summary.at[a, "total_ret"] - summary.at["reference_weekly_equal", "total_ret"],
              "dedup_B_minus_A": summary.at[b, "total_ret"] - summary.at[a, "total_ret"],
              "emotion_C_minus_B": summary.at[c, "total_ret"] - summary.at[b, "total_ret"],
              "emotion_gross_C_minus_B": summary.at[c, "gross_ret"] - summary.at[b, "gross_ret"],
              "emotion_incremental_cost": (summary.at[c, "total_ret"] - summary.at[c, "gross_ret"])
                                          - (summary.at[b, "total_ret"] - summary.at[b, "gross_ret"])}
    np.testing.assert_allclose(bridge["emotion_gross_C_minus_B"] + bridge["emotion_incremental_cost"],
                               bridge["emotion_C_minus_B"], atol=1e-12)
    write_json(output / "return_bridge.json", bridge)
    subperiods = []
    for label, start, end in [("2025样本段", "2025-09-01", "2025-12-31"),
                              ("2026样本段", "2026-01-01", "2026-10-09")]:
        for name in targets:
            daily = results[name]["net"]["daily"].loc[start:end]
            subperiods.append({"period": label, "strategy": name, "net_return": float((1 + daily["return"]).prod() - 1)})
    save(pd.DataFrame(subperiods), output, "subperiods", False)
    print("[4/4] 参数不变、未来不泄漏、账本与旧基准复现校验", flush=True)
    cutoff = dates[len(dates) // 2]
    truncated, _, _ = pe.build_targets(scores.loc[:cutoff], mapping, panels["close"].loc[:cutoff],
        panels["volume"].loc[:cutoff], dates[dates <= cutoff], **kwargs)
    for name, frame in truncated.items():
        pd.testing.assert_frame_equal(frame, targets[name].loc[:cutoff])
    prior = pd.read_csv(ROOT / "report/etf_attribution_20261009/equal_weight_net_daily.csv", index_col="date", parse_dates=["date"])
    np.testing.assert_allclose(results["reference_weekly_equal"]["net"]["nav"], prior.nav, atol=1e-10)
    for frame in targets.values():
        np.testing.assert_allclose(frame.sum(axis=1), 1, atol=1e-12)
        assert (frame >= 0).all().all()
    assert (targets[b].gt(0) == targets[c].gt(0)).all().all()
    assert audit.loc[audit.group_size.gt(1), "minimum_pair_correlation"].ge(dcfg["correlation_threshold"]).all()
    assert audit.loc[audit.group_size.gt(1), "minimum_pair_observations"].ge(dcfg["minimum_pair_observations"]).all()
    for path, sha in manifest.items():
        assert digest(ROOT / path) == sha, f"运行中输入变化: {path}"
    validation = {"passed": True, "input_hashes_unchanged": True, "future_truncation_invariant": True,
                  "gross_periods_independently_reconciled": True, "all_cost_accounts_compound": True,
                  "prior_weekly_net_nav_reproduced": True, "same_representatives_B_C": True,
                  "group_pair_thresholds_verified": True,
                  "code_hashes": {str(path.relative_to(ROOT)): digest(path) for path in
                      (Path(__file__), ROOT / "emotion/portfolio_experiment.py", ROOT / "emotion/etf_backtest.py",
                       ROOT / "emotion/attribution.py", ROOT / "emotion/backtest.py", ROOT / "scripts/backtest_etf_prices.py")}}
    write_json(output / "validation.json", validation)
    make_plot(output, results)
    write_report(output, plan, summary, diagnostics, bridge, decision, subperiods)
    print(summary[["gross_ret", "total_ret", "max_dd", "stress_ret"]].to_string(), flush=True)
    print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)


def make_plot(output, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for name, modes in results.items():
        nav = modes["net"]["nav"]
        axes[0].plot(nav.index, nav, label=name, lw=1.5)
    b, c = (results[name]["net"]["nav"] for name in ("B_dedup_equal", "C_dedup_emotion"))
    axes[1].plot(c.index, (c / b - 1) * 100, color="#a24b2a", label="C / B - 1 (net, %)")
    axes[1].axhline(0, color="gray", lw=.7)
    axes[0].set_title("Final fixed-rule ETF comparison | 5 bps fee + 5 bps slippage each side")
    axes[0].set_ylabel("Net asset value")
    axes[1].set_ylabel("Relative wealth (%)")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)


def write_report(output, plan, summary, diagnostics, bridge, decision, subperiods):
    lines = ["# 最后一轮：组合去重与情绪增量对照", "",
             "**结论：" + ("通过样本内筛选，仅保留研究候选；不能上线。" if decision["all_pass"] else
                            "未通过事前退出条件，终止此情绪交易路线，项目保留为观察工具。") + "**", "",
             "2025-09-01—2026-10-09；情绪因子、权重、平滑、阈值与原配置全部不变。",
             f"规则SHA256：`{decision['plan_sha256']}`。计算前写入input_lock.json并复制实验规则；重跑拒绝规则或输入变更。",
             "这是已观察历史上的最后一轮固定规则诊断，不是独立样本外验证。", "",
             "## 主对照与费用", "",
             "A/B/C每月第一个既有周度信号后，下一A股交易日开盘成交；原周度等权只作补充。",
             "A等权全部可用ETF；B按历史相关性分组，每组选择一个不依赖情绪的代表并等权；C使用完全相同代表，按组内平均情绪排名作±20%的相对权重倾斜。",
             "去重使用截至信号日60日收益、至少20对有效观测、相关性≥0.90的贪心完全连接分组；未知相关性不合并。代表是组内平均相关性最高者，平手按代码。",
             "早期历史不足时保持独立；组间不保证相互独立，也不等于消除了行业/政策共同风险。",
             "三者均仅月度调仓，没有硬换手上限。标准成本为买卖各5bps费用+5bps滑点；压力测试两者均加倍，首次建仓收费、期末不强制平仓。", "",
             "| 组合 | 无成本 | 扣费及滑点 | 成本损耗 | 最大回撤 | 双倍成本净收益 |", "|---|---:|---:|---:|---:|---:|"]
    for name, row in summary.iterrows():
        lines.append(f"| {LABELS[name]} | {row.gross_ret:+.2%} | {row.total_ret:+.2%} | {row.cost_drag:.2%} | {row.max_dd:.2%} | {row.stress_ret:+.2%} |")
    lines += ["", "## 增量分解", "",
              f"- 月度化A相对原周度等权：{bridge['monthly_minus_weekly'] * 100:+.3f}个百分点。",
              f"- 去重B相对A：{bridge['dedup_B_minus_A'] * 100:+.3f}个百分点。",
              f"- 情绪C相对B：{bridge['emotion_C_minus_B'] * 100:+.6f}个百分点；无成本增量{bridge['emotion_gross_C_minus_B'] * 100:+.6f}个百分点，增量成本贡献{bridge['emotion_incremental_cost'] * 100:+.6f}个百分点。",
              "本轮去重未改善基准，代表基金更替仍产生换手；不能把去重视作已验证的收益来源。",
              "以上是固定对照的收益差，不是唯一因果归因；不以某一个有利片段推翻完整结果。", "",
              "## 事前退出条件（必须全部通过）", ""]
    descriptions = ["C扣费收益超过B", "C扣费收益超过A", "C最大回撤不差于B",
                    "C−B配对日对数收益的20日移动区块bootstrap 95%区间下界>0",
                    "至少12个完整月度执行期，且≥2/3期间C扣费跑赢B", "费用和滑点同时加倍后C仍跑赢B"]
    for description, passed in zip(descriptions, decision["checks"].values()):
        lines.append(f"- {'通过' if passed else '未通过'}：{description}。")
    lo, hi = decision["annualized_relative_ci95"]
    lines += ["", f"C/B年化相对收益描述性95%区间：[{lo:+.2%}, {hi:+.2%}]。5000次重采样，种子20261013；不能解释为未来收益范围或经多重试验校正的显著性。",
              f"完整月度期{decision['completed_periods']}个，其中C胜出{decision['positive_completed_periods']}个；最后{decision['right_censored_periods']}个未完成期计入总净值，但不计入分期胜率。",
              f"实际执行月度信号{decision['executed_monthly_signals']}次；待执行：{decision['pending_monthly_signals']}。", "",
              "## 去重与分段诊断", "",
              f"候选ETF数范围{diagnostics.eligible_etfs.min()}—{diagnostics.eligible_etfs.max()}，分组数范围{diagnostics.groups.min()}—{diagnostics.groups.max()}。包括待执行信号，详见groups.csv与dedup_diagnostics.csv。",
              "完全连接分组消除组内重复配置，但部分组间代表仍高度相关；本轮未将其解释为完全独立的风险配置。",
              "| 已观察样本段 | 组合 | 净收益 |", "|---|---|---:|"]
    for row in subperiods:
        lines.append(f"| {row['period']} | {LABELS[row['strategy']]} | {row['net_return']:+.2%} |")
    lines += ["", "## 项目处理与边界", "",
              "未通过即停止本路线交易开发，不再反转情绪方向或调整聚类阈值、幅度、周期寻找赢家。保留数据、评分、冷热观察、相关性与历史账本；日报和看板以观察语义展示。"
              if not decision["all_pass"] else "只保留研究候选，仍需历史时点数据和未来未观察样本验证，不自动部署。",
              "当前成分与ETF映射仍为快照；本轮没有将其伪装成历史时点数据。短样本不能证明情绪永远无效，也不足以支持继续以盈利为目标投入本版本。",
              "归一化复权份额未模拟整手、最低佣金、容量和涨跌停排队；成本是固定假设。",
              "校验通过：配置/行情/映射哈希不变、过去目标不受未来数据影响、无成本周期独立复算、所有成本账户复利核对、旧周度基准逐日复现、B/C持有同一批代表。", "",
              f"![净值与情绪增量]({(output / 'comparison.png').as_posix()})", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "report/final_portfolio_20261009")
    run(parser.parse_args().output)
