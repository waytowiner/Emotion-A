"""冻结策略参数，完整自适应信号 + 真实 ETF 次日开盘执行回测。"""
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
from emotion import backtest, etf_backtest, pipeline, score  # noqa: E402
from emotion.factors import compute_all_factors  # noqa: E402

CACHE = ROOT / "data" / "cache"
LABELS = {"adaptive_original": "自适应原排名（缺口留现金）",
          "adaptive_mapped_pool": "自适应映射池（部署对照）",
          "etf_equal_weight": "ETF池等权（周度再平衡）",
          "csi300_buy_hold": "沪深300 ETF 510300（买入持有）"}

# 已核查的整日停牌，不制造成交日线；其余缺失仍按未知缺口报告。
VERIFIED_HALTS = {
    ("159583", "2025-09-11"): "https://static.cninfo.com.cn/finalpage/2025-09-11/1224649010.PDF",
    ("159528", "2025-12-04"): "https://www.fullgoal.com.cn/wbs-file/fund_report/20251204/CN_50100000_159528_FC900010_20250001.pdf",
}


def price_panel(mapping, calendar):
    """不填补缺失价格，不回填上市前行情。"""
    prices_by_code, audit = {}, []
    for code in sorted(set(mapping.etf_code.astype(str)) | {"510300"}):
        path = CACHE / f"etf_kline_{code}.parquet"
        if not path.exists():
            raise ValueError(f"缺少ETF行情: {code}")
        prices = pd.read_parquet(path).set_index("date").sort_index()
        if not prices.index.is_unique:
            raise ValueError(f"重复 ETF 日期: {code}")
        prices = prices.reindex(calendar)
        good = (prices[["open", "close", "high", "low"]].notna().all(axis=1)
                & (prices[["open", "close", "high", "low"]] > 0).all(axis=1))
        bad = good & ((prices["low"] > prices[["open", "close"]].min(axis=1) + 1e-6)
                      | (prices["high"] < prices[["open", "close"]].max(axis=1) - 1e-6))
        if bad.any():
            raise ValueError(f"OHLC异常: {code}")
        observed = prices.index[good]
        if not len(observed) or observed[-1] != calendar[-1]:
            raise ValueError(f"ETF截止日缺失/过期: {code}")
        prices_by_code[code] = prices.where(good, np.nan)
        missing_dates = prices.index[(~good) & (prices.index >= observed[0])]
        confirmed = sum((code, str(d.date())) in VERIFIED_HALTS for d in missing_dates)
        audit.append({"etf_code": code, "first_date": observed[0], "last_date": observed[-1],
                      "n_bars": int(good.sum()), "calendar_days": len(calendar),
                      "missing_after_first": int((~good.loc[observed[0]:]).sum()),
                      "confirmed_halt_days": confirmed,
                      "unexplained_gaps": len(missing_dates) - confirmed,
                      "missing_dates": ",".join(str(d.date()) for d in missing_dates),
                      "zero_volume": int((prices.loc[good, "volume"] <= 0).sum())})
    panels = {key: pd.DataFrame({code: p[key] for code, p in prices_by_code.items()})
              for key in ("open", "close", "volume")}
    return panels, pd.DataFrame(audit)


def result_stats(result, since=None, last=None):
    nav = result["nav"]
    if since is not None:
        pos = nav.index.searchsorted(pd.Timestamp(since))
        nav = nav.iloc[max(0, pos - 1):]  # 区间前一交易日收盘权益
    if last is not None:
        nav = nav.iloc[-(last + 1):]
    return backtest.performance_stats(nav / nav.iloc[0])


def write_report(output, metadata, results, summary, decisions, selections, coverage, sensitivity):
    lines = ["# 真实 ETF 自适应回测", "",
             f"数据：{metadata['start']}—{metadata['end']}，{metadata['sessions']}个 A 股交易日。",
             f"配置 SHA256：`{metadata['config_sha256']}`；策略参数全部沿用 config.yaml。", "",
             "## 完整样本：费用和滑点后", "",
             "| 组合 | 累计净收益 | 无成本收益 | 成本损耗（净值百分点） | 最大回撤 | 夏普 | 平均现金 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, row in summary.iterrows():
        lines.append(f"| {LABELS[name]} | {row.total_ret:+.2%} | {row.gross_ret:+.2%} | {row.cost_drag:.2%} | {row.max_dd:.2%} | {row.sharpe:.2f} | {row.mean_cash_weight:.1%} |")
    lines += ["", "## 与可交易基准比较", "",
              "| 策略 | 基准 | 收益差（百分点） | 相对财富收益 | 年化相对收益区块bootstrap 95% CI |",
              "|---|---|---:|---:|---|"]
    comparisons = []
    evaluation = metadata["parameters"].get("evaluation", {})
    for name in ("adaptive_original", "adaptive_mapped_pool"):
        for bench in ("etf_equal_weight", "csi300_buy_hold"):
            diff = summary.at[name, "total_ret"] - summary.at[bench, "total_ret"]
            relative = results[name]["nav"].iloc[-1] / results[bench]["nav"].iloc[-1] - 1
            active = np.log(results[name]["nav"]).diff() - np.log(results[bench]["nav"]).diff()
            lo, hi = backtest._moving_block_bootstrap_interval(
                active, block_size=evaluation.get("rolling_block_days", 20),
                samples=evaluation.get("bootstrap_samples", 1000))
            lo, hi = float(np.expm1(lo * 252)), float(np.expm1(hi * 252))
            comparisons.append({"strategy": name, "benchmark": bench, "return_difference": diff,
                                "relative_return": relative, "annual_relative_ci_low": lo,
                                "annual_relative_ci_high": hi})
            lines.append(f"| {LABELS[name]} | {LABELS[bench]} | {diff:+.2%} | {relative:+.2%} | [{lo:+.1%}, {hi:+.1%}] |")
    pd.DataFrame(comparisons).to_csv(output / "comparisons.csv", index=False, encoding="utf-8-sig")
    lines += ["", "## 连续账户近期表现", "",
              "| 区间 | 原排名自适应 | 映射池自适应 | ETF等权 | 沪深300ETF |", "|---|---:|---:|---:|---:|"]
    periods = []
    for label, since, last in [("2026-07-01以来", "2026-07-01", None), ("最近60交易日", None, 60), ("最近30交易日", None, 30)]:
        returns = [result_stats(result, since=since, last=last)["total_ret"] for result in results.values()]
        lines.append("| " + label + " | " + " | ".join(f"{r:+.2%}" for r in returns) + " |")
        periods.extend({"period": label, "name": name, "total_ret": value}
                       for name, value in zip(results, returns))
    pd.DataFrame(periods).to_csv(output / "recent_periods.csv", index=False, encoding="utf-8-sig")
    lines += ["", "## 费用敏感性（仅改变滑点，不搜索策略参数）", "",
              "| 单边滑点 | 原排名自适应 | 映射池自适应 | ETF等权 | 沪深300ETF |", "|---|---:|---:|---:|---:|"]
    for slip, group in sensitivity.groupby("slippage_bps"):
        values = group.set_index("name").reindex(results)["total_ret"]
        lines.append(f"| {slip:g}bps | " + " | ".join(f"{v:+.2%}" for v in values) + " |")
    directional = selections[selections.direction != "neutral"]
    missed = directional[directional.status != "mapped"]
    lines += ["", "## 覆盖与执行审计", "",
              f"- {len(coverage)}只ETF均覆盖截止日；原映射196个指数，其中{metadata['mapped_scored_boards']}个有情绪分；{metadata['scored_boards']}个有分板块。",
              f"- 原排名方向性选择{len(directional)}条，无法映射/当日无可用行情{len(missed)}条；缺口留现金，不事后替换ETF。original_selections.csv保存全部选择。",
              f"- 上市后缺失日线{int(coverage.missing_after_first.sum())}条，无量日{int(coverage.zero_volume.sum())}条；不回填上市前价格。",
              f"- 经基金管理人公告确认整日停牌{int(coverage.confirmed_halt_days.sum())}条；未解释缺口{int(coverage.unexplained_gaps.sum())}条。停牌延用已知收盘价估值，不生成可成交行情。",
              f"- 周度方向次数：{decisions.direction.value_counts().to_dict()}；末次信号{metadata['pending_signal_dates']}待下一交易日执行。",
              f"- 持仓缺失估值日数：{summary.stale_holding_days.astype(int).to_dict()}；无法成交订单：{summary.blocked_orders.astype(int).to_dict()}。", "",
              "## 口径与限制", "",
              f"- 收盘后周度信号在下一个A股交易日开盘成交；隔夜收益归旧持仓，开盘后才计算新持仓收益。首次建仓收费，期末四个账户均按收盘估值、不虚构平仓。",
              f"- 买入、卖出分别收{metadata['fee_bps_each_side']:g}bps费用（含规费的总费率假设），成交价分别上浮、下浮{metadata['slippage_bps_each_side']:g}bps。ETF免印花税，管理费已反映在价格中，不重复扣除。",
              "- neutral保持原规则：等权全部候选板块，并非空仓。原排名缺失映射留现金；部署对照仅收窄落仓池，IC、风控与方向仍来自完整原池。两者不属于完全相同的持仓策略。",
              "- 等权基准持有映射池独立ETF，每周等权再平衡，仅纳入当时已有日线且有成交量的基金；510300在相同首次执行日买入持有。所有账户从初始现金开始，信号不足阶段保留，不截掉亏损期。",
              "- 使用连续复权OHLC与归一化虚拟份额，近似分红再投资收益。未做100份整手、最低佣金、涨跌停排队、逐笔撮合或资金容量建模；因此是可交易标的的成本后研究回测，不是指定账户的实盘收益。",
              "- 当前指数成分快照、按当前规模选ETF映射存在幸存者/映射选择偏差。未进行真正样本外检验，20日区块bootstrap仅为描述性区间。",
              "- 目录保存决策、目标、日账本、交易、持仓、行情覆盖和输入哈希，支持复核。", "",
              "停牌依据：[159583于2025-09-11停牌](https://static.cninfo.com.cn/finalpage/2025-09-11/1224649010.PDF)，[159528于2025-12-04停牌](https://www.fullgoal.com.cn/wbs-file/fund_report/20251204/CN_50100000_159528_FC900010_20250001.pdf)。", "",
              "费用依据：[上交所ETF介绍（免印花税）](https://etf.sse.com.cn/fund/learning/knowledge/c/5704298.shtml)，[上交所收费表](https://www.sse.com.cn/services/tradingservice/charge/ssecharge/)。具体佣金以券商协议为准。"]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--end", default=None)
    ap.add_argument("--slippage-bps", type=float, default=5.0)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    config_path = ROOT / "config.yaml"
    frozen_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    cfg = pipeline.load_config()
    end = args.end or pipeline.cache_end_date(cfg)
    output = Path(args.output) if args.output else ROOT / "report" / f"etf_adaptive_{pd.Timestamp(end):%Y%m%d}"
    output.mkdir(parents=True, exist_ok=True)
    print(f"[1/4] 参数冻结，离线重建全样本信号，截止 {end}", flush=True)
    data = pipeline.prepare(cfg, end_override=end, offline=True)
    factors = compute_all_factors(data["stock_daily"], data["board_map"], data["board_kline"],
                                  market_total=data.get("market_total"))
    _, smoothed, _ = score.build_scores(factors, cfg["weights"], cfg.get("smooth_window", 5),
        lookback=cfg.get("score_lookback", 250), z_clip=cfg.get("score_z_clip", 3.0))
    index_close = data["board_kline"].pivot(index="date", columns="board_code", values="close").sort_index()
    smoothed = smoothed.reindex(index=index_close.index, columns=index_close.columns)
    print("[2/4] 共用日报风控/确认规则，验证ETF行情", flush=True)
    regime = cfg.get("regime", {})
    top_n = cfg["signal"].get("top_n", 3)
    decisions = backtest.regime_decisions(smoothed, index_close,
        confirm=regime.get("confirm_readings", 1), min_pick=top_n, **backtest.regime_parameters(regime))
    mapping = pd.read_parquet(CACHE / "etf_trading_map.parquet")
    if mapping.board_code.duplicated().any():
        raise ValueError("每个指数须唯一交易映射")
    panels, coverage = price_panel(mapping, index_close.index)
    targets, selections = etf_backtest.map_targets(smoothed, decisions, mapping,
        panels["close"], panels["volume"], top_n=top_n)
    mapped_targets, mapped_selections = etf_backtest.map_targets(smoothed, decisions, mapping,
        panels["close"], panels["volume"], top_n=top_n, mapped_only=True)
    basket = mapped_targets * 0.0
    pool = set(mapping.loc[mapping.board_code.isin(smoothed.dropna(axis=1, how="all").columns), "etf_code"])
    for date in basket.index:
        eligible = sorted(code for code in pool if pd.notna(panels["close"].at[date, code])
                          and panels["volume"].at[date, code] > 0)
        if eligible:
            basket.loc[date, eligible] = 1 / len(eligible)
    broad = basket.iloc[:1] * 0.0
    if len(broad):
        broad["510300"] = 1.0
    all_targets = {"adaptive_original": targets, "adaptive_mapped_pool": mapped_targets,
                   "etf_equal_weight": basket, "csi300_buy_hold": broad}
    fee_bps = float(cfg["backtest"].get("cost_bps", 5))
    def run(weights, fee, slip):
        return etf_backtest.simulate(panels["open"], panels["close"], panels["volume"],
                                    weights, fee_bps=fee, slippage_bps=slip)
    print("[3/4] 次日开盘自融资账本，每笔买卖分别计费", flush=True)
    results = {name: run(weights, fee_bps, args.slippage_bps) for name, weights in all_targets.items()}
    gross = {name: run(weights, 0, 0) for name, weights in all_targets.items()}
    records = []
    for name, result in results.items():
        daily = result["daily"]
        records.append({"name": name, **result_stats(result),
            "gross_ret": result_stats(gross[name])["total_ret"],
            "cost_drag": gross[name]["nav"].iloc[-1] - result["nav"].iloc[-1],
            "fee_paid_initial_equity": daily.fee.sum(), "slippage_paid_initial_equity": daily.slippage.sum(),
            "gross_turnover": daily.gross_turnover.sum(), "mean_cash_weight": daily.cash_weight.mean(),
            "blocked_orders": int(daily.blocked_orders.sum()),
            "stale_holding_days": int((daily.stale_holdings > 0).sum())})
        daily.to_csv(output / f"{name}_daily.csv", encoding="utf-8-sig")
        result["trades"].to_csv(output / f"{name}_trades.csv", index=False, encoding="utf-8-sig")
        result["holdings"].to_parquet(output / f"{name}_holdings.parquet")
        all_targets[name].to_csv(output / f"{name}_targets.csv", encoding="utf-8-sig")
    summary = pd.DataFrame(records).set_index("name")
    summary.to_csv(output / "summary.csv", encoding="utf-8-sig")
    for filename, frame in [("regime_decisions", decisions), ("original_selections", selections),
                            ("mapped_selections", mapped_selections), ("quote_coverage", coverage)]:
        frame.to_csv(output / f"{filename}.csv", encoding="utf-8-sig")
    smoothed.to_parquet(output / "scores.parquet")
    pd.DataFrame({name: result["nav"] for name, result in results.items()}).to_csv(output / "nav.csv", encoding="utf-8-sig")
    sensitivity = []
    for slippage in sorted(set([0.0, args.slippage_bps, 10.0])):
        for name, weights in all_targets.items():
            result = results[name] if slippage == args.slippage_bps else run(weights, fee_bps, slippage)
            sensitivity.append({"slippage_bps": slippage, "name": name, **result_stats(result)})
    sensitivity = pd.DataFrame(sensitivity)
    sensitivity.to_csv(output / "cost_sensitivity.csv", index=False, encoding="utf-8-sig")
    if hashlib.sha256(config_path.read_bytes()).hexdigest() != frozen_hash:
        raise RuntimeError("运行中策略配置改变")
    metadata = {"config_sha256": frozen_hash, "start": str(index_close.index[0].date()),
        "end": str(index_close.index[-1].date()), "sessions": len(index_close), "top_n": top_n,
        "fee_bps_each_side": fee_bps, "slippage_bps_each_side": args.slippage_bps,
        "parameters": cfg, "source": "Tencent ifzq qfqday/day ETF OHLC",
        "execution": "signal close -> next A-share session open",
        "scored_boards": int(smoothed.notna().any().sum()),
        "mapped_scored_boards": len(set(smoothed.dropna(axis=1, how="all").columns) & set(mapping.board_code)),
        "pending_signal_dates": [str(d.date()) for d in results["adaptive_original"]["pending_signals"]],
        "calendar_audit": data.get("calendar_audit"),
        "mapping_sha256": hashlib.sha256((CACHE / "etf_trading_map.parquet").read_bytes()).hexdigest(),
        "price_hashes": {code: hashlib.sha256((CACHE / f"etf_kline_{code}.parquet").read_bytes()).hexdigest()
                         for code in panels["close"].columns}}
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("[4/4] 输出报告、基准比较及敏感性", flush=True)
    write_report(output, metadata, results, summary, decisions, selections, coverage, sensitivity)
    print(summary.to_string(), flush=True)
    print(f"报告：{output / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
