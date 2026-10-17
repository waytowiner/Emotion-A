"""独立复核持久化 ETF 账本、成交时序、输入冻结及共用方向判定。"""
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
from emotion import backtest  # noqa: E402
from scripts.backtest_etf_prices import LABELS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--directory", type=Path, required=True)
    args = ap.parse_args()
    directory = args.directory
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    assert hashlib.sha256((ROOT / "config.yaml").read_bytes()).hexdigest() == metadata["config_sha256"]
    for code, expected in metadata["price_hashes"].items():
        path = ROOT / "data" / "cache" / f"etf_kline_{code}.parquet"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, code
    scores = pd.read_parquet(directory / "scores.parquet")
    daily_scores = pd.read_parquet(ROOT / "data/dashboard/scores.parquet").set_index("date")
    pd.testing.assert_frame_equal(scores, daily_scores.reindex(index=scores.index, columns=scores.columns))
    prices = pd.read_parquet(ROOT / "data/dashboard/board_kline.parquet")
    index_close = prices.pivot(index="date", columns="board_code", values="close").sort_index()
    index_close = index_close.reindex(index=scores.index, columns=scores.columns)
    regime = metadata["parameters"].get("regime", {})
    decisions = backtest.regime_decisions(scores, index_close,
        confirm=regime.get("confirm_readings", 1), min_pick=metadata["top_n"],
        **backtest.regime_parameters(regime))
    saved = pd.read_csv(directory / "regime_decisions.csv", index_col="date", parse_dates=["date"])
    for column in ("direction", "raw_direction", "confirmed_direction", "state_direction",
                   "guard_hit", "vol_gate", "robust_gate", "mono_gate"):
        assert decisions[column].tolist() == saved[column].tolist(), column
    np.testing.assert_allclose(decisions.ic_mean, saved.ic_mean, atol=1e-12, equal_nan=True)
    quote_close = pd.DataFrame({code: pd.read_parquet(ROOT / "data/cache" / f"etf_kline_{code}.parquet")
                               .set_index("date").close for code in metadata["price_hashes"]}).reindex(scores.index)
    summary = pd.read_csv(directory / "summary.csv", index_col="name")
    mapping_path = ROOT / "data/cache/etf_trading_map.parquet"
    assert hashlib.sha256(mapping_path.read_bytes()).hexdigest() == metadata["mapping_sha256"]
    checks = []
    for name in LABELS:
        daily = pd.read_csv(directory / f"{name}_daily.csv", index_col="date", parse_dates=["date"])
        trades = pd.read_csv(directory / f"{name}_trades.csv", dtype={"etf_code": str},
                             parse_dates=["signal_date", "execution_date"])
        shares = pd.Series(0.0, index=quote_close.columns)
        cash = 1.0
        last_prices = pd.Series(np.nan, index=shares.index)
        max_error = 0.0
        fee = metadata["fee_bps_each_side"] / 10000
        slip = metadata["slippage_bps_each_side"] / 10000
        for date in daily.index:
            executions = trades[trades.execution_date.eq(date)]
            fee_paid = slip_paid = 0.0
            for row in executions.itertuples():
                assert date > row.signal_date
                assert date == daily.index[daily.index.get_loc(row.signal_date) + 1]
                sign = 1 if row.side == "buy" else -1
                assert np.sign(row.adjusted_units) == sign
                assert np.isclose(row.execution_price, row.open * (1 + sign * slip))
                assert np.isclose(row.fee, abs(row.adjusted_units) * row.execution_price * fee)
                assert np.isclose(row.slippage, abs(row.adjusted_units) * row.open * slip)
                cash -= row.adjusted_units * row.execution_price + row.fee
                shares[row.etf_code] += row.adjusted_units
                fee_paid += row.fee
                slip_paid += row.slippage
            assert cash >= -1e-9
            assert (shares >= -1e-9).all()
            last_prices = quote_close.loc[date].combine_first(last_prices)
            independent_nav = cash + float((shares * last_prices).fillna(0).sum())
            max_error = max(max_error, abs(independent_nav - daily.at[date, "nav"]))
            assert np.isclose(independent_nav, daily.at[date, "nav"], atol=1e-10, rtol=0)
            assert np.isclose(cash / independent_nav, daily.at[date, "cash_weight"], atol=1e-10)
            assert np.isclose(fee_paid, daily.at[date, "fee"], atol=1e-12)
            assert np.isclose(slip_paid, daily.at[date, "slippage"], atol=1e-12)
        assert np.isclose(daily.nav.iloc[-1] / daily.nav.iloc[0] - 1, summary.at[name, "total_ret"])
        if name == "csi300_buy_hold":
            assert len(trades) == 1 and trades.iloc[0].etf_code == "510300"
            first = trades.iloc[0]
            expected = quote_close.at[daily.index[-1], "510300"] / (
                first.open * (1 + slip) * (1 + fee))
            assert np.isclose(daily.nav.iloc[-1], expected, atol=1e-10)
        checks.append({"name": name, "days": len(daily), "trades": len(trades),
                       "max_independent_nav_error": max_error, "passed": True})
    pd.DataFrame(checks).to_csv(directory / "validation.csv", index=False, encoding="utf-8-sig")
    lines = ["# ETF回测独立复核", "", "结论：全部通过。", "",
             f"- 配置和{len(metadata['price_hashes'])}只ETF行情输入哈希未变。",
             "- 重建完整情绪分与修复后的日报/看板完全一致。",
             f"- {len(decisions)}次周度决策的方向、确认、四道闸门和IC一致。",
             "- 所有订单均在信号后的下一A股交易日；买卖费用及不利滑点逐笔核对。",
             "- 独立按成交价重放现金/份额账本，所有账户现金和份额非负，累计收益与汇总一致。", ""]
    for row in checks:
        lines.append(f"- {row['name']}：{row['days']}日、{row['trades']}笔，最大净值误差{row['max_independent_nav_error']:.3g}。")
    (directory / "validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
