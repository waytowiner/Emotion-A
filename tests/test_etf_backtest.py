"""成交时序、自融资费用、缺失行情与映射缺口的回归验证。"""
import numpy as np
import pandas as pd
import pytest

from emotion.etf_backtest import map_targets, simulate


def prices(open_values, close_values=None, codes=("A",)):
    dates = pd.to_datetime(["2026-09-30", "2026-10-08", "2026-10-09", "2026-10-12"])
    opening = pd.DataFrame(open_values, index=dates, columns=codes, dtype=float)
    close = opening.copy() if close_values is None else pd.DataFrame(close_values, index=dates, columns=codes, dtype=float)
    volume = opening * 0 + 1000
    return opening, close, volume


def test_signal_executes_next_actual_session_without_overnight_gain():
    op, cp, vol = prices([100, 200, 200, 200], [100, 220, 220, 220])
    target = pd.DataFrame({"A": [1.0]}, index=op.index[:1])
    result = simulate(op, cp, vol, target, fee_bps=0, slippage_bps=0)
    assert result["nav"].tolist() == pytest.approx([1, 1.1, 1.1, 1.1])
    trade = result["trades"].iloc[0]
    assert trade.execution_date == pd.Timestamp("2026-10-08")
    assert trade.signal_date == pd.Timestamp("2026-09-30")
    assert trade.open == 200


def test_old_holdings_receive_overnight_before_switching():
    op, cp, vol = prices([[100, 100], [100, 100], [200, 200], [200, 200]],
                         [[100, 100], [100, 100], [220, 300], [220, 300]], codes=("A", "B"))
    targets = pd.DataFrame([[1, 0], [0, 1]], index=op.index[:2], columns=op.columns)
    result = simulate(op, cp, vol, targets, fee_bps=0, slippage_bps=0)
    assert result["nav"].iloc[2] == pytest.approx(3.0)  # A隔夜翻倍，再赚B日内50%


def test_buy_sell_costs_and_slippage_both_count_with_no_negative_cash():
    op, cp, vol = prices([100, 100, 100, 100])
    targets = pd.DataFrame({"A": [1.0, 0.0]}, index=op.index[:2])
    result = simulate(op, cp, vol, targets, fee_bps=5, slippage_bps=5)
    fee = slip = 0.0005
    bought = 1 / ((1 + slip) * (1 + fee))
    final = bought * (1 - slip) * (1 - fee)
    assert result["nav"].iloc[1] == pytest.approx(bought)
    assert result["nav"].iloc[-1] == pytest.approx(final)
    assert result["trades"].side.tolist() == ["buy", "sell"]
    assert (result["trades"].fee > 0).all()
    assert (result["daily"].cash_weight >= 0).all()


def test_weights_drift_and_rebalancing_uses_current_holdings():
    op, cp, vol = prices([[100, 100], [100, 100], [200, 100], [200, 100]], codes=("A", "B"))
    targets = pd.DataFrame([[0.5, 0.5], [0.5, 0.5]], index=op.index[[0, 2]], columns=op.columns)
    result = simulate(op, cp, vol, targets, fee_bps=0, slippage_bps=0)
    assert result["holdings"].iloc[2].tolist() == pytest.approx([2/3, 1/3])
    assert result["holdings"].iloc[3].tolist() == pytest.approx([0.5, 0.5])
    assert result["daily"].gross_turnover.iloc[3] == pytest.approx(1/3)


def test_missing_execution_price_leaves_cash_and_does_not_renormalize():
    op, cp, vol = prices([[100, 100], [100, np.nan], [100, 100], [100, 100]], codes=("A", "B"))
    target = pd.DataFrame([[0.5, 0.5]], index=op.index[:1], columns=op.columns)
    result = simulate(op, cp, vol, target, fee_bps=0, slippage_bps=0)
    assert result["daily"].cash_weight.iloc[1] == pytest.approx(0.5)
    assert result["daily"].blocked_orders.iloc[1] == 1
    assert result["trades"].etf_code.tolist() == ["A"]


def test_suspended_holding_is_frozen_without_borrowing_and_staleness_logged():
    op, cp, vol = prices([[100, 100], [100, 100], [np.nan, 100], [100, 100]], codes=("A", "B"))
    targets = pd.DataFrame([[1, 0], [0, 1]], index=op.index[:2], columns=op.columns)
    result = simulate(op, cp, vol, targets, fee_bps=5, slippage_bps=5)
    assert result["daily"].blocked_orders.iloc[2] == 1
    assert result["daily"].stale_holdings.iloc[2] == 1
    assert result["holdings"].iloc[2]["A"] == pytest.approx(1)
    assert result["holdings"].iloc[2]["B"] == 0
    assert result["daily"].cash_weight.min() >= 0


def test_partially_frozen_holdings_reserve_cash_for_costs_before_buying():
    op, cp, vol = prices([[100, 100], [100, 100], [np.nan, 100], [100, 100]], codes=("A", "B"))
    targets = pd.DataFrame([[0.5, 0.5], [0, 1]], index=op.index[:2], columns=op.columns)
    result = simulate(op, cp, vol, targets, fee_bps=5, slippage_bps=5)
    assert result["daily"].cash_weight.min() >= 0
    assert result["holdings"].iloc[2].sum() == pytest.approx(1)
    # 不能卖A，也不能借款补满B；现有B只做微小费用约束调整。
    assert result["holdings"].iloc[2]["A"] >= 0.5 - 1e-10


def test_last_signal_pending_and_future_prices_cannot_change_earlier_nav():
    op, cp, vol = prices([100, 100, 100, 100])
    target = pd.DataFrame({"A": [1, 0]}, index=op.index[[0, 3]])
    result = simulate(op, cp, vol, target)
    assert result["pending_signals"] == [op.index[-1]]
    changed = cp.copy()
    changed.iloc[-1] = 999
    future = simulate(op, changed, vol, target)
    pd.testing.assert_series_equal(result["nav"].iloc[:-1], future["nav"].iloc[:-1])
    assert len(result["trades"]) == 1


def test_original_rank_unmapped_stays_cash_aliases_aggregate_and_listing_not_backfilled():
    op, cp, vol = prices([[100, np.nan], [100, np.nan], [100, 100], [100, 100]], codes=("A", "B"))
    scores = pd.DataFrame([[4, 3, 2, 1]]*4, index=cp.index, columns=["a", "alias", "unmapped", "new"])
    mapping = pd.DataFrame({"board_code": ["a", "alias", "new"], "etf_code": ["A", "A", "B"]})
    decisions = pd.DataFrame({"direction": ["top", "top"]}, index=cp.index[[0, 3]])
    target, audit = map_targets(scores, decisions, mapping, cp, vol, top_n=3)
    assert target.iloc[0]["A"] == pytest.approx(2/3)
    assert target.iloc[0]["B"] == 0
    assert audit.status.tolist().count("no_mapping") == 2
    mapped, _ = map_targets(scores, decisions, mapping, cp, vol, top_n=3, mapped_only=True)
    assert mapped.iloc[0]["A"] == pytest.approx(1)
    assert mapped.iloc[-1].tolist() == pytest.approx([2/3, 1/3])
