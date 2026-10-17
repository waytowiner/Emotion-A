"""执行期边界、精确归因桥接与未来待执行信号的回归检查。"""
import numpy as np
import pandas as pd
import pytest

from emotion import attribution, etf_backtest


def market():
    dates = pd.to_datetime(["2026-09-24", "2026-09-28", "2026-09-29",
                            "2026-09-30", "2026-10-08", "2026-10-09"])
    opening = pd.DataFrame([[100, 100], [100, 100], [100, 100],
                            [200, 100], [200, 100], [200, 150]], index=dates, columns=["A", "B"])
    closing = opening.copy()
    targets = pd.DataFrame([[1, 0], [0, 1], [1, 0]], index=dates[[0, 2, 5]], columns=opening.columns)
    panels = {"open": opening, "close": closing, "volume": opening * 0 + 1000}
    return panels, targets


def test_period_boundary_assigns_overnight_to_old_position_and_omits_pending_signal():
    panels, targets = market()
    periods = attribution.execution_periods(panels["close"].index, targets.index)
    assert len(periods) == 2
    assert periods.entry_date.tolist() == list(pd.to_datetime(["2026-09-28", "2026-09-30"]))
    assert periods.exit_mark.tolist() == ["open", "close"]
    result = etf_backtest.simulate(panels["open"], panels["close"], panels["volume"], targets,
                                   fee_bps=0, slippage_bps=0)
    account = attribution.account_periods(result, panels, periods)
    direct = attribution.gross_period_returns(targets, panels, periods)
    assert direct.tolist() == pytest.approx([1.0, 0.5])
    assert account.period_return.tolist() == pytest.approx([1.0, 0.5])
    assert account.pnl_initial_equity.sum() == pytest.approx(2.0)
    assert (1 + account.period_return).prod() == pytest.approx(result["nav"].iloc[-1])


def test_entry_costs_belong_to_new_period_and_compound_to_full_net_nav():
    panels, targets = market()
    periods = attribution.execution_periods(panels["close"].index, targets.index)
    result = etf_backtest.simulate(panels["open"], panels["close"], panels["volume"], targets)
    account = attribution.account_periods(result, panels, periods)
    assert (account.entry_fee > 0).all()
    assert (account.entry_slippage > 0).all()
    assert account.period_return.iloc[0] == pytest.approx(2 / (1.0005 ** 2) - 1)
    assert (1 + account.period_return).prod() == pytest.approx(result["nav"].iloc[-1])
    assert account.entry_fee.sum() == pytest.approx(result["daily"].fee.sum())


def test_gross_shortcut_rejects_missing_execution_instead_of_inventing_fill():
    panels, targets = market()
    periods = attribution.execution_periods(panels["close"].index, targets.index)
    panels["open"].loc[periods.entry_date.iloc[0], "A"] = np.nan
    with pytest.raises(ValueError, match="缺失报价"):
        attribution.gross_period_returns(targets, panels, periods)


def test_return_bridge_is_exact_and_anchor_dependence_is_explicit():
    gross = {"top": .1, "bottom": -.2, "adaptive": -.05, "equal": .02}
    net = {"top": .08, "bottom": -.23, "adaptive": -.07, "equal": .01}
    top = attribution.active_return_bridge(gross, net, strategy="adaptive", benchmark="equal", anchor="top")
    bottom = attribution.active_return_bridge(gross, net, strategy="adaptive", benchmark="equal", anchor="bottom")
    assert top["ranking_pp"] == pytest.approx(.08)
    assert top["switching_pp"] == pytest.approx(-.15)
    assert bottom["switching_pp"] == pytest.approx(.15)
    assert top["incremental_cost_pp"] == pytest.approx(-.01)
    assert top["total_excess_pp"] == pytest.approx(bottom["total_excess_pp"])
    assert top["total_excess_pp"] == pytest.approx(-.08)


def test_circular_diagnostic_preserves_actual_path_at_zero_shift():
    returns = pd.DataFrame({"fixed_top": [.1, -.1, .2], "fixed_bottom": [-.1, .2, .1],
                            "equal_weight": [.01, .02, .03]})
    result = attribution.circular_timing_test(returns, ["top", "bottom", "neutral"])
    assert len(result) == 3
    assert result.iloc[0].gross_return == pytest.approx(1.1 * 1.2 * 1.03 - 1)
    assert result.iloc[1].gross_return == pytest.approx(1.01 * .9 * 1.1 - 1)
