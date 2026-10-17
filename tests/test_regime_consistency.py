"""真实缺陷回归：假期 IC 可知时点、日报闸门、周度确认及无前视。"""
import numpy as np
import pandas as pd
import pytest

from emotion import backtest
from scripts.backtest_regime_adaptive import adaptive_parameters, adaptive_rotation
from scripts.backtest_portfolio_layer import decision_sequence


def market(periods=40, reversed_returns=False):
    dates = pd.bdate_range("2026-01-05", periods=periods)
    columns = [f"B{i:02d}" for i in range(20)]
    values = np.arange(20.0)
    scores = pd.DataFrame(np.tile(values, (periods, 1)), index=dates, columns=columns)
    rates = np.linspace(-0.005, 0.005, 20)
    if reversed_returns:
        rates = -rates
    close = pd.DataFrame(100 * np.cumprod(1 + np.tile(rates, (periods, 1)), axis=0),
                         index=dates, columns=columns)
    return scores, close


def test_five_day_ic_becomes_known_after_five_a_share_sessions():
    dates = pd.to_datetime(["2026-09-24", "2026-09-28", "2026-09-29",
                            "2026-09-30", "2026-10-08", "2026-10-09"])
    known = backtest.realized_series(pd.Series([0.2], index=dates[:1]), dates, lag=5)
    assert known.index.tolist() == [pd.Timestamp("2026-10-09")]


def test_unknown_ic_date_is_not_silently_shifted_to_next_session():
    with pytest.raises(ValueError, match="交易日历"):
        backtest.realized_series(pd.Series([0.2], index=pd.to_datetime(["2026-01-10"])),
                                 pd.bdate_range("2026-01-05", periods=10))


@pytest.mark.parametrize("gate", ["robust", "mono"])
def test_adaptive_backtest_honors_daily_gates(gate):
    scores, close = market(reversed_returns=gate == "mono")
    known = pd.Series(0.2, index=close.index)
    parameters = {"ic_window": 15, "threshold": 0.02, "min_obs": 10,
                  "guard_threshold": -99, "horizon": 5}
    if gate == "robust":
        known = pd.Series([-0.18, 0.20] * 20, index=close.index)
        parameters.update(require_robust_signal=True, robust_ic_window=15,
                          robust_bootstrap_samples=200)
    else:
        parameters["mono_days"] = 3
    cfg = {**parameters, "confirm_readings": 2}
    rot = adaptive_rotation(scores, close, known, **adaptive_parameters(cfg))
    last = rot["regime_log"].iloc[-1]
    daily = backtest.regime_state(scores, close, ic_known=known, **parameters)
    assert bool(last[gate + "_gate"]) is True
    assert daily[gate + "_gate"] is True
    assert last["direction"] == daily["direction"] == "neutral"
    assert last["ic"] == pytest.approx(daily["ic_mean"])
    assert last["n_holdings"] == 20


def test_confirmation_preserved_and_risk_gate_is_immediate():
    scores, close = market()
    known = pd.Series(0.2, index=close.index)
    mkt = pd.Series(0.0, index=close.index)
    mkt.iloc[-1] = -0.04
    rot = adaptive_rotation(scores, close, known, ic_window=15, min_obs=1,
                            confirm=2, guard=-0.03, mkt_ret=mkt)
    log = rot["regime_log"]
    assert log.iloc[0]["direction"] == "neutral"
    assert log.iloc[1]["direction"] == "top"
    assert log.iloc[-1]["confirmed_direction"] == "top"
    assert log.iloc[-1]["guard_hit"]
    assert log.iloc[-1]["direction"] == "neutral"


def test_injected_future_ic_and_spread_cannot_change_a_past_state():
    scores, close = market()
    date = close.index[24]
    known = pd.Series(0.2, index=close.index)
    spread = pd.Series(0.01, index=close.index)
    known.loc[known.index > date] = -0.8
    spread.loc[spread.index > date] = -0.5
    state = backtest.regime_state(scores.loc[:date], close.loc[:date], ic_known=known,
                                  spread_known=spread, mono_days=3, guard_threshold=-99)
    assert state["direction"] == "top"
    assert not state["mono_gate"]
    assert state["ic_mean"] == pytest.approx(0.2)


def test_snapshot_matches_rotation_and_research_baseline():
    scores, close = market()
    cfg = {"ic_window": 15, "threshold": 0.02, "min_obs": 10,
           "horizon": 5, "guard_threshold": -0.03, "confirm_readings": 2,
           "require_robust_signal": True, "robust_ic_window": 60,
           "robust_bootstrap_samples": 200, "mono_days": 3}
    known = backtest.realized_series(backtest.daily_ic_series(scores, close, 5), close.index)
    rot = adaptive_rotation(scores, close, known, **adaptive_parameters(cfg))
    last = rot["regime_log"].iloc[-1]
    snapshot = backtest.regime_snapshot(scores, close, cfg)
    assert snapshot["direction"] == last["state_direction"]
    assert snapshot["rebalance_direction"] == last["direction"]
    assert snapshot["rebalance_date"] == last["date"].strftime("%Y-%m-%d")
    _, _, directions, _ = decision_sequence(scores, close, known, ic_window=15,
                                            threshold=0.02, min_obs=10, confirm=2,
                                            regime_config=cfg)
    assert directions.tolist() == rot["regime_log"]["direction"].tolist()


def test_future_price_changes_do_not_change_past_rotation_decisions():
    scores, close = market()
    cfg = {"horizon": 5, "guard_threshold": -0.03, "confirm_readings": 2,
           "require_robust_signal": True, "mono_days": 3,
           "robust_bootstrap_samples": 100}
    def run(prices):
        known = backtest.realized_series(backtest.daily_ic_series(scores, prices, 5), prices.index)
        return adaptive_rotation(scores, prices, known, **adaptive_parameters(cfg))["regime_log"]
    cutoff = close.index[24]
    changed = close.copy()
    changed.loc[changed.index > cutoff] *= np.linspace(0.5, 1.5, 20)
    original, modified = run(close), run(changed)
    pd.testing.assert_frame_equal(original[original["date"] <= cutoff],
                                  modified[modified["date"] <= cutoff])
