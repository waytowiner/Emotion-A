"""回测层口径测试：合成数据，锁定轮动回测的关键不变式。

1) T+1 执行：调仓信号收盘后生效，调仓当日收益归旧持仓（不前视）；
2) 调仓日为每周真实最后交易日：周五休市的周也要正常调仓（不静默跳周）。
"""

import numpy as np
import pandas as pd
import pytest

from emotion.backtest import (apply_confirm_hysteresis, equal_weight_portfolio,
                              ic_robustness, portfolio_turnover,
                              quantile_spread_series, rebalance_dates,
                              rolling_ic_validation, rotation_backtest)

W1 = pd.bdate_range("2026-01-05", periods=5)  # 1/5(一) ~ 1/9(五)
W2 = pd.bdate_range("2026-01-12", periods=5)  # 1/12(一) ~ 1/16(五)


class TestRebalanceTiming:
    def test_rebalance_does_not_capture_same_day_return(self):
        """中选当天（周五）的涨幅不应计入净值：分数收盘才可得。"""
        days = W1.append(W2)
        # 第1周五 A 分最高且 A 当天 +10%；第2周五 B 分最高且 B 当天 +10%
        score = pd.DataFrame(
            {"A": [0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
             "B": [1, 1, 1, 1, 1, 1, 1, 1, 1, 3]}, index=days, dtype=float)
        close = pd.DataFrame(
            {"A": [100, 100, 100, 100, 110, 110, 110, 110, 110, 110],
             "B": [100] * 9 + [110]}, index=days, dtype=float)
        nav = rotation_backtest(score, close, top_n=1, cost_bps=0)["nav"]
        # 两个调仓日若前视，净值会先变 1.1 再变 1.21；T+1 下应始终为 1.0
        assert nav[days[4]] == pytest.approx(1.0)
        assert nav[days[9]] == pytest.approx(1.0)

    def test_new_holdings_earn_from_next_day(self):
        """周五收盘买入的板块，下周一起正常享受收益。"""
        days = pd.bdate_range("2026-01-05", periods=6)  # 1/5(一) ~ 1/13(二)
        score = pd.DataFrame({"A": [0, 0, 0, 0, 2, 0],
                              "B": [1.0] * 6}, index=days)
        close = pd.DataFrame({"A": [100, 100, 100, 100, 100, 110],
                              "B": [100.0] * 6}, index=days)
        nav = rotation_backtest(score, close, top_n=1, cost_bps=0)["nav"]
        assert nav[days[4]] == pytest.approx(1.0)   # 调仓日不吃当天收益
        assert nav[days[5]] == pytest.approx(1.1)   # 次一交易日起生效

    def test_holiday_friday_week_still_rebalances(self):
        """周五休市时，调仓应发生在该周真实最后交易日（周四）。"""
        days = pd.DatetimeIndex(["2026-01-05", "2026-01-06", "2026-01-07",
                                 "2026-01-08", "2026-01-09",
                                 "2026-01-12", "2026-01-13", "2026-01-14",
                                 "2026-01-15",  # 第2周最后交易日（周五 1/16 休市）
                                 "2026-01-19"])
        # 第1周五 B 分高 -> 持 B；第2周周四 A 分高 -> 应换到 A
        score = pd.DataFrame({"A": [0, 0, 0, 0, 0, 0, 0, 0, 2, 0],
                              "B": [1.0] * 10}, index=days)
        # A 在下周一 1/19 +10%；只有周四成功换仓才能吃到
        close = pd.DataFrame({"A": [100] * 9 + [110],
                              "B": [100.0] * 10}, index=days)
        nav = rotation_backtest(score, close, top_n=1, cost_bps=0)["nav"]
        assert nav[pd.Timestamp("2026-01-19")] == pytest.approx(1.1)

    def test_switch_charges_cost(self):
        """换仓 1 只按单边成本扣费（价格全平盘，净值只剩成本损耗）。"""
        days = W1.append(W2)
        score = pd.DataFrame({"A": [0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
                              "B": [1.0] * 10}, index=days)
        close = pd.DataFrame({"A": [100.0] * 10, "B": [100.0] * 10}, index=days)
        rot = rotation_backtest(score, close, top_n=1, cost_bps=100)
        # 第2周五换 A->B：tc = 0.01 * 1 / 1
        assert rot["nav"][days[9]] == pytest.approx(0.99)
        assert rot["total_cost_drag"] == pytest.approx(0.01)

    def test_variable_size_turnover_is_weight_symmetric(self):
        """Top3 ↔ 全市场等权的换手必须相同，且不可能超过 100%。"""
        top3 = equal_weight_portfolio(["A", "B", "C"])
        broad = equal_weight_portfolio(["A", "B", "C"] + [f"X{i}" for i in range(97)])
        assert portfolio_turnover(top3, broad) == pytest.approx(0.97)
        assert portfolio_turnover(broad, top3) == pytest.approx(0.97)

    def test_open_last_week_is_not_a_historical_rebalance(self):
        days = pd.bdate_range("2026-01-05", periods=7)  # 最后一周只到周二
        dates = rebalance_dates(days, freq="W-FRI")
        assert dates.tolist() == [pd.Timestamp("2026-01-09")]


class TestIcRobustness:
    def test_hac_bootstrap_and_rolling_blocks_have_expected_shape(self):
        ic = pd.Series([0.05 + i * 0.001 for i in range(20)],
                       index=pd.bdate_range("2026-01-05", periods=20))
        stat = ic_robustness(ic, horizon=5, bootstrap_samples=100)
        assert stat["IC_nonoverlap_n"] == 4
        assert stat["IC_nonoverlap_mean"] > 0
        assert stat["IC_bootstrap_low"] > 0
        assert stat["IC_bootstrap_high"] >= stat["IC_bootstrap_low"]
        blocks = rolling_ic_validation(ic, block_days=7)
        assert blocks["n_obs"].tolist() == [7, 7, 6]
        assert pd.isna(blocks.iloc[0]["previous_IC_mean"])


class TestQuantileSpreadSeries:
    """分层价差（Q5−Q1）的符号必须跟随分组排序结构：动量结构为正、反转为负。"""

    def _make(self, rates: np.ndarray):
        days = pd.bdate_range("2026-01-05", periods=10)
        cols = [f"B{i:02d}" for i in range(20)]
        score = pd.DataFrame(np.tile(np.arange(20.0), (10, 1)),
                             index=days, columns=cols)
        close = pd.DataFrame(100 * np.cumprod(1 + np.tile(rates, (10, 1)), axis=0),
                             index=days, columns=cols)
        return score, close

    def test_spread_positive_in_momentum_structure(self):
        # 分高的板块每天涨得更多 → Q5−Q1 > 0（最后 5 日前瞻未走完，不出序列）
        score, close = self._make(np.linspace(-0.01, 0.01, 20))
        sp = quantile_spread_series(score, close, horizon=5)
        assert len(sp) == 5
        assert (sp > 0).all()

    def test_spread_negative_in_reversal_structure(self):
        # 分高的板块每天跌得更多 → Q5−Q1 < 0
        score, close = self._make(np.linspace(0.01, -0.01, 20))
        sp = quantile_spread_series(score, close, horizon=5)
        assert (sp < 0).all()


class TestConfirmHysteresis:
    def test_single_reading_does_not_flip(self):
        """单次反向读数不生效：先观望，连续 2 次同向才确认。"""
        d = pd.Series(["top", "top", "bottom", "bottom", "bottom"],
                      index=pd.bdate_range("2026-01-05", periods=5))
        out = apply_confirm_hysteresis(d, confirm=2)
        assert out.tolist() == ["neutral", "top", "neutral", "bottom", "bottom"]

    def test_neutral_immediate_and_resets_streak(self):
        """观望即刻生效，且会打断方向读数的连续计数。"""
        d = pd.Series(["top", "neutral", "top", "top"],
                      index=pd.bdate_range("2026-01-05", periods=4))
        out = apply_confirm_hysteresis(d, confirm=2)
        assert out.tolist() == ["neutral", "neutral", "neutral", "top"]

    def test_confirm_1_is_identity(self):
        d = pd.Series(["top", "bottom", "neutral"],
                      index=pd.bdate_range("2026-01-05", periods=3))
        assert apply_confirm_hysteresis(d, confirm=1).tolist() == d.tolist()
