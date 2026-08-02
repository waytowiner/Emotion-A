"""回测层口径测试：合成数据，锁定轮动回测的关键不变式。

1) T+1 执行：调仓信号收盘后生效，调仓当日收益归旧持仓（不前视）；
2) 调仓日为每周真实最后交易日：周五休市的周也要正常调仓（不静默跳周）。
"""

import pandas as pd
import pytest

from emotion.backtest import rotation_backtest

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
