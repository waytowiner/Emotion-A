"""因子层口径测试：全部使用合成数据，不依赖网络。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from emotion.factors import (
    factor_breadth,
    factor_limit_up_strength,
    factor_money_effect,
    factor_promotion_rate,
    factor_rps,
    factor_turnover_heat,
    flag_limit_up,
    limit_threshold,
)

D1, D2, D3 = pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-06"), pd.Timestamp("2026-01-07")


def _row(date, code, close, high, pct, name="测试", status="1", amount=1e8):
    return {"date": date, "code": code, "close": close, "high": high,
            "pct_chg": pct, "name": name, "trade_status": status, "amount": amount}


class TestLimitThreshold:
    def test_main_board(self):
        assert limit_threshold("600519", "贵州茅台") == 9.8
        assert limit_threshold("000001", "平安银行") == 9.8

    def test_chinext_star(self):
        assert limit_threshold("300750", "宁德时代") == 19.8
        assert limit_threshold("688981", "中芯国际") == 19.8

    def test_st(self):
        assert limit_threshold("600000", "ST测试") == 4.8
        assert limit_threshold("000002", "*ST某某") == 4.8

    def test_st_on_chinext_star_still_20(self):
        # 创业板/科创板 ST 涨跌停仍为 20%，不应被降为 5%
        assert limit_threshold("300750", "ST测试") == 19.8
        assert limit_threshold("688981", "*ST某某") == 19.8
        # 显式 is_st 参数同样不影响 30/68
        assert limit_threshold("300750", "宁德时代", is_st=True) == 19.8


class TestFlagLimitUp:
    def test_sealed_limit_up(self):
        # 主板：开盘 10 元，最高/收盘 11 元 → 涨 10%，封板
        df = pd.DataFrame([_row(D1, "600000", 11.0, 11.0, 10.0)])
        out = flag_limit_up(df)
        assert out.iloc[0]["is_limit_up"] is np.True_ or out.iloc[0]["is_limit_up"]
        assert not out.iloc[0]["is_touch"]

    def test_broken_board_is_touch(self):
        # 最高 11 元触板，收盘 10.5 元 → 炸板
        df = pd.DataFrame([_row(D1, "600000", 10.5, 11.0, 5.0)])
        out = flag_limit_up(df)
        assert not out.iloc[0]["is_limit_up"]
        assert out.iloc[0]["is_touch"]

    def test_normal_gain_not_limit(self):
        df = pd.DataFrame([_row(D1, "600000", 10.9, 10.9, 9.0)])
        out = flag_limit_up(df)
        assert not out.iloc[0]["is_limit_up"]
        assert not out.iloc[0]["is_touch"]

    def test_st_limit(self):
        # ST 股 5% 涨停：10 → 10.5 封板
        df = pd.DataFrame([_row(D1, "600001", 10.5, 10.5, 5.0, name="ST测试")])
        out = flag_limit_up(df)
        assert out.iloc[0]["is_limit_up"]


def _fixture_two_boards():
    """两个板块各两只股票，D1 各涨停一只，D2 观察次日表现。"""
    rows = [
        # 板块A 股票1：D1 涨停，D2 涨 3%；股票2：D1 平盘，D2 跌 1%
        _row(D1, "600001", 11.0, 11.0, 10.0), _row(D2, "600001", 11.33, 11.4, 3.0),
        _row(D1, "600002", 10.0, 10.0, 0.0), _row(D2, "600002", 9.9, 10.0, -1.0),
        # 板块B 股票3：D1 涨停，D2 跌 2%；股票4：D1 涨 2%，D2 涨 1%
        _row(D1, "000003", 11.0, 11.0, 10.0), _row(D2, "000003", 10.78, 11.0, -2.0),
        _row(D1, "000004", 10.2, 10.2, 2.0), _row(D2, "000004", 10.3, 10.3, 1.0),
    ]
    daily = pd.DataFrame(rows)
    board_map = pd.DataFrame({
        "code": ["600001", "600002", "000003", "000004"],
        "board_code": ["A", "A", "B", "B"],
    })
    return flag_limit_up(daily), board_map


class TestFactors:
    def test_limit_up_strength(self):
        flagged, bm = _fixture_two_boards()
        f = factor_limit_up_strength(flagged, bm)
        assert f.loc[D1, "A"] == pytest.approx(0.5)
        assert f.loc[D1, "B"] == pytest.approx(0.5)
        assert f.loc[D2, "A"] == pytest.approx(0.0)

    def test_money_effect_uses_median(self):
        flagged, bm = _fixture_two_boards()
        f = factor_money_effect(flagged, bm)
        # D2：板块A 昨日涨停股今日 +3%，板块B 昨日涨停股今日 -2%
        assert f.loc[D2, "A"] == pytest.approx(3.0)
        assert f.loc[D2, "B"] == pytest.approx(-2.0)
        # D1 没有昨日数据，不产生记录
        assert D1 not in f.index

    def test_breadth(self):
        flagged, bm = _fixture_two_boards()
        f = factor_breadth(flagged, bm)
        # D1 A：1涨1平 → (1-0)/2；B：2涨 → 1.0
        assert f.loc[D1, "A"] == pytest.approx(0.5)
        assert f.loc[D1, "B"] == pytest.approx(1.0)
        # D2 A：1涨1跌 → 0
        assert f.loc[D2, "A"] == pytest.approx(0.0)

    def test_promotion_rate_min_sample(self):
        flagged, bm = _fixture_two_boards()
        # 每板块昨日涨停仅 1 只（<2），晋级率应为 NaN
        f = factor_promotion_rate(flagged, bm)
        if D2 in f.index and "A" in f.columns:
            assert pd.isna(f.loc[D2, "A"])

    def test_promotion_rate_computed(self):
        # 板块C 三只股票 D1 涨停，D2 两只继续涨停 → 2/3
        rows = []
        for i, code in enumerate(["600011", "600012", "600013"]):
            rows.append(_row(D1, code, 11.0, 11.0, 10.0))
        rows += [
            _row(D2, "600011", 12.1, 12.1, 10.0),
            _row(D2, "600012", 12.1, 12.1, 10.0),
            _row(D2, "600013", 11.5, 11.6, 4.5),
        ]
        flagged = flag_limit_up(pd.DataFrame(rows))
        bm = pd.DataFrame({"code": ["600011", "600012", "600013"],
                           "board_code": ["C", "C", "C"]})
        f = factor_promotion_rate(flagged, bm)
        assert f.loc[D2, "C"] == pytest.approx(2 / 3)

    def test_rps_rank(self):
        k = pd.DataFrame({
            "board_code": ["A"] * 3 + ["B"] * 3,
            "date": [D1, D2, D3] * 2,
            "close": [100, 110, 121, 100, 100, 100],
        })
        f = factor_rps(k, window=1)
        # D2：A 涨10% 排第一(分位1.0)，B 平盘排最后(0.5，rank 平均法升序)
        assert f.loc[D2, "A"] == pytest.approx(1.0)
        assert f.loc[D2, "B"] == pytest.approx(0.5)

    def test_turnover_heat(self):
        flagged, bm = _fixture_two_boards()
        f = factor_turnover_heat(flagged, bm, window=1)
        # 两板块成交额相等 → 占比变化为 0
        assert f.loc[D2, "A"] == pytest.approx(0.0)

    def test_limit_up_strength_excludes_suspended(self):
        """停牌股不计入涨停强度分母（与宽度因子同口径）。"""
        rows = [
            _row(D1, "600001", 11.0, 11.0, 10.0),              # 涨停
            _row(D1, "600002", 10.0, 10.0, 0.0),               # 正常交易
            _row(D1, "600003", 10.0, 10.0, 0.0, status="0"),   # 停牌
        ]
        flagged = flag_limit_up(pd.DataFrame(rows))
        bm = pd.DataFrame({"code": ["600001", "600002", "600003"],
                           "board_code": ["A", "A", "A"]})
        f = factor_limit_up_strength(flagged, bm)
        # 分母只算 2 只交易中的股票：1/2，而非 1/3
        assert f.loc[D1, "A"] == pytest.approx(0.5)

    def test_turnover_heat_market_total(self):
        """提供全市场成交额时，占比为板块成交额 / 全市场成交额。"""
        flagged, bm = _fixture_two_boards()  # 每股每日 amount=1e8，板块池合计 4e8
        dates = [D1, D2]
        # 全市场成交额是板块池的 2 倍 -> 两板块占比各 25%
        market_total = pd.Series([8e8, 8e8], index=pd.DatetimeIndex(dates))
        f = factor_turnover_heat(flagged, bm, window=1, market_total=market_total)
        assert f.loc[D2, "A"] == pytest.approx(0.0)   # 占比无变化仍为 0
        # 全市场 D2 缩量一半 -> 板块占比从 25% 升到 50%，变化 +25 个百分点
        market_total2 = pd.Series([8e8, 4e8], index=pd.DatetimeIndex(dates))
        f2 = factor_turnover_heat(flagged, bm, window=1, market_total=market_total2)
        assert f2.loc[D2, "A"] == pytest.approx(25.0)
