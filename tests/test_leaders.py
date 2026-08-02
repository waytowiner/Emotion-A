"""emotion/leaders.py 单元测试：合成数据，不依赖网络与缓存。"""

from __future__ import annotations

import pandas as pd
import pytest

from emotion.leaders import (compute_leaders, liquidity_leaders,
                             momentum_leaders)

DATES = pd.bdate_range("2026-06-01", periods=12)  # 12 个交易日


def _row(date, code, name, amount, *, limit_up=False, trade_status="1"):
    """合成一行个股日线。limit_up=True 时构造满足 9.8% 主板涨停口径的数据。"""
    if limit_up:
        close, high, pct = 11.0, 11.0, 10.0   # prev=10, high_pct=10 ≥ 9.8，收盘封板
    else:
        close, high, pct = 10.0, 10.2, 2.0
    return {"date": date, "code": code, "name": name, "close": close,
            "high": high, "pct_chg": pct, "amount": amount,
            "trade_status": trade_status}


@pytest.fixture
def sample():
    """两个板块：
    B1：600001~600006（含停牌股、3 连板股、单日涨停股）
    B2：600010/600011（无涨停）
    """
    rows = []
    for i, dt in enumerate(DATES):
        in_liq_win = i >= len(DATES) - 5
        # B1 成分股：成交额固定区分名次
        rows.append(_row(dt, "600001", "流动王", 30e8 if in_liq_win else 1e8))
        rows.append(_row(dt, "600002", "老二", 20e8 if in_liq_win else 1e8))
        rows.append(_row(dt, "600003", "老三", 10e8 if in_liq_win else 1e8))
        # 停牌股：近 5 日停牌但挂着巨额成交，必须被剔除
        rows.append(_row(dt, "600004", "停牌哥", 99e8,
                         trade_status="0" if in_liq_win else "1"))
        # 情绪股：600005 在第 7/8/9 天三连板；600006 第 10 天单日涨停
        rows.append(_row(dt, "600005", "连板王", 5e8,
                         limit_up=i in (6, 7, 8)))
        rows.append(_row(dt, "600006", "首板", 5e8, limit_up=(i == 9)))
        # B2：平平无奇
        rows.append(_row(dt, "600010", "稳重", 8e8))
        rows.append(_row(dt, "600011", "躺平", 6e8))

    stock_daily = pd.DataFrame(rows)
    board_map = pd.DataFrame({
        "code": ["600001", "600002", "600003", "600004", "600005", "600006",
                 "600010", "600011"],
        "board_code": ["B1"] * 6 + ["B2"] * 2,
        "board_name": ["板块一"] * 6 + ["板块二"] * 2,
    })
    return stock_daily, board_map


def test_liquidity_ranks_top3_by_avg_amount(sample):
    stock_daily, board_map = sample
    out = liquidity_leaders(stock_daily, board_map)
    b1 = out["B1"]
    # 近 5 日日均额：600001(30亿) > 600002(20亿) > 600003(10亿)，取前 3
    assert [s["code"] for s in b1] == ["600001", "600002", "600003"]
    assert b1[0]["name"] == "流动王"
    assert b1[0]["avg_amount"] == pytest.approx(30e8)
    assert len(b1) == 3


def test_liquidity_excludes_suspended(sample):
    stock_daily, board_map = sample
    out = liquidity_leaders(stock_daily, board_map)
    assert "600004" not in [s["code"] for s in out["B1"]]


def test_momentum_streak_and_order(sample):
    stock_daily, board_map = sample
    out = momentum_leaders(stock_daily, board_map)
    b1 = out["B1"]
    assert b1[0]["code"] == "600005"
    assert b1[0]["max_streak"] == 3          # 三连板
    assert b1[1]["code"] == "600006"
    assert b1[1]["max_streak"] == 1          # 单日涨停
    assert b1[0]["name"] == "连板王"


def test_momentum_empty_without_limit_up(sample):
    stock_daily, board_map = sample
    out = momentum_leaders(stock_daily, board_map)
    assert "B2" not in out                    # 无涨停记录的板块不出现


def test_compute_leaders_structure(sample):
    stock_daily, board_map = sample
    out = compute_leaders(stock_daily, board_map)
    assert set(out.keys()) == {"B1", "B2"}    # 覆盖全部板块
    for entry in out.values():
        assert set(entry.keys()) == {"liquidity", "momentum"}
    assert out["B2"]["momentum"] == []        # 无涨停 -> 空列表
    assert out["B2"]["liquidity"][0]["code"] == "600010"  # 8亿 > 6亿
    liq = out["B1"]["liquidity"][0]
    assert set(liq.keys()) == {"code", "name", "avg_amount"}
    mom = out["B1"]["momentum"][0]
    assert set(mom.keys()) == {"code", "name", "max_streak"}
