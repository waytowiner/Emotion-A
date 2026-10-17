"""固定信号/行情下的执行期归因。归因参照须明确，不把事后对照称为因果证明。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def execution_periods(calendar, signal_dates):
    """周期从本次成交前开盘权益，到下次成交前开盘权益；末段到末日收盘。"""
    calendar = pd.DatetimeIndex(calendar)
    positions = calendar.get_indexer(signal_dates)
    if (positions < 0).any() or not calendar.is_monotonic_increasing or not calendar.is_unique:
        raise ValueError("信号必须位于有序且唯一的交易日历")
    if not pd.Index(signal_dates).is_unique or not pd.Index(signal_dates).is_monotonic_increasing:
        raise ValueError("信号日期须有序且唯一")
    pairs = [(signal, calendar[pos + 1]) for signal, pos in zip(signal_dates, positions)
             if pos + 1 < len(calendar)]
    records = []
    for i, (signal, start) in enumerate(pairs):
        terminal = i == len(pairs) - 1
        end = calendar[-1] if terminal else pairs[i + 1][1]
        records.append({"signal_date": signal, "entry_date": start, "exit_date": end,
                        "exit_mark": "close" if terminal else "open"})
    return pd.DataFrame(records).set_index("signal_date")


def gross_period_returns(targets, panels, periods):
    """相同无成本权重每期持有到下次开盘；只允许无冻结订单的归因样本。

    任一目标标的在成交日无价/无量时拒绝这种简式归因，避免把未成交仓位当成交。
    完整执行引擎仍负责处理冻结，此函数用于独立验证与固定时段的方向对照。
    """
    known_close = panels["close"].ffill()
    returns = {}
    for signal, row in periods.iterrows():
        weights = targets.loc[signal]
        codes = weights.index[weights > 1e-12]
        opening = panels["open"].loc[row.entry_date, codes]
        volume = panels["volume"].loc[row.entry_date, codes]
        if not (opening.gt(0) & volume.gt(0)).all():
            raise ValueError("成交日有缺失报价/无量，不能使用简式周期归因")
        if row.exit_mark == "close":
            ending = known_close.loc[row.exit_date, codes]
        else:
            ending = panels["open"].loc[row.exit_date, codes].astype(float).combine_first(
                known_close.shift(1).loc[row.exit_date, codes])
        if not ending.gt(0).all():
            raise ValueError("持仓缺少期末可知估值")
        returns[signal] = float((weights[codes] * (ending / opening - 1)).sum())
    return pd.Series(returns, name="period_return")


def account_periods(result, panels, periods):
    """从完整账户计算执行期收益；调仓日的隔夜收益归前一周期。"""
    daily = result["daily"]
    known_close = panels["close"].ffill()
    checkpoints = []
    for date in periods.entry_date:
        pos = daily.index.get_loc(date)
        if pos == 0:
            raise ValueError("首笔成交须发生于首个信号之后")
        previous = daily.index[pos - 1]
        marks = panels["open"].loc[date].where(panels["open"].loc[date] > 0).combine_first(
            known_close.loc[previous])
        ratio = marks / known_close.loc[previous]
        value = daily.at[previous, "nav"] * (
            daily.at[previous, "cash_weight"]
            + (result["holdings"].loc[previous] * ratio).fillna(0).sum())
        checkpoints.append(float(value))
    checkpoints.append(float(daily.nav.iloc[-1]))
    opening, ending = np.array(checkpoints[:-1]), np.array(checkpoints[1:])
    out = periods.copy()
    out["opening_equity"] = opening
    out["ending_equity"] = ending
    out["period_return"] = ending / opening - 1
    out["pnl_initial_equity"] = ending - opening
    out["entry_fee"] = daily.loc[periods.entry_date, "fee"].to_numpy()
    out["entry_slippage"] = daily.loc[periods.entry_date, "slippage"].to_numpy()
    return out


def active_return_bridge(gross, net, *, strategy, benchmark, anchor):
    """以固定方向为参照的精确收益差分解，单位为初始净值/收益百分点。

    排名项=固定方向无成本相对基准；切换项=自适应无成本相对固定方向；
    成本项=自适应与基准的成本损耗之差。更换参照会改变前两项，不改变合计。
    """
    rank = gross[anchor] - gross[benchmark]
    switching = gross[strategy] - gross[anchor]
    cost = (net[strategy] - gross[strategy]) - (net[benchmark] - gross[benchmark])
    total = net[strategy] - net[benchmark]
    if not np.isclose(rank + switching + cost, total, atol=1e-12):
        raise ValueError("归因桥接不平")
    return {"anchor": anchor, "ranking_pp": rank, "switching_pp": switching,
            "incremental_cost_pp": cost, "total_excess_pp": total}


def circular_timing_test(period_returns, directions):
    """固定当前方向序列的次数和相邻结构，循环错位作择时诊断；不作为显著性p值。"""
    columns = {"top": "fixed_top", "bottom": "fixed_bottom", "neutral": "equal_weight"}
    directions = np.asarray(directions)
    if len(period_returns) != len(directions):
        raise ValueError("方向与执行周期不一致")
    rows = []
    for offset in range(len(directions)):
        rotated = np.roll(directions, offset)
        values = [period_returns.iloc[i][columns[d]] for i, d in enumerate(rotated)]
        rows.append({"offset": offset, "gross_return": float(np.prod(1 + np.asarray(values)) - 1)})
    return pd.DataFrame(rows)
