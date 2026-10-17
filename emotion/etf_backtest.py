"""ETF 执行层：收盘后目标权重，下一交易日开盘成交，真实份额漂移。

输入为连续复权 OHLC，使用归一化虚拟份额（不做整手撮合）。费用与滑点按
每笔买卖成交额分别扣除；不填补缺失成交价，不把缺失目标权重转给其余基金。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def map_targets(scores, decisions, mapping, close, volume, *, top_n=3,
                mapped_only=False):
    """保持原始方向；原池缺失映射留现金，mapped_only 为明确的交易池对照。"""
    lookup = mapping.set_index("board_code")["etf_code"].to_dict()
    columns = close.columns
    targets, audit = [], []
    for date, decision in decisions.iterrows():
        values = scores.loc[date].dropna()
        eligible = {board for board, code in lookup.items()
                    if code in columns and pd.notna(close.at[date, code])
                    and close.at[date, code] > 0 and volume.at[date, code] > 0}
        if mapped_only:
            values = values[values.index.isin(eligible)]
        direction = decision["direction"]
        pick = (values.nlargest(top_n) if direction == "top" else
                values.nsmallest(top_n) if direction == "bottom" else values)
        weights = pd.Series(0.0, index=columns)
        for board in pick.index:
            if board in eligible:
                weights[lookup[board]] += 1 / len(pick)
            audit.append({"signal_date": date, "direction": direction,
                          "board_code": board, "etf_code": lookup.get(board),
                          "target_weight": 1 / len(pick),
                          "status": "mapped" if board in eligible else
                          "no_mapping" if board not in lookup else "no_signal_day_quote"})
        targets.append(weights.rename(date))
    frame = pd.DataFrame(targets, columns=columns)
    frame.index = pd.DatetimeIndex(frame.index, name="signal_date")
    return frame, pd.DataFrame(audit)


def simulate(open_price: pd.DataFrame, close: pd.DataFrame, volume: pd.DataFrame,
             targets: pd.DataFrame, *, fee_bps=5.0, slippage_bps=5.0) -> dict:
    """自融资现金/份额账本，含首次建仓费，不在末日虚构平仓。

    下一交易日由价格面板的 A 股日历确定。不能成交时原持仓冻结、买单留现金；
    缺失收盘仅用于原持仓的账面估值延用已知价格，明记 stale_holdings。
    通过求解交易后权益保证扣费后不借款、不负现金，且目标权重以交易后权益计。
    """
    if fee_bps < 0 or slippage_bps < 0 or slippage_bps >= 10000:
        raise ValueError("费用/滑点必须非负且滑点小于10000bps")
    dates, codes = close.index, close.columns
    if not dates.is_unique or not dates.is_monotonic_increasing:
        raise ValueError("交易日历须升序且无重复")
    if not open_price.index.equals(dates) or not volume.index.equals(dates):
        raise ValueError("OHLC/成交量日历不一致")
    if not open_price.columns.equals(codes) or not volume.columns.equals(codes):
        raise ValueError("行情代码不一致")
    target = targets.reindex(columns=codes, fill_value=0.0).fillna(0.0)
    if (target < 0).any().any() or (target.sum(axis=1) > 1 + 1e-10).any():
        raise ValueError("目标权重须非负，总和不得超过1")
    if not target.index.is_unique or not target.index.isin(dates).all():
        raise ValueError("信号日期须唯一且在交易日历中")
    schedule, pending = {}, []
    for signal_date in target.index:
        pos = dates.get_loc(signal_date) + 1
        if pos < len(dates):
            schedule[dates[pos]] = signal_date
        else:
            pending.append(signal_date)
    shares = np.zeros(len(codes))
    cash, previous_nav = 1.0, 1.0
    last_close = np.full(len(codes), np.nan)
    fee, slip = fee_bps / 10000, slippage_bps / 10000
    rows, trades, holdings = [], [], []
    for date in dates:
        op = open_price.loc[date].to_numpy(dtype=float)
        cp = close.loc[date].to_numpy(dtype=float)
        vol = volume.loc[date].to_numpy(dtype=float)
        valid_open = np.isfinite(op) & (op > 0) & np.isfinite(vol) & (vol > 0)
        mark_open = np.where(np.isfinite(op) & (op > 0), op, last_close)
        mark_open = np.nan_to_num(mark_open)
        opening_equity = cash + float(shares @ mark_open)
        day_fee = day_slip = traded = 0.0
        blocked = 0
        signal_date = schedule.get(date)
        if signal_date is not None:
            desired = target.loc[signal_date].to_numpy(dtype=float)
            blocked = int(((desired > 0) | (shares > 1e-12))[~valid_open].sum())
            desired = np.where(valid_open, desired, 0.0)
            frozen_value = float(shares[~valid_open] @ mark_open[~valid_open])

            def proposed(equity):
                new = shares.copy()
                allocation = desired * equity
                if allocation.sum() > 0:
                    allocation *= min(1.0, max(0.0, equity - frozen_value) / allocation.sum())
                new[valid_open] = allocation[valid_open] / op[valid_open]
                delta = new - shares
                notionals = np.abs(delta) * mark_open
                executed = notionals * (1 + np.sign(delta) * slip)
                return new, delta, notionals, executed * fee, notionals * slip

            # 自融资约束：交易后权益 = 开盘权益 - 费用 - 滑点。
            low, high = 0.0, opening_equity
            for _ in range(50):
                mid = (low + high) / 2
                _, _, _, commissions, slippages = proposed(mid)
                if mid + commissions.sum() + slippages.sum() > opening_equity:
                    high = mid
                else:
                    low = mid
            new, delta, notionals, commissions, slippages = proposed((low + high) / 2)
            day_fee, day_slip = float(commissions.sum()), float(slippages.sum())
            traded = float(notionals.sum())
            cash -= float(delta @ mark_open) + day_fee + day_slip
            if cash < -1e-9:
                raise RuntimeError("交易后现金为负")
            cash = max(cash, 0.0)
            for j in np.flatnonzero(np.abs(delta) > 1e-12):
                trades.append({"signal_date": signal_date, "execution_date": date,
                               "etf_code": codes[j], "side": "buy" if delta[j] > 0 else "sell",
                               "adjusted_units": float(delta[j]), "open": float(op[j]),
                               "execution_price": float(op[j] * (1 + np.sign(delta[j]) * slip)),
                               "notional": float(notionals[j]), "fee": float(commissions[j]),
                               "slippage": float(slippages[j])})
            shares = new
        valid_close = np.isfinite(cp) & (cp > 0)
        mark_close = np.where(valid_close, cp, last_close)
        if ((shares > 1e-12) & ~np.isfinite(mark_close)).any():
            raise ValueError("持仓无可知估值价格")
        stale = int(((shares > 1e-12) & ~valid_close).sum())
        mark_close = np.nan_to_num(mark_close)
        values = shares * mark_close
        nav = cash + float(values.sum())
        rows.append({"date": date, "nav": nav, "return": nav / previous_nav - 1,
                     "cash_weight": cash / nav, "fee": day_fee, "slippage": day_slip,
                     "traded_notional": traded, "gross_turnover": traded / opening_equity,
                     "n_holdings": int((shares > 1e-12).sum()),
                     "blocked_orders": blocked, "stale_holdings": stale,
                     "signal_date": signal_date})
        holdings.append(pd.Series(values / nav, index=codes, name=date))
        last_close = np.where(valid_close, cp, last_close)
        previous_nav = nav
    daily = pd.DataFrame(rows).set_index("date")
    return {"daily": daily, "nav": daily["nav"], "trades": pd.DataFrame(trades),
            "holdings": pd.DataFrame(holdings), "pending_signals": pending}
