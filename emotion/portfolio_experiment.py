"""最后一轮固定规则研究：仅用信号日已知价格去重，情绪只作有界权重倾斜。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def monthly_signals(weekly_dates):
    dates = pd.DatetimeIndex(weekly_dates)
    if not dates.is_unique or not dates.is_monotonic_increasing:
        raise ValueError("信号日期必须升序且唯一")
    return dates[~dates.to_period("M").duplicated()]


def correlation_groups(correlation, threshold):
    """确定性的贪心完全连接分组；禁止单链把低相关的两端合到一起。"""
    remaining = sorted(correlation.index)
    groups = []
    while remaining:
        group = [remaining.pop(0)]
        for code in remaining.copy():
            if correlation.loc[code, group].ge(threshold).all():
                group.append(code)
                remaining.remove(code)
        groups.append(group)
    return groups


def representative(group, correlation):
    if len(group) == 1:
        return group[0]
    means = correlation.loc[group, group].mean(axis=1)
    return sorted(group, key=lambda code: (-means[code], code))[0]


def tilted_weights(group_scores, amplitude):
    if not 0 <= amplitude < 1:
        raise ValueError("倾斜幅度须在[0,1)内")
    n = len(group_scores)
    if n == 0:
        return group_scores.copy()
    rank = group_scores.rank(method="average")
    centered = 2 * (rank - 1) / (n - 1) - 1 if n > 1 else rank * 0
    weights = 1 + amplitude * centered
    return weights / weights.sum()


def build_targets(scores, mapping, close, volume, signal_dates, *, lookback=60,
                  min_observations=20, threshold=0.9, amplitude=0.2):
    """A/B/C共享当日可用池、信号与代表基金。无历史证据的标的保持独立。"""
    mapping = mapping.set_index("board_code").etf_code
    if not mapping.index.is_unique:
        raise ValueError("板块映射必须唯一")
    returns = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    targets = {name: pd.DataFrame(0.0, index=signal_dates, columns=close.columns)
               for name in ("A_monthly_equal", "B_dedup_equal", "C_dedup_emotion")}
    audit, diagnostics = [], []
    for date in signal_dates:
        values = scores.loc[date].replace([np.inf, -np.inf], np.nan).dropna()
        values = values[values.index.isin(mapping.index)]
        codes = mapping.reindex(values.index)
        etf_scores = pd.Series(values.to_numpy(), index=codes.to_numpy()).groupby(level=0).mean()
        eligible = close.loc[date].gt(0) & volume.loc[date].gt(0)
        etf_scores = etf_scores.reindex(eligible.index[eligible]).dropna().sort_index()
        if etf_scores.empty:
            raise ValueError(f"{date}: 相同候选池为空，拒绝静默改成现金")
        hist = returns.loc[:date, etf_scores.index].tail(lookback)
        corr = hist.corr(min_periods=min_observations)
        groups = correlation_groups(corr, threshold)
        reps = [representative(group, corr) for group in groups]
        group_scores = pd.Series([etf_scores[group].mean() for group in groups], index=reps)
        tilt = tilted_weights(group_scores, amplitude)
        targets["A_monthly_equal"].loc[date, etf_scores.index] = 1 / len(etf_scores)
        targets["B_dedup_equal"].loc[date, reps] = 1 / len(groups)
        targets["C_dedup_emotion"].loc[date, reps] = tilt
        for group_id, (group, rep) in enumerate(zip(groups, reps)):
            sub = corr.loc[group, group].to_numpy()
            pairs = sub[np.triu_indices(len(group), 1)]
            observed = hist[group].notna().astype(int).T @ hist[group].notna().astype(int)
            pair_counts = observed.to_numpy()[np.triu_indices(len(group), 1)]
            for code in group:
                audit.append({"signal_date": date, "group_id": group_id,
                              "etf_code": code, "representative": rep, "group_size": len(group),
                              "etf_score": etf_scores[code], "group_score": group_scores[rep],
                              "B_group_weight": 1 / len(groups), "C_group_weight": tilt[rep],
                              "minimum_pair_correlation": float(pairs.min()) if len(pairs) else np.nan,
                              "minimum_pair_observations": int(pair_counts.min()) if len(pairs) else 0,
                              "history_end": hist.index[-1]})
        pairs = corr.loc[reps, reps].to_numpy()[np.triu_indices(len(reps), 1)]
        diagnostics.append({"signal_date": date, "eligible_etfs": len(etf_scores),
                            "groups": len(groups), "merged_etfs": len(etf_scores) - len(groups),
                            "largest_group": max(map(len, groups)),
                            "remaining_high_correlation_rep_pairs": int((pairs >= threshold).sum()),
                            "B_effective_holdings": len(groups),
                            "C_effective_holdings": 1 / float((tilt ** 2).sum()),
                            "max_C_weight": tilt.max()})
    for frame in targets.values():
        frame.index.name = "signal_date"
    return targets, pd.DataFrame(audit), pd.DataFrame(diagnostics)


def exit_checks(summary, interval_low, completed_excess):
    """固定的合取退出规则；证据不足也退出，不把不显著包装成成功。"""
    a, b, c = (summary.loc[key] for key in
               ("A_monthly_equal", "B_dedup_equal", "C_dedup_emotion"))
    stable = len(completed_excess) >= 12 and completed_excess.gt(0).mean() >= 2 / 3
    checks = {
        "net_C_beats_B": c.total_ret > b.total_ret,
        "net_C_beats_A": c.total_ret > a.total_ret,
        "drawdown_C_no_worse_than_B": c.max_dd >= b.max_dd - 1e-10,
        "paired_95pct_interval_positive": np.isfinite(interval_low) and interval_low > 0,
        "completed_period_stability": stable,
        "double_cost_C_beats_B": c.stress_ret > b.stress_ret,
    }
    return {key: bool(value) for key, value in checks.items()}
