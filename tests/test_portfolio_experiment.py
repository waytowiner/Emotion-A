"""去重与增量研究的前视、分组、同池和退出决策回归测试。"""
import numpy as np
import pandas as pd
import pytest

from emotion import portfolio_experiment as pe


def market():
    dates = pd.bdate_range("2025-09-01", periods=90)
    rng = np.random.default_rng(8)
    r = rng.normal(0, .01, (90, 3))
    r[:, 1] = r[:, 0]
    close = pd.DataFrame(100 * np.cumprod(1 + r, axis=0), index=dates, columns=["A", "B", "C"])
    volume = close * 0 + 1000
    scores = pd.DataFrame(np.tile([2., 0., -1.], (90, 1)), index=dates, columns=["a", "b", "c"])
    mapping = pd.DataFrame({"board_code": ["a", "b", "c"], "etf_code": ["A", "B", "C"]})
    return scores, mapping, close, volume


def test_complete_link_does_not_merge_a_low_correlation_chain():
    corr = pd.DataFrame([[1, .95, .5], [.95, 1, .96], [.5, .96, 1]], index=list("ABC"), columns=list("ABC"))
    assert pe.correlation_groups(corr, .9) == [["A", "B"], ["C"]]
    corr.loc["A", "B"] = corr.loc["B", "A"] = np.nan
    assert pe.correlation_groups(corr, .9) == [["A"], ["B", "C"]]


def test_same_representatives_and_bounded_tilt_without_emotion_selection():
    scores, mapping, close, volume = market()
    dates = close.index[[30, 60]]
    targets, audit, _ = pe.build_targets(scores, mapping, close, volume, dates)
    assert set(audit.representative) == {"A", "C"}
    b, c = targets["B_dedup_equal"], targets["C_dedup_emotion"]
    np.testing.assert_allclose(b.loc[:, ["A", "C"]], .5)
    np.testing.assert_allclose(c.loc[:, ["A", "C"]], [[.6, .4], [.6, .4]])
    assert (b.gt(0) == c.gt(0)).all().all()
    switched, _, _ = pe.build_targets(-scores, mapping, close, volume, dates)
    pd.testing.assert_frame_equal(switched["B_dedup_equal"], b)
    np.testing.assert_allclose(switched["C_dedup_emotion"].loc[:, ["A", "C"]], [[.4, .6], [.4, .6]])


def test_future_prices_scores_and_volume_do_not_change_past_targets():
    scores, mapping, close, volume = market()
    dates = close.index[[30, 60, 80]]
    original, audit, _ = pe.build_targets(scores, mapping, close, volume, dates)
    cutoff = dates[1]
    close.loc[close.index > cutoff] *= np.array([20, .01, 5])
    scores.loc[scores.index > cutoff] = 999
    volume.loc[volume.index > cutoff] = 0
    changed, changed_audit, _ = pe.build_targets(scores, mapping, close, volume, dates[:2])
    for name, frame in changed.items():
        pd.testing.assert_frame_equal(frame, original[name].loc[:cutoff])
    pd.testing.assert_frame_equal(changed_audit, audit[audit.signal_date <= cutoff])


def test_short_history_and_missing_observations_cannot_create_duplicates():
    scores, mapping, close, volume = market()
    targets, audit, diag = pe.build_targets(scores, mapping, close, volume, close.index[[5]])
    assert diag.groups.iloc[0] == 3
    close.loc[close.index[:25], "B"] = np.nan
    _, audit, diag = pe.build_targets(scores, mapping, close, volume, close.index[[30]])
    assert diag.groups.iloc[0] == 3
    assert audit.group_size.eq(1).all()


def test_aliases_are_averaged_and_all_accounts_share_signal_day_eligibility():
    scores, mapping, close, volume = market()
    scores["alias"] = 4.
    mapping = pd.concat([mapping, pd.DataFrame({"board_code": ["alias"], "etf_code": ["A"]})])
    date = close.index[30]
    volume.loc[date, "C"] = 0
    targets, audit, diag = pe.build_targets(scores, mapping, close, volume, [date])
    assert audit.set_index("etf_code").at["A", "etf_score"] == 3
    assert diag.eligible_etfs.iloc[0] == 2
    assert targets["A_monthly_equal"].loc[date].tolist() == [.5, .5, 0]
    assert all(frame.loc[date, "C"] == 0 for frame in targets.values())


def test_equal_scores_or_zero_amplitude_exactly_recover_dedup_baseline():
    scores, mapping, close, volume = market()
    for values, amplitude in ((scores, 0), (scores * 0, .2)):
        targets, _, _ = pe.build_targets(values, mapping, close, volume, close.index[[30]], amplitude=amplitude)
        pd.testing.assert_frame_equal(targets["B_dedup_equal"], targets["C_dedup_emotion"])
    assert pe.tilted_weights(pd.Series([9.]), .2).tolist() == [1.]


def test_monthly_schedule_keeps_first_signal_and_final_pending_date():
    dates = pd.to_datetime(["2026-08-07", "2026-08-14", "2026-09-04", "2026-09-11", "2026-10-09"])
    assert pe.monthly_signals(dates).tolist() == dates[[0, 2, 4]].tolist()
    with pytest.raises(ValueError):
        pe.monthly_signals(dates[::-1])


def test_exit_rule_requires_all_evidence_and_does_not_accept_small_positive_return_alone():
    summary = pd.DataFrame({"total_ret": [.01, .02, .03], "max_dd": [-.1, -.1, -.09],
                            "stress_ret": [.009, .019, .029]},
                           index=["A_monthly_equal", "B_dedup_equal", "C_dedup_emotion"])
    assert all(pe.exit_checks(summary, .00001, pd.Series([.001] * 12)).values())
    checks = pe.exit_checks(summary, -.0001, pd.Series([.001] * 11))
    assert not checks["paired_95pct_interval_positive"]
    assert not checks["completed_period_stability"]
    summary.loc["C_dedup_emotion", "stress_ret"] = .018
    summary.loc["C_dedup_emotion", "max_dd"] = -.11
    checks = pe.exit_checks(summary, .00001, pd.Series([.001] * 12))
    assert not checks["double_cost_C_beats_B"]
    assert not checks["drawdown_C_no_worse_than_B"]
