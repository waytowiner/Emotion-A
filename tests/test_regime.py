"""regime_state 风格判定测试：合成数据，锁死三件事。

1) 无前视：最后 horizon 日的前瞻收益尚未实现，不得参与判定；
2) 方向判定：滚动 IC 越阈值给 top/bottom，中间观望；
3) 回撤护栏/波动闸门：触发即强制观望。

合成数据口径：每个板块有固定"质量"分，日收益 = 噪声 + 质量×系数，
使情绪分截面与 5 日前瞻收益截面稳定相关（正相关=动量市，负相关=反转市）。
"""

import numpy as np
import pandas as pd
import pytest

from emotion.backtest import regime_state

rng = np.random.default_rng(7)
DAYS = pd.bdate_range("2026-01-05", periods=80)
BOARDS = [f"B{i:02d}" for i in range(20)]
# 固定质量：B00 最高，B19 最低（rank 归一到 [-0.5, 0.5]）
QUALITY = pd.Series(np.linspace(0.5, -0.5, len(BOARDS)), index=BOARDS)


def _make(corr=1.0, drift=0.0, day_ret_scale=0.01):
    """corr=1：高质量板块未来收益高（动量市）；corr=-1：反过来（反转市）。"""
    noise = rng.normal(0, 0.02, (len(DAYS), len(BOARDS)))
    daily_ret = (drift + noise + corr * day_ret_scale * QUALITY.values)
    rets = pd.DataFrame(daily_ret, index=DAYS, columns=BOARDS)
    close = (1 + rets).cumprod() * 100
    score = pd.DataFrame(
        np.tile(QUALITY.values, (len(DAYS), 1))
        + rng.normal(0, 0.1, (len(DAYS), len(BOARDS))),
        index=DAYS, columns=BOARDS)
    return score, close


class TestRegimeState:
    def test_positive_ic_gives_top(self):
        score, close = _make(corr=1.0)
        st = regime_state(score, close, ic_window=15, threshold=0.02,
                          min_obs=10, guard_threshold=-99)  # 护栏关闭
        assert st["direction"] == "top"
        assert st["ic_mean"] > 0.02
        assert st["n_obs"] == 15

    def test_negative_ic_gives_bottom(self):
        score, close = _make(corr=-1.0)
        st = regime_state(score, close, ic_window=15, threshold=0.02,
                          min_obs=10, guard_threshold=-99)
        assert st["direction"] == "bottom"
        assert st["ic_mean"] < -0.02

    def test_no_lookahead_last_days_excluded(self):
        """决策日 T 只使用 d+5 ≤ T 的已实现 IC：最后 5 个交易日的 IC 不得入样本。"""
        from emotion.backtest import daily_ic_series
        score, close = _make(corr=1.0)
        st = regime_state(score, close, ic_window=15, threshold=0.02,
                          min_obs=10, guard_threshold=-99)
        cal = close.index
        # 可知时点后移 5 个交易日：IC(d) 在 d+5 才可知 → d 的位置 ≤ len-6
        ic_s = daily_ic_series(score, close, 5)
        usable = ic_s[ic_s.index <= cal[-6]]
        assert st["n_obs"] == len(usable.tail(15))
        assert st["ic_mean"] == pytest.approx(usable.tail(15).mean())
        # 最后 5 个交易日的 IC（前瞻窗口未走完）根本不存在于序列中
        assert not (set(cal[-5:]) & set(ic_s.index))
        # 判定样本里最晚的 IC 来自 T-5（可知时点恰为 T），不含任何 T 之后的信息
        assert usable.index.max() == cal[-6]

    def test_guard_forces_neutral(self):
        score, close = _make(corr=1.0, drift=-0.01)  # 每日 -1% 的暴跌市
        st = regime_state(score, close, ic_window=15, threshold=0.02,
                          min_obs=10, guard_days=10, guard_threshold=-0.03)
        assert st["guard_hit"] is True
        assert st["direction"] == "neutral"

    def test_vol_gate_forces_neutral(self):
        score, close = _make(corr=1.0)
        # 人为放大波动：隔日 ±20%
        close2 = close.copy()
        close2.iloc[::2, :] = close2.iloc[::2, :].values * 1.2
        close2.iloc[1::2, :] = close2.iloc[1::2, :].values * 0.8
        st = regime_state(score, close2, ic_window=15, threshold=0.02,
                          min_obs=10, guard_threshold=-99,
                          vol_window=20, vol_threshold=0.05)
        assert st["vol_gate"] is True
        assert st["direction"] == "neutral"

    def test_insufficient_samples_neutral(self):
        score, close = _make()
        # 只给 5 天历史：不足 min_obs=10
        st = regime_state(score.iloc[-5:], close.iloc[-5:],
                          ic_window=15, threshold=0.02, min_obs=10,
                          guard_threshold=-99)
        assert st["direction"] == "neutral"
        assert st["n_obs"] < 10

    def test_robustness_gate_blocks_a_weak_direction(self):
        """均值刚过阈值、但区块 bootstrap 跨 0 时不得给方向性信号。"""
        score, close = _make(corr=1.0)
        # 15 个已实现 IC 的均值约 +0.023，方向本会是 top，但正负交替并不稳健。
        weak = pd.Series([0.20, -0.18] * 7 + [0.20], index=close.index[-15:])
        st = regime_state(score, close, ic_window=15, threshold=0.02,
                          min_obs=10, guard_threshold=-99, ic_known=weak,
                          require_robust_signal=True, robust_ic_window=15,
                          robust_bootstrap_samples=200)
        assert st["robust_gate"] is True
        assert st["direction"] == "neutral"

    def test_mono_gate_downgrades_top_on_inverted_spread(self):
        """方向仍为 top、但最近 3 个可知日分层价差持续为负（高分组跑输低分组）时，
        单调性破坏预警应把方向提前降级为观望——不等 IC 均值翻负。"""
        score, close = _make(corr=1.0)
        sp = pd.Series([-0.01] * 10, index=close.index[-10:])
        st = regime_state(score, close, ic_window=15, threshold=0.02,
                          min_obs=10, guard_threshold=-99,
                          mono_days=3, spread_known=sp)
        assert st["direction"] == "neutral"
        assert st["mono_gate"] is True
        assert st["spread_last"] == pytest.approx(-0.01)

    def test_mono_gate_not_triggered_when_spread_mixed(self):
        """价差有正有负（未连续倒置）时不触发预警，方向维持 top。"""
        score, close = _make(corr=1.0)
        sp = pd.Series([-0.01, 0.02, -0.01, 0.02, -0.01],
                       index=close.index[-5:])
        st = regime_state(score, close, ic_window=15, threshold=0.02,
                          min_obs=10, guard_threshold=-99,
                          mono_days=3, spread_known=sp)
        assert st["mono_gate"] is False
        assert st["direction"] == "top"

    def test_mono_gate_confirms_bottom_not_blocks(self):
        """反转市里价差为负是确认信号（与 bottom 方向同向），不触发预警。"""
        score, close = _make(corr=-1.0)
        sp = pd.Series([-0.01] * 10, index=close.index[-10:])
        st = regime_state(score, close, ic_window=15, threshold=0.02,
                          min_obs=10, guard_threshold=-99,
                          mono_days=3, spread_known=sp)
        assert st["mono_gate"] is False
        assert st["direction"] == "bottom"

    def test_mono_gate_disabled_by_default(self):
        """不传 mono_days 时无预警逻辑，行为与旧版一致。"""
        score, close = _make(corr=1.0)
        st = regime_state(score, close, ic_window=15, threshold=0.02,
                          min_obs=10, guard_threshold=-99)
        assert st["mono_gate"] is False
        assert st["direction"] == "top"
