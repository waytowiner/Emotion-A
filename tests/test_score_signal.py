"""合成层与信号层测试（合成数据）。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from emotion.score import build_scores, combine_factors, percentile_rank, zscore_cross_section
from emotion.signal import detect_signals

DATES = pd.date_range("2026-01-05", periods=30, freq="B")
BOARDS = ["A", "B", "C", "D"]


def _panel(values_fn):
    return pd.DataFrame(
        {b: [values_fn(b, i) for i in range(len(DATES))] for b in BOARDS},
        index=DATES,
    )


class TestScore:
    def test_zscore_row_mean_zero(self):
        f = _panel(lambda b, i: i + BOARDS.index(b))
        z = zscore_cross_section(f)
        assert z.mean(axis=1).abs().max() < 1e-10
        assert z.std(axis=1).sub(1).abs().max() < 1e-10

    def test_combine_weights_normalize(self):
        # 板块间有差异（避免横截面 std=0），且两因子互为相反数
        f1 = _panel(lambda b, i: i + BOARDS.index(b))
        f2 = _panel(lambda b, i: -(i + BOARDS.index(b)))
        combined = combine_factors({"x": f1, "y": f2}, {"x": 1.0, "y": 1.0})
        assert combined.abs().max().max() < 1e-10  # 等权对冲

    def test_missing_factor_dropped(self):
        f1 = _panel(lambda b, i: i + BOARDS.index(b))
        combined = combine_factors({"x": f1, "ghost": None}, {"x": 1.0, "ghost": 1.0})
        assert combined.notna().any().any()

    def test_combine_z_clip(self):
        # 21 板块构造单因子极端值：[100, 0, 0, ...] 的横截面 z≈4.37 > 3
        boards = [f"B{i}" for i in range(21)]
        f = pd.DataFrame(
            {b: [(100.0 if b == "B0" else 0.0)] * len(DATES) for b in boards},
            index=DATES)
        unclipped = combine_factors({"x": f}, {"x": 1.0}, z_clip=None)
        assert unclipped["B0"].iloc[-1] > 3.0
        # 缺省 z_clip=3.0：极端单因子被截断，不再绑架综合分
        clipped = combine_factors({"x": f}, {"x": 1.0})
        assert clipped["B0"].iloc[-1] == pytest.approx(3.0)

    def test_build_scores_shapes(self):
        f = _panel(lambda b, i: i + BOARDS.index(b) * 2)
        raw, smoothed, pct = build_scores({"x": f}, {"x": 1.0}, smooth_window=5)
        assert raw.shape == smoothed.shape == (len(DATES), len(BOARDS))
        # 平滑后头部窗口期有 NaN
        assert smoothed.iloc[0].isna().all()

    def test_percentile_monotone(self):
        s = pd.DataFrame({"A": np.arange(100.0)}, index=pd.date_range("2025-01-01", periods=100))
        pct = percentile_rank(s, lookback=50)
        assert pct["A"].iloc[-1] == pytest.approx(1.0)


class TestSignal:
    def _make(self):
        # A 连续上升至顶部，B 自顶部连续下降，E 自底部缓慢上升，C/D 震荡
        boards = ["A", "B", "C", "D", "E"]
        smoothed = pd.DataFrame(
            {b: [{
                "A": float(i) + 10, "B": float(30 - i),
                "C": float(i % 2), "D": float((i + 1) % 2),
                "E": float(i) * 0.1 - 5}[b] for i in range(len(DATES))]
            for b in boards}, index=DATES)
        # 显式构造时序分位面板（真实管线里由 percentile_rank 生成）：
        # A 高位、B 中位、E 低位
        pct = smoothed.copy()
        pct.loc[:, :] = 0.5
        pct.iloc[-1, :] = [0.95, 0.60, 0.50, 0.50, 0.05]
        return smoothed, pct

    def test_heating_cooling(self):
        smoothed, pct = self._make()
        sig = detect_signals(smoothed, pct, consecutive_days=3)
        assert sig["heating"] == ["A", "E"]  # 按最新情绪分降序
        assert sig["cooling"] == ["B"]

    def test_hot_cold(self):
        smoothed, pct = self._make()
        sig = detect_signals(smoothed, pct, consecutive_days=3,
                             hot_quantile=0.9, cold_quantile=0.1)
        assert sig["hot"] == ["A"]
        assert sig["cold"] == ["E"]

    def test_switching_alert(self):
        smoothed, pct = self._make()
        sig = detect_signals(smoothed, pct, consecutive_days=3, recent_top_k=2)
        # B 曾是前 2 名（前期 B 最高），现在正在降温；E 从低位升温
        assert any("B" in s for s in sig["switching"])
        assert any("E" in s for s in sig["switching"])

    def test_empty_input(self):
        sig = detect_signals(pd.DataFrame(), pd.DataFrame())
        assert sig == {"heating": [], "cooling": [], "hot": [], "cold": [],
                       "switching": [], "bottom_reversal": [],
                       "extreme_hot": [], "extreme_cold": [],
                       "fund_emotion_divergence": []}


class TestBreakthrough:
    """P1 突变穿透规则测试（合成数据，验证不进合成情绪分、不经平滑）。

    注意：默认 ndays=3（fund_positive_days 已从 2 收紧到 3），
    默认 fund_flow_z_threshold=1.5（已从 2.0 放宽）。
    """

    DATES = pd.date_range("2026-01-05", periods=12, freq="B")

    def _panel(self, boards, fill=0.0):
        return pd.DataFrame(fill, index=self.DATES, columns=boards, dtype=float)

    def test_bottom_reversal(self):
        # 8 板块：E 末日资金巨量流入（横截面 z≈2.47>2）且 E 处冰点
        boards = ["A", "B", "C", "D", "E", "F", "G", "H"]
        fund_flow = self._panel(boards)
        fund_flow.iloc[-1, boards.index("E")] = 1e9
        smoothed = self._panel(boards, fill=0.5)
        smoothed.iloc[-1, boards.index("E")] = -2.0
        pct = self._panel(boards, fill=0.5)
        pct.iloc[-1, boards.index("E")] = 0.05
        sig = detect_signals(smoothed, pct, fund_flow=fund_flow,
                             cold_quantile=0.1, hot_quantile=0.9)
        assert sig["bottom_reversal"] == ["E"]

    def test_extreme_hot(self):
        # A 亢奋 + 单因子横截面 z≈2.47>2 → 热端
        boards = ["A", "B", "C", "D", "E", "F", "G", "H"]
        factor = self._panel(boards)
        factor.iloc[-1, 0] = 10.0
        smoothed = self._panel(boards, fill=0.0)
        pct = self._panel(boards, fill=0.5)
        pct.iloc[-1, 0] = 0.95
        sig = detect_signals(smoothed, pct, factors={"f": factor},
                             cold_quantile=0.1, hot_quantile=0.9)
        assert "A" in sig["extreme_hot"]
        assert sig["extreme_cold"] == []  # 只有亢奋没有冰点，冷端应为空

    def test_extreme_cold(self):
        # E 冰点 + 单因子横截面 z≈2.47>2 → 冷端
        boards = ["A", "B", "C", "D", "E", "F", "G", "H"]
        factor = self._panel(boards)
        factor.iloc[-1, boards.index("E")] = 10.0
        smoothed = self._panel(boards, fill=0.0)
        pct = self._panel(boards, fill=0.5)
        pct.iloc[-1, boards.index("E")] = 0.05
        sig = detect_signals(smoothed, pct, factors={"f": factor},
                             cold_quantile=0.1, hot_quantile=0.9)
        assert "E" in sig["extreme_cold"]
        assert sig["extreme_hot"] == []  # 只有冰点没有亢奋，热端应为空

    def test_fund_emotion_divergence(self):
        # B 资金连续 3 日为正 + 情绪分为负（ndays 默认 3）
        boards = ["A", "B", "C", "D", "E", "F", "G", "H"]
        fund_flow = self._panel(boards)
        fund_flow.iloc[-3, 1] = 1e8
        fund_flow.iloc[-2, 1] = 1e8
        fund_flow.iloc[-1, 1] = 1e8
        smoothed = self._panel(boards, fill=0.0)
        smoothed.iloc[-1, 1] = -1.5
        pct = self._panel(boards, fill=0.5)
        sig = detect_signals(smoothed, pct, fund_flow=fund_flow,
                             cold_quantile=0.1, hot_quantile=0.9)
        assert "B" in sig["fund_emotion_divergence"]

    def test_no_inputs_no_breakthrough(self):
        boards = ["A", "B", "C", "D", "E"]
        smoothed = self._panel(boards, fill=0.0)
        pct = self._panel(boards, fill=0.5)
        sig = detect_signals(smoothed, pct)
        assert sig["bottom_reversal"] == []
        assert sig["extreme_hot"] == []
        assert sig["extreme_cold"] == []
        assert sig["fund_emotion_divergence"] == []

    def test_disabled_flag(self):
        boards = ["A", "B", "C", "D", "E", "F", "G", "H"]
        fund_flow = self._panel(boards)
        fund_flow.iloc[-1, boards.index("E")] = 1e9
        smoothed = self._panel(boards, fill=0.5)
        smoothed.iloc[-1, boards.index("E")] = -2.0
        pct = self._panel(boards, fill=0.5)
        pct.iloc[-1, boards.index("E")] = 0.05
        sig = detect_signals(smoothed, pct, fund_flow=fund_flow,
                             breakthrough_cfg={"enabled": False})
        assert sig["bottom_reversal"] == []
