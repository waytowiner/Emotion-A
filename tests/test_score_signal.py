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
        assert sig == {"heating": [], "cooling": [], "hot": [], "cold": [], "switching": []}
