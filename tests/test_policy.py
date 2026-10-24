"""政策（时政）支柱测试：词典、命中计数、三口径、双因状态、优雅降级。

合成数据，不依赖网络。红线：政策层不得影响情绪侧既有口径。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from emotion.factors import FACTOR_NAMES
from emotion.policy import (align_hits_to_calendar, build_policy_scores,
                            compute_policy_factors, detect_policy_signals,
                            dual_factor_states, load_keyword_map,
                            map_news_to_boards)
from emotion.score import zscore_cross_section

DATES = pd.date_range("2026-01-05", periods=30, freq="B")
BOARDS = ["B1", "B2", "B3", "B4"]


def _news(rows):
    return pd.DataFrame(rows, columns=["date", "title", "content"])


def _kw():
    return {
        "B1": ["芯片", "半导体"],
        "B2": ["新能源", "光伏"],
        "B3": ["医药"],
    }


class TestKeywordMap:
    def test_load_real_dict_codes_covered(self):
        kw = load_keyword_map()
        # 真实词典存在且非空；词条为 list[str]
        assert kw, "config/policy_keywords.yaml 应有非空词条"
        for code, terms in kw.items():
            assert isinstance(code, str) and code
            assert terms and all(isinstance(t, str) for t in terms)

    def test_missing_file_returns_empty(self, tmp_path):
        assert load_keyword_map(tmp_path / "nope.yaml") == {}


class TestMapNews:
    def test_hit_counts_by_news_item(self):
        # 同一条新闻命中 B1 两个关键词仍计 1；两条新闻各命中计 2
        news = _news([
            ("2026-01-05", "芯片半导体产业大会召开", "集成电路全线突破"),
            ("2026-01-05", "无关新闻一条", "今天天气不错"),
            ("2026-01-05", "又一芯片政策发布", "内容"),
        ])
        hit = map_news_to_boards(news, _kw(), board_names={b: b for b in BOARDS})
        assert hit.loc["2026-01-05", "B1"] == 2
        assert hit.loc["2026-01-05", "B2"] == 0
        # B4 不在词典 -> 不出现在列中
        assert "B4" not in hit.columns

    def test_board_names_filter(self):
        news = _news([("2026-01-05", "芯片新闻", "")])
        # 词典有 B1/B2/B3，但板块池只认 B2/B3 -> B1 被过滤
        hit = map_news_to_boards(news, _kw(), board_names={"B2": "B2", "B3": "B3"})
        assert list(hit.columns) == ["B2", "B3"]

    def test_empty_inputs(self):
        assert map_news_to_boards(pd.DataFrame(), _kw()).empty
        assert map_news_to_boards(_news([("2026-01-05", "x", "")]), {}).empty

    def test_no_hit_returns_zero_matrix(self):
        news = _news([("2026-01-05", "无关", "无关键词")])
        hit = map_news_to_boards(news, _kw(), board_names={b: b for b in BOARDS})
        assert not hit.empty
        assert (hit == 0).all().all()

    def test_content_or_title(self):
        # 关键词只在正文也命中
        news = _news([("2026-01-05", "标题干净", "正文中出现光伏新政")])
        hit = map_news_to_boards(news, _kw(), board_names={b: b for b in BOARDS})
        assert hit.loc["2026-01-05", "B2"] == 1


class TestAlignCalendar:
    def test_weekend_rolls_to_next_trading_day(self):
        # 1/10 周六命中并入 1/12 周一
        hit = pd.DataFrame(
            {"B1": [1.0, 2.0]},
            index=pd.DatetimeIndex(["2026-01-10", "2026-01-12"]))
        cal = pd.DatetimeIndex(["2026-01-09", "2026-01-12", "2026-01-13"])
        out = align_hits_to_calendar(hit, cal)
        assert list(out.index) == [pd.Timestamp("2026-01-09"),
                                   pd.Timestamp("2026-01-12")]
        assert out.loc["2026-01-09", "B1"] == 0.0  # 无命中的交易日补 0
        assert out.loc["2026-01-12", "B1"] == 3.0  # 周六 1 + 周一 2

    def test_no_outer_extrapolation(self):
        hit = pd.DataFrame({"B1": [1.0]}, index=pd.DatetimeIndex(["2026-01-05"]))
        cal = pd.DatetimeIndex(["2026-01-05", "2026-02-01"])
        out = align_hits_to_calendar(hit, cal)
        # 日历超出命中末日 [2026-01-05, 2026-01-05] 的部分不外插
        assert list(out.index) == [pd.Timestamp("2026-01-05")]


class TestPolicyFactors:
    def test_intensity_breadth_heat(self):
        idx = pd.date_range("2026-01-05", periods=12, freq="B")
        hit = pd.DataFrame(
            {"B1": [1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             "B2": [0] * 12},
            index=idx, dtype=float)
        f = compute_policy_factors(hit, heat_window=5, breadth_window=10)
        assert f["policy_intensity"].loc[idx[0], "B1"] == 1
        # 5 日滚动和：第3日 = 1+0+2
        assert f["policy_heat"].loc[idx[2], "B1"] == 3
        # 10 日广度：idx[9] 的窗口为 idx[0..9]，其中 idx0/idx2 两天命中 -> 0.2；
        # idx[2] 窗口 idx[0..2] 命中 2/3 -> 2/3
        assert f["policy_breadth"].loc[idx[9], "B1"] == pytest.approx(0.2)
        assert f["policy_breadth"].loc[idx[2], "B1"] == pytest.approx(2 / 3)
        assert (f["policy_breadth"]["B2"] == 0).all()

    def test_empty(self):
        f = compute_policy_factors(pd.DataFrame())
        assert set(f) == {"policy_intensity", "policy_breadth", "policy_heat"}
        assert all(v.empty for v in f.values())


class TestPolicyScores:
    def test_shapes_and_decoupling(self):
        idx = pd.date_range("2026-01-05", periods=60, freq="B")
        rng = np.random.default_rng(7)
        hit = pd.DataFrame(
            rng.poisson(0.3, (len(idx), len(BOARDS))).astype(float),
            index=idx, columns=BOARDS)
        raw, smoothed, pct = build_policy_scores(hit, smooth_window=5, lookback=40)
        assert raw.shape == smoothed.shape == (len(idx), len(BOARDS))
        # 政策层不得污染情绪因子口径
        assert "policy_heat" not in FACTOR_NAMES

    def test_all_zero_day_intensity_is_nan_but_rolling_factors_alive(self):
        # 全员 0 的日期 intensity 横截面 std=0 -> z 全 NaN；
        # 但 breadth/heat 为滚动量仍有横截面差异，评分继续可用。
        idx = pd.date_range("2026-01-05", periods=6, freq="B")
        hit = pd.DataFrame(
            {"B1": [1, 0, 3, 0, 0, 0], "B2": [2, 0, 1, 0, 0, 0],
             "B3": [0, 0, 2, 0, 0, 0], "B4": [1, 0, 1, 0, 0, 0]},
            index=idx, dtype=float)
        f = compute_policy_factors(hit)
        assert f["policy_intensity"].loc[idx[1]].tolist() == [0, 0, 0, 0]
        raw, _, _ = build_policy_scores(hit)
        # intensity NaN 被 groupby.mean 跳过，该日评分仍非全空
        assert raw.loc[idx[1]].notna().all()
        assert raw.loc[idx[0]].notna().all()


class TestDualFactor:
    def _pct(self, e_vals, p_vals):
        idx = pd.DatetimeIndex(["2026-01-05"])
        e = pd.DataFrame([e_vals], index=idx, columns=BOARDS, dtype=float)
        p = pd.DataFrame([p_vals], index=idx, columns=BOARDS, dtype=float)
        return e, p

    def test_state_classification(self):
        e, p = self._pct([0.9, 0.1, 0.9, 0.1], [0.85, 0.85, 0.1, 0.1])
        st = dual_factor_states(e, p)
        row = st.iloc[0]
        assert row["B1"] == "resonance"
        assert row["B2"] == "lurking"
        assert row["B3"] == "funds_first"
        assert row["B4"] == "dual_cold"

    def test_neutral_middle_band(self):
        e, p = self._pct([0.5, 0.85, 0.5, 0.95], [0.5, 0.5, 0.85, 0.95])
        st = dual_factor_states(e, p)
        row = st.iloc[0]
        assert row["B1"] == "neutral"
        assert row["B2"] == "neutral"
        assert row["B3"] == "neutral"
        assert row["B4"] == "resonance"  # 边界值 >= hot_q

    def test_nan_side_is_neutral(self):
        e, p = self._pct([np.nan, 0.9, 0.5, 0.5], [0.9, np.nan, 0.5, 0.5])
        st = dual_factor_states(e, p)
        assert (st.iloc[0] == "neutral").all()

    def test_empty_inputs(self):
        assert dual_factor_states(pd.DataFrame(), pd.DataFrame()).empty

    def test_detect_signals(self):
        e, p = self._pct([0.9, 0.1, 0.5, 0.9], [0.85, 0.85, 0.85, 0.1])
        sig = detect_policy_signals(e, p)
        assert sig["policy_resonance"] == ["B1"]
        assert sig["policy_lurking"] == ["B2"]
        # B3 情绪中间带不进任何榜；B4 政策冷不是潜伏
        assert "B3" not in sig["policy_resonance"] + sig["policy_lurking"]
        assert "B4" not in sig["policy_lurking"]

    def test_detect_signals_empty(self):
        sig = detect_policy_signals(pd.DataFrame(), pd.DataFrame())
        assert sig == {"policy_resonance": [], "policy_lurking": []}


class TestZscoreReuse:
    def test_policy_uses_same_zscore_convention(self):
        # 政策合成复用情绪侧 zscore_cross_section：均值 0、方差 1
        f = pd.DataFrame(
            {b: [float(i + BOARDS.index(b)) for i in range(10)] for b in BOARDS},
            index=pd.date_range("2026-01-05", periods=10, freq="B"))
        z = zscore_cross_section(f)
        assert z.mean(axis=1).abs().max() < 1e-10
        assert z.std(axis=1).sub(1).abs().max() < 1e-10
