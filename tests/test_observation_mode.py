"""最终退出后，日报/接口不得把旧模型状态重新解释为可交易优势。"""
import pandas as pd
import pytest

from emotion import report
from emotion.project_status import OBSERVATION_NOTICE, project_status


@pytest.mark.parametrize("direction,ic", [("top", .1), ("bottom", -.1), ("neutral", 0.)])
def test_reports_remain_observational_for_every_legacy_direction(direction, ic):
    dates = pd.bdate_range("2026-10-08", periods=2)
    scores = pd.DataFrame({"A": [1., 2.], "B": [-1., -2.]}, index=dates)
    original = scores.copy()
    health = {"n": 60, "ic_mean": ic, "ic_win_rate": .6, "spread_mean": -.01}
    regime = {"direction": direction, "ic_mean": ic, "ic_win_rate": .6, "n_obs": 15,
              "rebalance_direction": direction, "rebalance_date": "2026-10-09"}
    text = report.build_report(dates[-1], scores, scores * 0 + .5, {}, {"A": "甲", "B": "乙"},
                               health=health, regime=regime)
    assert OBSERVATION_NOTICE in text
    assert "高分观察榜" in text and "低分观察榜" in text
    for obsolete in ("买点", "抄底", "榜单反向使用", "显著有效", "顺势使用", "降低仓位"):
        assert obsolete not in text
    pd.testing.assert_frame_equal(scores, original)
    assert regime["direction"] == direction


def test_observation_status_overrides_old_dashboard_cache(monkeypatch):
    import server
    cached = {"latest": {"regime": {"direction": "top"},
                         "project_status": {"mode": "old"}}, "meta": {"date": "2026-10-09"}}
    monkeypatch.setattr(server, "require_data", lambda: cached)
    result = server.api_overview()
    assert result["latest"]["project_status"] == project_status()
    assert not result["latest"]["project_status"]["trading_eligible"]
    assert result["latest"]["regime"]["direction"] == "top"
    assert cached["latest"]["project_status"] == {"mode": "old"}


def test_legacy_log_keeps_original_records_and_adds_dated_status_even_on_same_day(tmp_path):
    path = tmp_path / "prospective_log.md"
    original = "# 原预注册\n\n| 2026-10-09 | 动量市 | 原始记录 |\n"
    path.write_text(original, encoding="utf-8")
    date = pd.Timestamp("2026-10-09")
    assert report.append_prospective_log(tmp_path, date, {}, {}) is None
    assert report.append_prospective_log(tmp_path, date, {}, {}) is None
    text = path.read_text(encoding="utf-8")
    assert original in text
    assert text.count(OBSERVATION_NOTICE) == 1
    assert text.count("| 2026-10-09 |") == 1
