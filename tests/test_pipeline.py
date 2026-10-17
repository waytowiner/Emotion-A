"""配置口径防呆测试。"""

import pytest

from emotion.pipeline import validate_config
from emotion import pipeline, score
import pandas as pd
import numpy as np


def _cfg():
    return {
        "weights": {"breadth": 1.0},
        "signal": {"hot_quantile": 0.9, "cold_quantile": 0.1},
        "universe": {"min_constituents": 20, "duplicate_jaccard": 1.0},
    }


def test_unknown_weight_fails_instead_of_being_silently_ignored():
    cfg = _cfg()
    cfg["weights"]["fund_flow"] = 1.0
    with pytest.raises(ValueError, match="未实现的因子"):
        validate_config(cfg)


def test_invalid_universe_threshold_fails_fast():
    cfg = _cfg()
    cfg["universe"]["duplicate_jaccard"] = 1.1
    with pytest.raises(ValueError, match="duplicate_jaccard"):
        validate_config(cfg)


def test_calendar_removes_foreign_sessions_before_smoothing():
    dates = pd.to_datetime(["2026-09-24", "2026-09-28", "2026-09-29",
                            "2026-09-30", "2026-10-08", "2026-10-09"])
    stocks = pd.DataFrame({"date": dates, "trade_status": "1"})
    foreign = pd.to_datetime(["2026-09-25", "2026-10-01", "2026-10-02",
                              "2026-10-05", "2026-10-06", "2026-10-07"])
    kline = pd.DataFrame({"date": dates.union(foreign), "board_code": "A", "close": 100.0})
    clean, audit = pipeline.align_board_calendar(kline, pipeline.stock_trading_calendar(stocks))
    assert clean["date"].tolist() == dates.tolist()
    assert audit["removed_board_rows"] == 6
    values = pd.DataFrame({"A": np.arange(6.0)}, index=dates)
    # 原来港股假期行进入 rolling(5)，使节后首日不足两个有效样本。
    polluted = score.smooth_score(values.reindex(dates.union(foreign)), 5)
    assert pd.isna(polluted.loc["2026-10-08", "A"])
    assert score.smooth_score(values.reindex(clean["date"]), 5).loc["2026-10-08", "A"] == 2.0


def test_prepare_applies_one_calendar_and_clips_both_sources(monkeypatch, tmp_path):
    from emotion.data import boards_etf
    dates = pd.to_datetime(["2026-09-23", "2026-09-24", "2026-09-28",
                            "2026-09-29", "2026-09-30", "2026-10-08"])
    stocks = pd.DataFrame({"date": dates, "code": "000001", "trade_status": "1",
                           "amount": 100.0, "pct_chg": 0.0})
    cons = pd.DataFrame({"code": ["000001"], "name": ["测试"],
                         "board_code": ["A"], "board_name": ["测试板块"]})
    cons.to_parquet(tmp_path / "all_cons_etf.parquet")
    monkeypatch.setattr(pipeline, "load_stock_chunks", lambda _: stocks.copy())
    monkeypatch.setattr(boards_etf, "get_board_list_etf", lambda *a, **k: cons[["board_code", "board_name"]])
    extra = pd.to_datetime(["2026-09-25", "2026-09-27", "2026-10-02"])
    kline = pd.DataFrame({"date": dates.union(extra), "close": 100.0})
    monkeypatch.setattr(boards_etf, "get_board_kline_etf", lambda *a, **k: kline.copy())
    cfg = {"paths": {"cache_dir": str(tmp_path)}, "history": {"start_date": "2026-09-24"},
           "board_source": "etf", "universe": {"enabled": False}}
    data = pipeline.prepare(cfg, offline=True, end_override="2026-09-30")
    expected = dates[1:5].tolist()
    assert data["stock_daily"]["date"].tolist() == expected
    assert data["board_kline"]["date"].tolist() == expected
    assert data["market_total"].index.tolist() == expected
    assert data["calendar_audit"]["removed_dates"] == ["2026-09-25", "2026-09-27"]
