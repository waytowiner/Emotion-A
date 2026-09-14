"""配置口径防呆测试。"""

import pytest

from emotion.pipeline import validate_config


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
