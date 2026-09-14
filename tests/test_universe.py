"""板块标的池治理测试。"""

import pandas as pd

from emotion.universe import constituent_counts, select_board_universe


def _sample():
    boards = pd.DataFrame({
        "board_code": ["A", "B2", "B3", "T"],
        "board_name": ["大板块", "重复板块Ⅱ", "重复板块Ⅲ", "极小板块"],
    })
    rows = []
    for code, stocks in {
        "A": ["000001", "000002", "000003"],
        "B2": ["000004", "000005"],
        "B3": ["000004", "000005"],
        "T": ["000006"],
    }.items():
        rows.extend({"board_code": code, "code": stock} for stock in stocks)
    return pd.DataFrame(rows), boards


def test_counts_deduplicates_member_rows():
    board_map, _ = _sample()
    duplicated = pd.concat([board_map, board_map.iloc[[0]]], ignore_index=True)
    assert constituent_counts(duplicated).to_dict()["A"] == 3


def test_select_drops_small_and_exact_duplicate_keeps_higher_tier():
    board_map, boards = _sample()
    selected_map, selected_boards, diag = select_board_universe(
        board_map, boards, min_constituents=2, duplicate_jaccard=1.0)
    assert selected_boards["board_code"].tolist() == ["A", "B2"]
    assert set(selected_map["board_code"]) == {"A", "B2"}
    assert diag["raw_board_count"] == 4
    assert diag["eligible_board_count"] == 2
    assert diag["dropped_small"][0]["code"] == "T"
    assert diag["dropped_duplicate"][0]["code"] == "B3"
    assert diag["dropped_duplicate"][0]["kept_code"] == "B2"


def test_disabled_preserves_raw_universe():
    board_map, boards = _sample()
    selected_map, selected_boards, diag = select_board_universe(
        board_map, boards, min_constituents=99, enabled=False)
    assert len(selected_map) == len(board_map)
    assert len(selected_boards) == len(boards)
    assert diag["eligible_board_count"] == 4
