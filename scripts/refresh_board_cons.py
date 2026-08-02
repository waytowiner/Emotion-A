"""刷新东财板块成分：重抓缺失或被 100 条上限截断的板块（pz 修复后）。"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from emotion.data import boards as boards_mod
from emotion.data import cache as c

CACHE = ROOT / "data" / "cache"


def main() -> None:
    boards = boards_mod.get_board_list(CACHE)
    todo = []
    for _, row in boards.iterrows():
        df = c.load(CACHE, f"board_cons_{row['board_code']}")
        if df is None or len(df) == 0 or len(df) % 100 == 0:
            todo.append(row)
    print(f"need refresh: {len(todo)} / {len(boards)}", flush=True)

    ok = fail = 0
    for row in todo:
        try:
            cons = boards_mod.get_board_cons(CACHE, row["board_code"], refresh=True)
            ok += 1
            print(f"OK {row['board_name']}: {len(cons)}", flush=True)
        except Exception as exc:
            fail += 1
            print(f"FAIL {row['board_name']}: {str(exc)[:60]}", flush=True)
        time.sleep(1.0)

    frames = []
    for _, row in boards.iterrows():
        df = c.load(CACHE, f"board_cons_{row['board_code']}")
        if df is not None and len(df) % 100 != 0:
            frames.append(df.assign(board_code=row["board_code"],
                                    board_name=row["board_name"]))
    if frames:
        allc = pd.concat(frames, ignore_index=True)
        c.save(CACHE, "all_cons", allc)
    print(f"DONE ok={ok} fail={fail}, complete boards={len(frames)}", flush=True)


if __name__ == "__main__":
    main()
