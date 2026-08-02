"""抓取同花顺板块数据：列表 -> 成分股 -> 指数日线。缓存增量，可断点续跑。"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from emotion.data import boards_ths
from emotion.data import cache as c

CACHE = ROOT / "data" / "cache"
START = "2023-01-01"
END = pd.Timestamp.today().strftime("%Y-%m-%d")


def main() -> None:
    boards = boards_ths.get_board_list_ths(CACHE)
    print(f"ths boards: {len(boards)}", flush=True)

    ok_c = fail_c = 0
    for _, row in boards.iterrows():
        if c.load(CACHE, f"board_cons_ths_{row['board_code']}") is not None:
            continue
        try:
            cons = boards_ths.get_board_cons_ths(CACHE, row["board_code"])
            ok_c += 1
            print(f"cons OK {row['board_name']}: {len(cons)}", flush=True)
        except Exception as exc:
            fail_c += 1
            print(f"cons FAIL {row['board_name']}: {str(exc)[:60]}", flush=True)
        time.sleep(1.0)

    ok_k = fail_k = 0
    for _, row in boards.iterrows():
        if c.load(CACHE, f"board_kline_ths_{row['board_name']}") is not None:
            continue
        try:
            k = boards_ths.get_board_kline_ths(CACHE, row["board_name"], START, END)
            ok_k += 1
            print(f"kline OK {row['board_name']}: {len(k)}", flush=True)
        except Exception as exc:
            fail_k += 1
            print(f"kline FAIL {row['board_name']}: {str(exc)[:60]}", flush=True)
        time.sleep(0.8)

    print(f"DONE cons ok={ok_c} fail={fail_c}; kline ok={ok_k} fail={fail_k}", flush=True)


if __name__ == "__main__":
    main()
