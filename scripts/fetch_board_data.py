"""抓取东财板块数据：板块列表 -> 成分股 -> 板块指数日线。缓存增量，可断点续跑。"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from emotion.data import boards as boards_mod
from emotion.data import quotes as quotes_mod

CACHE = ROOT / "data" / "cache"
START = "2023-01-01"
END = pd.Timestamp.today().strftime("%Y-%m-%d")


def main() -> None:
    boards = boards_mod.get_board_list(CACHE, refresh=False)
    print(f"boards: {len(boards)}", flush=True)

    ok_c = fail_c = 0
    for _, row in boards.iterrows():
        try:
            cons = boards_mod.get_board_cons(CACHE, row["board_code"], refresh=False)
            ok_c += 1
            print(f"cons {row['board_name']}: {len(cons)}", flush=True)
        except Exception as exc:
            fail_c += 1
            print(f"cons {row['board_name']} FAIL: {exc}", flush=True)
        time.sleep(0.8)

    # 合并成分总表
    frames = []
    for _, row in boards.iterrows():
        from emotion.data import cache as c
        df = c.load(CACHE, f"board_cons_{row['board_code']}")
        if df is not None:
            frames.append(df.assign(board_code=row["board_code"],
                                    board_name=row["board_name"]))
    if frames:
        c.save(CACHE, "all_cons", pd.concat(frames, ignore_index=True))
        print(f"all_cons saved: {sum(len(f) for f in frames)} rows", flush=True)

    ok_k = fail_k = 0
    for _, row in boards.iterrows():
        try:
            k = quotes_mod.get_board_kline(CACHE, row["board_code"], START, END)
            ok_k += 1
            print(f"kline {row['board_name']}: {len(k)}", flush=True)
        except Exception as exc:
            fail_k += 1
            print(f"kline {row['board_name']} FAIL: {exc}", flush=True)
        time.sleep(0.5)

    print(f"DONE cons ok={ok_c} fail={fail_c}; kline ok={ok_k} fail={fail_k}", flush=True)


if __name__ == "__main__":
    main()
