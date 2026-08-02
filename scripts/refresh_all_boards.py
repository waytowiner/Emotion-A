"""全量刷新东财板块池：板块列表 -> 缺失/截断成分 -> 缺失K线。

背景：clist 接口 pz 上限 100，此前板块列表只抓到第一页（按涨跌幅排序的前100，
严重偏向当时强势的科技类），白酒/证券/医药等大板块缺失。修复后重抓。
缓存增量，可反复运行直到全部成功。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from emotion.data import boards as boards_mod
from emotion.data import cache as c
from emotion.data import quotes as quotes_mod

CACHE = ROOT / "data" / "cache"
START = "2023-01-01"
END = pd.Timestamp.today().strftime("%Y-%m-%d")


def main() -> None:
    boards = boards_mod.get_board_list(CACHE, refresh=True)
    print(f"board list: {len(boards)}", flush=True)

    ok_c = fail_c = 0
    for _, row in boards.iterrows():
        df = c.load(CACHE, f"board_cons_{row['board_code']}")
        if df is not None and len(df) > 0 and len(df) % 100 != 0:
            continue
        try:
            cons = boards_mod.get_board_cons(CACHE, row["board_code"], refresh=True)
            ok_c += 1
            print(f"cons OK {row['board_name']}: {len(cons)}", flush=True)
        except Exception as exc:
            fail_c += 1
            print(f"cons FAIL {row['board_name']}: {str(exc)[:60]}", flush=True)
        time.sleep(0.8)

    frames = []
    for _, row in boards.iterrows():
        df = c.load(CACHE, f"board_cons_{row['board_code']}")
        if df is not None and len(df) > 0 and len(df) % 100 != 0:
            frames.append(df.assign(board_code=row["board_code"],
                                    board_name=row["board_name"]))
    if frames:
        c.save(CACHE, "all_cons", pd.concat(frames, ignore_index=True))
    print(f"cons complete boards: {len(frames)}/{len(boards)}", flush=True)

    ok_k = fail_k = 0
    for _, row in boards.iterrows():
        df = c.load(CACHE, f"board_kline_{row['board_code']}")
        if df is not None and not df.empty:
            continue
        try:
            k = quotes_mod.get_board_kline(CACHE, row["board_code"], START, END)
            ok_k += 1
            print(f"kline OK {row['board_name']}: {len(k)}", flush=True)
        except Exception as exc:
            fail_k += 1
            print(f"kline FAIL {row['board_name']}: {str(exc)[:60]}", flush=True)
        time.sleep(0.5)

    print(f"DONE cons ok={ok_c} fail={fail_c}; kline ok={ok_k} fail={fail_k}", flush=True)


if __name__ == "__main__":
    main()
