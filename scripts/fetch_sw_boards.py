"""抓取申万行业板块（二级 131 个）：行业列表 -> 成分股 -> 指数日线。缓存增量。

用法：python scripts/fetch_sw_boards.py [--refresh]
  --refresh：强制重抓成分与指数（默认缓存有效即跳过）。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from emotion.data import boards_sw
from emotion.data import cache as cache_mod

CACHE = ROOT / "data" / "cache"
END = pd.Timestamp.today().strftime("%Y-%m-%d")


def main() -> None:
    refresh = "--refresh" in sys.argv
    boards = boards_sw.get_board_list_sw(CACHE, refresh=refresh)
    print(f"申万二级行业: {len(boards)}", flush=True)

    ok_c = fail_c = 0
    for _, row in boards.iterrows():
        try:
            cons = boards_sw.get_board_cons_sw(CACHE, row["board_code"],
                                               refresh=refresh)
            ok_c += 1
            print(f"cons {row['board_name']}: {len(cons)}", flush=True)
        except Exception as exc:  # noqa: BLE001
            fail_c += 1
            print(f"cons {row['board_name']} FAIL: {exc}", flush=True)
        time.sleep(0.2)

    frames = []
    for _, row in boards.iterrows():
        df = cache_mod.load(CACHE, f"board_cons_sw_{row['board_code']}")
        if df is not None:
            frames.append(df.assign(board_code=row["board_code"],
                                    board_name=row["board_name"]))
    if frames:
        cache_mod.save(CACHE, "all_cons_sw", pd.concat(frames, ignore_index=True))
        print(f"all_cons_sw saved: {sum(len(f) for f in frames)} rows", flush=True)

    ok_k = fail_k = 0
    for _, row in boards.iterrows():
        try:
            k = boards_sw.get_board_kline_sw(CACHE, row["board_code"],
                                             "2023-01-01", END, refresh=refresh)
            ok_k += 1
            print(f"kline {row['board_name']}: {len(k)} (至 {str(k['date'].max())[:10]})",
                  flush=True)
        except Exception as exc:  # noqa: BLE001
            fail_k += 1
            print(f"kline {row['board_name']} FAIL: {exc}", flush=True)
        time.sleep(0.2)

    print(f"DONE cons ok={ok_c} fail={fail_c}; kline ok={ok_k} fail={fail_k}",
          flush=True)


if __name__ == "__main__":
    main()
