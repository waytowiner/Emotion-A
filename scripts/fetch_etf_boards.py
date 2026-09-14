"""抓取可交易 ETF 板块池：指数列表 -> 成分股 -> 指数日线。缓存增量，可断点续跑。

用法：python scripts/fetch_etf_boards.py [--refresh]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from emotion.data import boards_etf
from emotion.data import cache as cache_mod

CACHE = ROOT / "data" / "cache"
END = pd.Timestamp.today().strftime("%Y-%m-%d")


def _last_closed_trade_date() -> str:
    """最近一个已收盘的交易日（近似：跳过周末；15:00 前视为昨日）。

    用于 K 线"已最新"判断：盘中运行时缓存到上一交易日即算最新，
    不再全池重发请求。此前用 END=今天判断，导致盘中任何一次重跑都会
    把 249 个板块全部重新请求一遍，耗尽中证接口限流配额后，
    剩余真正滞后的板块反而抓不到（2026-09-09 实测 81 个卡在旧日期）。
    法定节假日未剔除：多抓一次空数据无害，漏抓由下一交易日补上。
    """
    t = pd.Timestamp.now()
    if t.hour < 15:
        t -= pd.Timedelta(days=1)
    d = t.normalize()
    while d.weekday() >= 5:
        d -= pd.Timedelta(days=1)
    return d.strftime("%Y-%m-%d")


def main() -> None:
    refresh = "--refresh" in sys.argv
    boards = boards_etf.get_board_list_etf(CACHE, refresh=refresh)
    print(f"可交易板块（中证行业/主题+有跟踪产品）: {len(boards)}", flush=True)

    ok_c = fail_c = 0
    consec = 0
    for _, row in boards.iterrows():
        if consec >= 8:
            print(f"连续 {consec} 次失败，疑似被限流，中止（已缓存的可断点续跑）",
                  flush=True)
            break
        cached = cache_mod.load(CACHE, f"board_cons_etf_{row['board_code']}")
        if cached is not None and not cached.empty and not refresh:
            ok_c += 1
            continue  # 缓存有效：无请求、无 sleep
        try:
            cons = boards_etf.get_board_cons_etf(CACHE, row["board_code"],
                                                 refresh=refresh)
            ok_c += 1
            consec = 0
            print(f"cons {row['board_name']}: {len(cons)}", flush=True)
        except Exception as exc:  # noqa: BLE001
            fail_c += 1
            consec += 1
            print(f"cons {row['board_name']} FAIL: {exc}", flush=True)
        time.sleep(2.0)

    frames = []
    for _, row in boards.iterrows():
        df = cache_mod.load(CACHE, f"board_cons_etf_{row['board_code']}")
        if df is not None:
            frames.append(df.assign(board_code=row["board_code"],
                                    board_name=row["board_name"]))
    if frames:
        cache_mod.save(CACHE, "all_cons_etf", pd.concat(frames, ignore_index=True))
        print(f"all_cons_etf saved: {sum(len(f) for f in frames)} rows", flush=True)

    ok_k = fail_k = 0
    consec = 0
    done_date = _last_closed_trade_date()
    if not refresh and len(boards) > 0 and done_date == END:
        # 收盘后探针：先抓 1 个板块，若中证当日数据尚未发布，
        # 把"已最新"标准降级到实际可得的最后交易日，避免全池空抓。
        try:
            probe = boards_etf.get_board_kline_etf(
                CACHE, boards.iloc[0]["board_code"], "2023-01-01", END)
            probe_last = str(probe["date"].max())[:10]
            if probe_last < END:
                done_date = probe_last
                print(f"[probe] 中证当日数据未发布，最新标准降至 {probe_last}",
                      flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[probe] 探测失败，按 {done_date} 判断：{exc}", flush=True)
    for _, row in boards.iterrows():
        if consec >= 8:
            print(f"连续 {consec} 次失败，疑似被限流，中止（已缓存的可断点续跑）",
                  flush=True)
            break
        if not refresh:
            cached = cache_mod.load(CACHE, f"board_kline_etf_{row['board_code']}")
            if cached is not None and not cached.empty \
                    and str(cached["date"].max())[:10] >= done_date:
                ok_k += 1
                continue  # 已覆盖最近收盘交易日：无请求、无 sleep
        try:
            k = boards_etf.get_board_kline_etf(CACHE, row["board_code"],
                                               "2023-01-01", END, refresh=refresh)
            ok_k += 1
            consec = 0
            print(f"kline {row['board_name']}: {len(k)} (至 {str(k['date'].max())[:10]})",
                  flush=True)
        except Exception as exc:  # noqa: BLE001
            fail_k += 1
            consec += 1
            print(f"kline {row['board_name']} FAIL: {exc}", flush=True)
        time.sleep(2.0)

    print(f"DONE cons ok={ok_c} fail={fail_c}; kline ok={ok_k} fail={fail_k}",
          flush=True)


if __name__ == "__main__":
    main()
