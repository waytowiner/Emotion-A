"""全市场资金流快照（东财 push2 clist）-> 合并进 fund_flow_{code}.parquet

与 scripts/fetch_fund_flow.py（逐股历史，单次上限 120 条、易限流）互补：
本接口一次分页拉全 A 股当日主力净流入，约 60 个请求覆盖全部成分股；
无历史——价值在"从今天起逐日累积"，盘后运行（15:00 后）抓当日终值。

字段口径：f62=主力净流入（超大单+大单，元）、f66 超大单、f72 大单、
f78 中单、f84 小单。只保留 all_cons_etf 股票池内的沪深 A 股。

用法：python scripts/fetch_fund_flow_snapshot.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from emotion.data import cache as cache_mod
from emotion.data.quotes import _UT
from emotion.transport import get_json

CACHE = ROOT / "data" / "cache"
LIST_URL = "https://push2.eastmoney.com/api/qt/clist/get"
PAGE_SIZE = 100


def _last_closed_trade_date() -> str:
    """最近一个已收盘的交易日（跳过周末；15:00 前视为昨日）。同 fetch_fund_flow。"""
    t = pd.Timestamp.now()
    if t.hour < 15:
        t -= pd.Timedelta(days=1)
    d = t.normalize()
    while d.weekday() >= 5:
        d -= pd.Timedelta(days=1)
    return d.strftime("%Y-%m-%d")


def fetch_snapshot() -> pd.DataFrame:
    """全市场当日资金流快照（分页拉全）。返回 [code, main_net, ...] + snap_date。"""
    frames = []
    total = None
    page = 0
    while True:
        page += 1
        d = get_json(
            LIST_URL,
            params={
                "fid": "f62", "po": 1, "pz": PAGE_SIZE, "pn": page,
                "np": 1, "fltt": 2, "invt": 2,
                "fs": "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23",
                "fields": "f12,f62,f66,f72,f78,f84,f124",
                "ut": _UT,
            },
            timeout=10, max_retries=6)
        data = d.get("data") or {}
        total = data.get("total")
        rows = data.get("diff") or []
        if not rows:
            break
        frames.append(pd.DataFrame(rows))
        if total is not None and page * PAGE_SIZE >= total:
            break
        time.sleep(0.5)
    if not frames:
        raise RuntimeError("快照为空（接口不可用或非交易日）")
    out = pd.concat(frames, ignore_index=True)
    out = out.rename(columns={"f12": "code", "f62": "main_net",
                              "f66": "super_net", "f72": "big_net",
                              "f78": "mid_net", "f84": "small_net",
                              "f124": "ts"})
    return out


def main() -> None:
    cons_path = CACHE / "all_cons_etf.parquet"
    if not cons_path.exists():
        raise SystemExit("缺少 all_cons_etf.parquet，请先运行 scripts/fetch_etf_boards.py")
    pool = set(pd.read_parquet(cons_path)["code"].astype(str).str.zfill(6))

    snap = fetch_snapshot()
    snap["date"] = pd.to_datetime(snap["ts"], unit="s").dt.normalize()
    snap_date = snap["date"].mode().iloc[0]
    done_date = pd.Timestamp(_last_closed_trade_date())
    print(f"快照 {len(snap)} 行，快照日 {snap_date:%Y-%m-%d}，"
          f"最近收盘交易日 {done_date:%Y-%m-%d}", flush=True)
    if snap_date != done_date:
        print("快照日 ≠ 最近收盘交易日（盘中动态值或非交易日），不落缓存",
              flush=True)
        return

    keep = snap[snap["code"].isin(pool)]
    print(f"池内命中 {len(keep)}/{len(pool)} 只", flush=True)

    updated = created = 0
    for code, grp in keep.groupby("code"):
        row = grp[["date", "main_net", "small_net", "mid_net", "big_net"]].head(1)
        name = f"fund_flow_{code}"
        old = cache_mod.load(CACHE, name)
        if old is not None and not old.empty:
            if str(old["date"].max())[:10] >= f"{snap_date:%Y-%m-%d}":
                continue  # 已有同日或更新数据（历史接口已写）：不覆盖
            k = (pd.concat([old, row], ignore_index=True)
                 .drop_duplicates(subset=["date"], keep="last")
                 .sort_values("date").reset_index(drop=True))
            updated += 1
        else:
            k = row.reset_index(drop=True)
            created += 1
        cache_mod.save(CACHE, name, k)
    print(f"DONE snapshot {snap_date:%Y-%m-%d}: 更新 {updated}，新建 {created}",
          flush=True)


if __name__ == "__main__":
    main()
