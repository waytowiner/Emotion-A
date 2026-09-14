"""抓取交易标的 ETF 的日线（腾讯 ifzq，前复权）-> data/cache/etf_kline_{code}.parquet

用于轮动回测的真实可执行价格。依赖 etf_trading_map.parquet
（先跑 scripts/build_etf_trading_map.py）。缓存增量，可断点续跑。
"""

from __future__ import annotations

import re
import socket
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CACHE = ROOT / "data" / "cache"
START = "2025-01-01"  # 覆盖 180 回测窗口 + 120 平滑 + 缓冲
END = pd.Timestamp.today().strftime("%Y-%m-%d")

socket.setdefaulttimeout(15)


def _tx_code(code6: str) -> str:
    return ("sh" if code6.startswith(("5", "6", "9")) else "sz") + code6


def fetch_etf_kline(code6: str, start: str, end: str) -> pd.DataFrame:
    """腾讯 ifzq 个股/ETF 日K（前复权）。返回 [date, open, close, high, low, volume]。"""
    url = (f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
           f"?param={_tx_code(code6)},day,{start},{end},800,qfq")
    text = None
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 200 and r.text.strip():
                text = r.text
                break
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    if text is None:
        raise RuntimeError(f"腾讯K线抓取失败：{code6}")
    data = __import__("json").loads(text)["data"][_tx_code(code6)]
    rows = data.get("qfqday") or data.get("day") or []
    out = pd.DataFrame(
        [r[:6] for r in rows],
        columns=["date", "open", "close", "high", "low", "volume"])
    out["date"] = pd.to_datetime(out["date"])
    for c in out.columns[1:]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def main() -> None:
    m = pd.read_parquet(CACHE / "etf_trading_map.parquet")
    print(f"交易标的 ETF: {len(m)}", flush=True)
    ok = fail = 0
    consec = 0
    for _, row in m.iterrows():
        code = row["etf_code"]
        p = CACHE / f"etf_kline_{code}.parquet"
        if p.exists():
            old = pd.read_parquet(p)
            if str(old["date"].max())[:10] >= END:
                ok += 1
                continue
        else:
            old = None
        if consec >= 8:
            print(f"连续 {consec} 次失败，中止（可断点续跑）", flush=True)
            break
        try:
            k = fetch_etf_kline(code, START, END)
            if old is not None:
                k = (pd.concat([old, k], ignore_index=True)
                     .drop_duplicates(subset=["date"], keep="last")
                     .sort_values("date").reset_index(drop=True))
            k.to_parquet(p, index=False)
            ok += 1
            consec = 0
            print(f"kline {row['etf_name']}({code}): {len(k)}", flush=True)
        except Exception as exc:  # noqa: BLE001
            fail += 1
            consec += 1
            print(f"kline {code} FAIL: {exc}", flush=True)
        time.sleep(0.3)
    print(f"DONE etf kline ok={ok} fail={fail}", flush=True)


if __name__ == "__main__":
    main()
