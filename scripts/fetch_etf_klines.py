"""抓取交易标的 ETF 的日线（腾讯 ifzq，前复权）-> data/cache/etf_kline_{code}.parquet

用于轮动回测的真实可执行价格。依赖 etf_trading_map.parquet
（先跑 scripts/build_etf_trading_map.py）。缓存增量，可断点续跑。
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def _last_closed_trade_date() -> str:
    """最近一个已收盘的交易日（近似：跳过周末；15:00 前视为昨日）。

    用于 K 线"已最新"判断：盘中运行时缓存到上一交易日即算最新，
    不再全量重发请求（此前用 >= 今天判断，盘中任何重跑都会把全部
    ETF 重新请求一遍）。法定节假日未剔除：多抓一次空数据无害。
    """
    t = pd.Timestamp.now()
    if t.hour < 15:
        t -= pd.Timedelta(days=1)
    d = t.normalize()
    while d.weekday() >= 5:
        d -= pd.Timedelta(days=1)
    return d.strftime("%Y-%m-%d")


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=START)
    ap.add_argument("--end", default=None, help="明确指定已收盘的数据截止日")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    done_date = min(args.end or _last_closed_trade_date(), _last_closed_trade_date())
    m = pd.read_parquet(CACHE / "etf_trading_map.parquet")
    codes = sorted(set(m["etf_code"].astype(str)) | {"510300"})
    print(f"交易标的 ETF + 沪深300基准: {len(codes)}; 截止 {done_date}", flush=True)

    def refresh(code):
        p = CACHE / f"etf_kline_{code}.parquet"
        old = pd.read_parquet(p) if p.exists() else None
        if (not args.force and old is not None
                and str(old["date"].max())[:10] >= done_date):
            return {"etf_code": code, "status": "cached", "rows": len(old),
                    "last_date": str(old["date"].max())[:10]}
        try:
            k = fetch_etf_kline(code, args.start, done_date)
            # 盘中抓到的当日 K 线是动态值：不落缓存，等收盘后
            # （done_date=今天）再补全，否则盘中残值会被"已最新"判断永久保留
            k = k[k["date"].between(pd.Timestamp(args.start), pd.Timestamp(done_date))]
            if k.empty or k[["open", "close", "high", "low"]].isna().any().any():
                raise RuntimeError("无可用K线或价格缺失")
            if (k[["open", "close", "high", "low"]] <= 0).any().any():
                raise RuntimeError("非正价格")
            if old is not None:
                # 全区间重新复权；不混拼区间内的旧复权基准。
                old = old[old["date"] < pd.Timestamp(args.start)]
                k = (pd.concat([old, k], ignore_index=True)
                     .drop_duplicates(subset=["date"], keep="last")
                     .sort_values("date").reset_index(drop=True))
            if k.empty:
                raise RuntimeError("无可用K线")
            k.to_parquet(p, index=False)
            return {"etf_code": code, "status": "updated", "rows": len(k),
                    "last_date": str(k["date"].max())[:10]}
        except Exception as exc:  # noqa: BLE001
            return {"etf_code": code, "status": "failed", "error": str(exc)}

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(refresh, code) for code in codes]):
            row = future.result()
            results.append(row)
            print(f"[{len(results)}/{len(codes)}] {row}", flush=True)
    manifest = {"source": "Tencent ifzq fqkline (qfqday or day)",
                "fetched_at": pd.Timestamp.now().isoformat(),
                "start": args.start, "end": done_date,
                "results": sorted(results, key=lambda r: r["etf_code"])}
    (CACHE / "etf_fetch_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    fail = sum(r["status"] == "failed" for r in results)
    print(f"DONE etf kline ok={len(results)-fail} fail={fail}", flush=True)
    if fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
