"""抓取全量 ETF 现货清单 -> data/cache/etf_list.parquet

东财 push2 clist 接口在本机间歇风控，采用"后台循环重试直到成功"模式：
每轮经 emotion.transport.get_json（curl/curl_cffi/requests 三通道）尝试
多个 host，全部失败则休眠后重来，默认最长 40 分钟。

用法：python scripts/fetch_etf_list.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from emotion.transport import get_json

CACHE = ROOT / "data" / "cache"

HOSTS = [
    "https://82.push2.eastmoney.com",
    "https://33.push2.eastmoney.com",
    "https://7.push2.eastmoney.com",
    "https://push2.eastmoney.com",
]
# 与 akshare fund_etf_spot_em 相同的 fs：全部场内 ETF
FS = "b:MK0021,b:MK0022,b:MK0023,b:MK0024"
FIELDS = "f12,f14,f2,f3,f5,f6,f20,f21"  # 代码/名称/最新价/涨跌幅/成交量/成交额/总市值/流通市值
MAX_MINUTES = 40


def fetch_once() -> pd.DataFrame | None:
    """单轮尝试：任一 host 成功即返回全量 ETF 表。"""
    params = {
        "pn": 1, "pz": 3000, "po": 1, "np": 1, "fltt": 2, "invt": 2,
        "fid": "f3", "fs": FS, "fields": FIELDS,
    }
    for host in HOSTS:
        try:
            data = get_json(f"{host}/api/qt/clist/get", params=params,
                            timeout=12, max_retries=2)
        except Exception as exc:
            print(f"  {host} 失败：{str(exc)[:70]}", flush=True)
            continue
        diff = (data.get("data") or {}).get("diff") or []
        if not diff:
            print(f"  {host} 返回空（疑似风控）", flush=True)
            continue
        df = pd.DataFrame(diff).rename(columns={
            "f12": "code", "f14": "name", "f2": "price", "f3": "pct_chg",
            "f5": "volume", "f6": "amount", "f20": "mktcap", "f21": "float_mktcap",
        })
        df = df[["code", "name", "price", "pct_chg", "volume", "amount",
                 "mktcap", "float_mktcap"]]
        df["code"] = df["code"].astype(str)
        for c in df.columns[2:]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.dropna(subset=["name"]).reset_index(drop=True)
    return None


SINA_URL = ("https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/"
            "Market_Center.getHQNodeData")


def fetch_sina() -> pd.DataFrame | None:
    """兜底：新浪 ETF 行情（分页，字段与东财对齐）。网络通即可，不经 push2。"""
    import json as _json
    import subprocess
    from urllib.parse import urlencode
    rows = []
    for page in range(1, 40):
        params = {"page": page, "num": 100, "sort": "symbol", "asc": 1,
                  "node": "etf_hq_fund"}
        url = f"{SINA_URL}?{urlencode(params)}"
        try:
            out = subprocess.run(["curl", "-s", "-m", "12", "--noproxy", "*", url],
                                 capture_output=True, timeout=20)
            data = _json.loads(out.stdout.decode("utf-8", errors="ignore") or "null")
        except Exception as exc:
            print(f"  新浪第 {page} 页失败：{str(exc)[:60]}", flush=True)
            return None
        if not data:
            break
        rows.extend(data)
        if len(data) < 100:
            break
        time.sleep(0.3)
    if not rows:
        return None
    df = pd.DataFrame(rows)
    out = pd.DataFrame({
        "code": df["code"].astype(str),
        "name": df["name"].astype(str),
        "price": pd.to_numeric(df["trade"], errors="coerce"),
        "pct_chg": pd.to_numeric(df["changepercent"], errors="coerce"),
        "volume": pd.to_numeric(df["volume"], errors="coerce"),
        "amount": pd.to_numeric(df["amount"], errors="coerce"),
        # 新浪 mktcap/nmc 单位为万元，换算为元
        "mktcap": pd.to_numeric(df["mktcap"], errors="coerce") * 1e4,
        "float_mktcap": pd.to_numeric(df["nmc"], errors="coerce") * 1e4,
    })
    return out.dropna(subset=["name"]).reset_index(drop=True)


def main() -> None:
    deadline = time.time() + MAX_MINUTES * 60
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        print(f"[{time.strftime('%H:%M:%S')}] 第 {attempt} 轮尝试 ...", flush=True)
        df = fetch_once()
        if df is None:
            print("  东财不可用，尝试新浪兜底 ...", flush=True)
            df = fetch_sina()
            if df is not None:
                print(f"  新浪通道成功（{len(df)} 只）", flush=True)
        if df is not None and len(df) > 100:
            out = CACHE / "etf_list.parquet"
            df.to_parquet(out, index=False)
            print(f"成功：{len(df)} 只 ETF -> {out}", flush=True)
            print(df.head(5).to_string(), flush=True)
            return
        wait = min(60 + attempt * 15, 180)
        print(f"  本轮未成功，{wait}s 后重试", flush=True)
        time.sleep(wait)
    print(f"超过 {MAX_MINUTES} 分钟仍未成功，退出（稍后可直接重跑本脚本）", flush=True)
    sys.exit(1)


if __name__ == "__main__":
    main()
