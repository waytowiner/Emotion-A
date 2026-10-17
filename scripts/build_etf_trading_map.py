"""生成「指数 → 规模最大 ETF」交易映射表。

链路：etf_list（全市场 ETF，含市值）-> 逐只抓天天基金档案页的"跟踪标的"
-> 名称归一化后匹配中证指数清单 -> 同一指数的多只 ETF 取市值最大者
（用户口径：规模最大最稳定，成交活跃度排名波动大）。

输出：data/cache/etf_trading_map.parquet
  [board_code, board_name, etf_code, etf_name, etf_float_mktcap, etf_amount]

可断点续跑：已抓过的 ETF 跟踪标的存 data/cache/etf_track/_{code}.txt。
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

from emotion.data import cache as cache_mod

CACHE = ROOT / "data" / "cache"
TRACK_DIR = CACHE / "etf_track"

socket.setdefaulttimeout(15)  # 防挂起

_STRIP = ("指数", "收益率", "人民币", "行业", "（", "）", "(", ")", " ", "　")


def _norm(s: str) -> str:
    s = str(s)
    for w in _STRIP:
        s = s.replace(w, "")
    return s


def _fetch_track_name(code: str) -> str | None:
    """天天基金档案页 -> 跟踪标的名称。带本地缓存。"""
    TRACK_DIR.mkdir(parents=True, exist_ok=True)
    p = TRACK_DIR / f"{code}.txt"
    if p.exists():
        t = p.read_text(encoding="utf-8").strip()
        return t or None
    url = f"https://fundf10.eastmoney.com/jbgk_{code}.html"
    for attempt in range(3):
        try:
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
            m = re.search(r"跟踪标的</th><td>([^<]*)</td>", r.text)
            if m:
                name = m.group(1).strip()
                p.write_text(name, encoding="utf-8")
                return name
            p.write_text("", encoding="utf-8")  # 页面无此字段（非指数ETF），缓存空
            return None
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    return None  # 不缓存，下次重试


def main() -> None:
    boards = pd.read_parquet(CACHE / "board_list_etf.parquet")
    idx_all = pd.read_parquet(CACHE / "csindex_all.parquet")
    etfs = pd.read_parquet(CACHE / "etf_list.parquet")
    print(f"指数池 {len(boards)}，全市场 ETF {len(etfs)}", flush=True)

    # 指数名归一化 -> 指数代码（全称+简称双路）
    name2code: dict[str, str] = {}
    for _, r in idx_all.iterrows():
        for col in ("指数全称", "指数简称"):
            n = _norm(r[col])
            if n:
                name2code.setdefault(n, str(r["指数代码"]))

    rows = []
    matched_etf = 0
    for i, etf in enumerate(etfs.itertuples()):
        track = _fetch_track_name(str(etf.code))
        if not track:
            continue
        idx_code = name2code.get(_norm(track))
        if idx_code is None:
            continue
        matched_etf += 1
        rows.append({"index_code": idx_code, "etf_code": str(etf.code),
                     "etf_name": etf.name,
                     "etf_float_mktcap": float(etf.float_mktcap or 0),
                     "etf_amount": float(etf.amount or 0)})
        if (i + 1) % 200 == 0:
            print(f"进度 {i + 1}/{len(etfs)}，已匹配 {matched_etf}", flush=True)
        time.sleep(0.25)

    m = pd.DataFrame(rows)
    print(f"跟踪标的匹配成功 ETF: {matched_etf}", flush=True)

    # 同指数取市值最大的 ETF（用户口径）
    out = (m.sort_values("etf_float_mktcap", ascending=False)
             .drop_duplicates(subset=["index_code"], keep="first"))
    out = out.merge(boards, left_on="index_code", right_on="board_code", how="inner")
    out = out[["board_code", "board_name", "etf_code", "etf_name",
               "etf_float_mktcap", "etf_amount"]].sort_values(
        "etf_float_mktcap", ascending=False).reset_index(drop=True)
    out.to_parquet(CACHE / "etf_trading_map.parquet", index=False)
    print(f"映射表: {len(out)}/{len(boards)} 个指数有对应 ETF", flush=True)
    print(out.head(10).to_string(), flush=True)
    missing = boards[~boards["board_code"].isin(out["board_code"])]
    if len(missing):
        print(f"未匹配到 ETF 的指数 {len(missing)} 个："
              f"{missing['board_name'].head(15).tolist()}", flush=True)


if __name__ == "__main__":
    main()
