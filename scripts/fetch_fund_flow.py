"""抓取个股历史资金流向（东财 push2his fflow/daykline）-> data/cache/fund_flow_{code}.parquet

用途：主力净流入因子诊断（scripts/diag_fund_flow.py）——现有 6 因子全部为价格/
情绪衍生，真实资金流是唯一未证伪的正交信息源（report/price_position_factor_*
判读记录）。

口径：fields2=f51,f52,f53,f54,f55 → date / 主力净流入 / 小单 / 中单 / 大单（元）。
股票池 = all_cons_etf.parquet（中证 ETF 板块成分，与因子计算同池）。
缓存增量、可断点续跑；盘中抓到的当日动态值不落缓存（同 fetch_etf_klines.py）。

用法：python scripts/fetch_fund_flow.py [--refresh] [--pilot N]
  --pilot N  只抓前 N 只（验证接口与解析），不写全量进度
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
START = "2025-06-01"  # history.start_date 之前留因子窗口缓冲
FLOW_URL = "https://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get"


def _last_closed_trade_date() -> str:
    """最近一个已收盘的交易日（近似：跳过周末；15:00 前视为昨日）。

    盘中运行时缓存到上一交易日即算最新，不再全量重发请求。
    法定节假日未剔除：多抓一次空数据无害。
    """
    t = pd.Timestamp.now()
    if t.hour < 15:
        t -= pd.Timedelta(days=1)
    d = t.normalize()
    while d.weekday() >= 5:
        d -= pd.Timedelta(days=1)
    return d.strftime("%Y-%m-%d")


def _secid(code: str) -> str:
    """东财 secid：沪市(6 开头)=1.，深市(0/3 开头)=0.；其余按深市试。"""
    return ("1." if code.startswith("6") else "0.") + code


def fetch_fund_flow(code: str) -> pd.DataFrame:
    """单只股票的历史日度资金流向。返回 [date, main_net, small_net, mid_net, big_net]。"""
    data = get_json(
        FLOW_URL,
        params={
            "secid": _secid(code), "ut": _UT,
            "fields1": "f1,f2,f3,f7",
            "fields2": "f51,f52,f53,f54,f55",
            "klt": 101, "lmt": 0,
        },
        timeout=10, max_retries=2)
    klines = (data.get("data") or {}).get("klines") or []
    if not klines:
        raise RuntimeError(f"无资金流数据：{code}")
    rows = [k.split(",")[:5] for k in klines]
    out = pd.DataFrame(rows, columns=["date", "main_net", "small_net",
                                      "mid_net", "big_net"])
    out["date"] = pd.to_datetime(out["date"])
    for c in out.columns[1:]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def main() -> None:
    refresh = "--refresh" in sys.argv
    pilot = None
    if "--pilot" in sys.argv:
        pilot = int(sys.argv[sys.argv.index("--pilot") + 1])

    cons_path = CACHE / "all_cons_etf.parquet"
    if not cons_path.exists():
        raise SystemExit("缺少 all_cons_etf.parquet，请先运行 scripts/fetch_etf_boards.py")
    codes = sorted(pd.read_parquet(cons_path)["code"].astype(str)
                   .str.zfill(6).unique())
    # 只保留沪深 A 股（0/3/6 开头纯数字）：沪港深指数成分里的港股（如 0005.HK）
    # 与北交所（4/8 开头）不在口径内，与 baostock 个股日线保持一致
    codes = [c for c in codes if c.isdigit() and c[0] in "036"]
    if pilot:
        codes = codes[:pilot]
    # 无缓存的排前面：限流恢复期每轮配额只有几十次，若按字典序，已缓存股票
    # 的"昨日→今日"刷新会排在新股票前面，把配额吃光而覆盖率不增长
    codes.sort(key=lambda c: 0 if not (CACHE / f"fund_flow_{c}.parquet").exists()
               else 1)
    print(f"个股资金流向抓取：{len(codes)} 只", flush=True)

    done_date = _last_closed_trade_date()
    start_ts = pd.Timestamp(START)
    done_ts = pd.Timestamp(done_date)
    ok = fail = skip = 0
    consec = 0
    for code in codes:
        name = f"fund_flow_{code}"
        if not refresh:
            cached = cache_mod.load(CACHE, name)
            if cached is not None and not cached.empty \
                    and str(cached["date"].max())[:10] >= done_date:
                skip += 1
                continue  # 已覆盖最近收盘交易日：无请求、无 sleep
        if consec >= 8:
            print(f"连续 {consec} 次失败，疑似被限流，中止（已缓存的可断点续跑）",
                  flush=True)
            break
        try:
            k = fetch_fund_flow(code)
            # 盘中运行时当日资金流是动态值：不落缓存，收盘后再补全
            k = k[(k["date"] >= start_ts) & (k["date"] <= done_ts)]
            if not refresh:
                old = cache_mod.load(CACHE, name)
                if old is not None and not old.empty:
                    k = (pd.concat([old, k], ignore_index=True)
                         .drop_duplicates(subset=["date"], keep="last")
                         .sort_values("date").reset_index(drop=True))
            if k.empty:
                raise RuntimeError(f"截至 {done_date} 无可用数据")
            cache_mod.save(CACHE, name, k)
            ok += 1
            consec = 0
            if pilot or ok % 100 == 0:
                print(f"flow {code}: {len(k)} 行（至 {str(k['date'].max())[:10]}）",
                      flush=True)
        except Exception as exc:  # noqa: BLE001
            fail += 1
            consec += 1
            print(f"flow {code} FAIL: {exc}", flush=True)
        time.sleep(1.5)

    print(f"DONE fund flow ok={ok} fail={fail} skip={skip}", flush=True)


if __name__ == "__main__":
    main()
