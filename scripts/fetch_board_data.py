"""抓取东财板块数据：板块列表 -> 成分股 -> 板块指数日线。缓存增量，可断点续跑。

日常增量性能要点：
- 成分股/K线缓存有效时零请求、零 sleep（此前每个板块固定 sleep 0.8/0.5s，
  全部命中缓存也要白等约 2 分钟）；
- 启动先探测东财连通性：不可达时全部沿用缓存、秒级结束（退出码 0），
  后续个股更新（baostock，独立通道）与看板导出仍可继续；
- 单板块 K 线抓取用轻量重试（有缓存时失败自动沿用缓存），K 线循环设
  软预算熔断，避免网络抖动时重试风暴拖满步骤超时（曾单次卡满 1800s
  被判「板块成分/指数」失败）；
- 收尾时若板块 K 线落后于个股日线（东财不可达期间），用成分股等权
  收益合成补齐缺口（官方数据恢复后按日期去重自动覆盖合成段）。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from emotion.data import boards as boards_mod
from emotion.data import cache as cache_mod
from emotion.data import quotes as quotes_mod
from emotion.transport import get_json

CACHE = ROOT / "data" / "cache"
START = "2023-01-01"
END = pd.Timestamp.today().strftime("%Y-%m-%d")

_KLINE_BUDGET_S = 600  # K线循环软预算（秒）：超出后剩余板块沿用缓存


def _em_reachable() -> bool:
    """启动探测：push2his 拉一根 BK0448 日线，判断东财行情接口是否可达。"""
    try:
        get_json(
            "https://push2his.eastmoney.com/api/qt/stock/kline/get",
            params={
                "secid": "90.BK0448", "ut": quotes_mod._UT,
                "fields1": "f1,f2,f3,f4,f5,f6", "fields2": "f51,f53",
                "klt": 101, "fqt": 0, "beg": "20240101", "end": "20500101",
                "lmt": 1,
            },
            timeout=8, max_retries=1)
        return True
    except Exception:  # noqa: BLE001
        return False


def _before_close() -> bool:
    """现在是否在今天 15:30 之前（当日 K 线仍是盘中动态值，需允许重抓覆盖）。"""
    now = pd.Timestamp.now()
    return (now.hour, now.minute) < (15, 30)


def _stock_last_date() -> str | None:
    """个股日线 chunk 的最新交易日（只读 date 列，秒级）。"""
    last = None
    for p in sorted((ROOT / "data" / "cache" / "stock_chunks").glob("chunk_*.parquet")):
        try:
            d = pd.read_parquet(p, columns=["date"])["date"]
        except Exception:  # noqa: BLE001
            continue
        if d.empty:
            continue
        mx = str(d.max())[:10]
        if last is None or mx > last:
            last = mx
    return last


def _synthesize_board_kline_gaps(boards: pd.DataFrame) -> None:
    """板块 K 线落后于个股日线时，用成分股等权日收益合成补齐缺口。

    合成段的 close 是等权收益累乘的再造指数（非官方点位）；东财恢复后
    get_board_kline 增量抓取按 date 去重（keep="last"）自动用官方数据覆盖
    合成段，无缝切回。板块成分划分口径不变，横截面因子（rps 等）不受影响。
    """
    stock_last = _stock_last_date()
    if stock_last is None:
        return
    behind = []
    for _, row in boards.iterrows():
        k = cache_mod.load(CACHE, f"board_kline_{row['board_code']}")
        if k is None or k.empty:
            continue  # 无历史基准，无法合成
        last = str(k["date"].max())[:10]
        if last < stock_last:
            behind.append((row["board_code"], last))
    if not behind:
        return
    print(f"板块K线落后个股日线（{stock_last}），成分股等权合成补位："
          f"{len(behind)} 个板块", flush=True)

    cons = cache_mod.load(CACHE, "all_cons")
    if cons is None or cons.empty:
        print("all_cons 缺失，跳过合成", flush=True)
        return
    gap_from = min(last for _, last in behind)

    frames = []
    for p in sorted((ROOT / "data" / "cache" / "stock_chunks").glob("chunk_*.parquet")):
        df = pd.read_parquet(
            p, columns=["date", "code", "pctChg", "amount", "tradestatus"])
        df = df[df["date"] > gap_from]  # chunk 的 date 为 ISO 字符串
        if not df.empty:
            frames.append(df)
    if not frames:
        return
    sd = pd.concat(frames, ignore_index=True)
    sd = sd[sd["tradestatus"].astype(str) == "1"]  # 与因子同口径：剔除停牌
    sd["pctChg"] = pd.to_numeric(sd["pctChg"], errors="coerce")
    sd["amount"] = pd.to_numeric(sd["amount"], errors="coerce")
    sd["code"] = sd["code"].astype(str).str.zfill(6)

    m = cons[["code", "board_code"]].drop_duplicates()
    m["code"] = m["code"].astype(str).str.zfill(6)
    g = sd.merge(m, on="code", how="inner").groupby(["board_code", "date"]).agg(
        pct_chg=("pctChg", "mean"), amount=("amount", "sum")).reset_index()

    done = 0
    for code, last in behind:
        sub = g[(g["board_code"] == code) & (g["date"] > last)].sort_values("date")
        if sub.empty:
            continue
        k = cache_mod.load(CACHE, f"board_kline_{code}")
        base_close = float(pd.to_numeric(k["close"], errors="coerce").dropna().iloc[-1])
        new = pd.DataFrame({
            "date": pd.to_datetime(sub["date"]),
            # 官方 K 线为 [date, open, close, high, low, volume, amount, ...]，
            # 合成段只填 close/amount/pct_chg，其余留 NaN
            "close": base_close * (1 + sub["pct_chg"] / 100).cumprod().to_numpy(),
            "amount": sub["amount"].to_numpy(),
            "pct_chg": sub["pct_chg"].to_numpy(),
        })
        out = pd.concat([k, new], ignore_index=True)
        out = out.drop_duplicates(subset=["date"], keep="last").sort_values("date")
        cache_mod.save(CACHE, f"board_kline_{code}", out.reset_index(drop=True))
        done += 1
    print(f"合成补位完成：{done} 个板块补齐到 {stock_last}（东财恢复后自动覆盖）",
          flush=True)


def main() -> None:
    boards = boards_mod.get_board_list(CACHE, refresh=False)
    print(f"boards: {len(boards)}", flush=True)

    net_ok = _em_reachable()
    print(f"eastmoney reachable: {net_ok}", flush=True)
    if not net_ok:
        print("东财不可达：本次全部沿用缓存，跳过网络抓取", flush=True)

    ok_c = fail_c = skip_c = 0
    for _, row in boards.iterrows():
        cached = cache_mod.load(CACHE, f"board_cons_{row['board_code']}")
        if cached is not None and not cached.empty:
            ok_c += 1
            continue  # 缓存有效：无请求、无 sleep
        if not net_ok:
            skip_c += 1
            continue
        try:
            cons = boards_mod.get_board_cons(CACHE, row["board_code"], refresh=False)
            ok_c += 1
            print(f"cons {row['board_name']}: {len(cons)}", flush=True)
        except Exception as exc:  # noqa: BLE001
            fail_c += 1
            print(f"cons {row['board_name']} FAIL: {exc}", flush=True)
        time.sleep(0.8)

    # 合并成分总表
    frames = []
    for _, row in boards.iterrows():
        df = cache_mod.load(CACHE, f"board_cons_{row['board_code']}")
        if df is not None:
            frames.append(df.assign(board_code=row["board_code"],
                                    board_name=row["board_name"]))
    if frames:
        cache_mod.save(CACHE, "all_cons", pd.concat(frames, ignore_index=True))
        print(f"all_cons saved: {sum(len(f) for f in frames)} rows", flush=True)

    ok_k = fail_k = skip_k = 0
    t0 = time.time()
    for _, row in boards.iterrows():
        cached = cache_mod.load(CACHE, f"board_kline_{row['board_code']}")
        if cached is not None and not cached.empty:
            last = str(cached["date"].max())[:10]
            # 已最新；但缓存最后一根就是今天且尚未收盘时不能跳过——
            # 那是盘中动态值，需重抓覆盖为收盘终值
            if last >= END and not (last == END and _before_close()):
                ok_k += 1
                continue  # 无请求、无 sleep
        if not net_ok or time.time() - t0 > _KLINE_BUDGET_S:
            skip_k += 1
            continue
        try:
            k = quotes_mod.get_board_kline(CACHE, row["board_code"], START, END,
                                           timeout=10, max_retries=2)
            ok_k += 1
            print(f"kline {row['board_name']}: {len(k)}", flush=True)
        except Exception as exc:  # noqa: BLE001
            fail_k += 1
            print(f"kline {row['board_name']} FAIL: {exc}", flush=True)
        time.sleep(0.5)

    print(f"DONE cons ok={ok_c} fail={fail_c} skip={skip_c}; "
          f"kline ok={ok_k} fail={fail_k} skip={skip_k}", flush=True)

    _synthesize_board_kline_gaps(boards)


if __name__ == "__main__":
    main()
