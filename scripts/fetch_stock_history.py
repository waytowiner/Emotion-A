"""baostock 全市场 A 股日线 -> data/cache/stock_chunks/chunk_*.parquet

两种模式：
- 全量抓取（默认）：python scripts/fetch_stock_history.py [start_date] [end_date]
  已存在的 chunk 文件直接跳过（断点续跑）。首次建库用。
- 增量更新：python scripts/fetch_stock_history.py --update [end_date]
  对每个已有 chunk 追加其最后交易日之后的新数据；新上市股票写入新 chunk。
  每日更新用此模式，无需删除重抓（约 20-40 分钟，瓶颈是 baostock 逐股查询）。

默认排除科创板（688 开头，50 万门槛）和北交所（baostock 不含）。
如需抓科创板：python scripts/fetch_stock_history.py --include-star
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CHUNK_DIR = ROOT / "data" / "cache" / "stock_chunks"
CHUNK_SIZE = 250

FIELDS = "date,code,close,high,preclose,pctChg,amount,turn,tradestatus,isST"

# 科创板（688）默认仍抓取——板块因子计算需要科创板成分股，排除会导致IC反转。
# 如确认不抓：python scripts/fetch_stock_history.py --exclude-star
INCLUDE_STAR = "--exclude-star" not in sys.argv


def _bs_code(code6: str) -> str:
    """6 位代码 -> baostock 代码（baostock 无北交所）。"""
    return ("sh" if code6.startswith(("6", "9")) else "sz") + "." + code6


def _fetch_batch(bs, batch: list[str], start: str, end: str, relogin) -> list[pd.DataFrame]:
    """抓取一批 baostock 代码的日线，带重试与断线重登。返回原始行 frames。"""
    frames = []
    consecutive_fail = 0
    for code in batch:
        got = False
        for attempt in range(3):
            try:
                rs = bs.query_history_k_data_plus(
                    code, FIELDS,
                    start_date=start, end_date=end,
                    frequency="d", adjustflag="3",
                )
                if rs.error_code != "0":
                    time.sleep(1.5 * (attempt + 1))
                    continue
                rows = []
                while rs.next():
                    rows.append(rs.get_row_data())
                if rows:
                    df = pd.DataFrame(rows, columns=rs.fields)
                    df["code"] = code.split(".")[1]
                    frames.append(df)
                got = True
                break
            except Exception as exc:  # noqa: BLE001
                print(f"  {code} attempt {attempt + 1} failed: {exc}", flush=True)
                time.sleep(2 * (attempt + 1))
        if got:
            consecutive_fail = 0
        else:
            consecutive_fail += 1
        time.sleep(0.15)  # 限速，避免触发 baostock 频率限制
        if consecutive_fail >= 30:
            print("30 consecutive failures -> relogin", flush=True)
            relogin()
            consecutive_fail = 0
    return frames


def _finalize(frames: list[pd.DataFrame], name6: dict[str, str]) -> pd.DataFrame:
    out = pd.concat(frames, ignore_index=True)
    out["name"] = out["code"].map(lambda c: name6.get(c, ""))
    return out


def _next_chunk_id() -> int:
    ids = [int(p.stem.split("_")[1]) for p in CHUNK_DIR.glob("chunk_*.parquet")]
    return max(ids) + 1 if ids else 0


def run_full(bs, codes: list[str], name6: dict[str, str],
             start: str, end: str, relogin) -> None:
    """全量抓取：按股票序号分 chunk，已存在的跳过。"""
    done_chunks = {p.stem for p in CHUNK_DIR.glob("chunk_*.parquet")}
    for ci in range(0, len(codes), CHUNK_SIZE):
        chunk_id = f"chunk_{ci // CHUNK_SIZE:03d}"
        if chunk_id in done_chunks:
            continue
        batch = codes[ci : ci + CHUNK_SIZE]
        frames = _fetch_batch(bs, batch, start, end, relogin)
        if frames:
            out = _finalize(frames, name6)
            out.to_parquet(CHUNK_DIR / f"{chunk_id}.parquet", index=False)
            print(f"{chunk_id} saved ({min(ci + CHUNK_SIZE, len(codes))}/{len(codes)}, "
                  f"{len(frames)} stocks)", flush=True)
        else:
            print(f"{chunk_id} EMPTY - will retry next run", flush=True)


def run_update(bs, codes: list[str], name6: dict[str, str],
               start: str, end: str, relogin) -> None:
    """增量更新：逐 chunk 追加最后交易日之后的数据；新上市股票写入新 chunk。"""
    end_ts = pd.Timestamp(end)
    covered: set[str] = set()
    for path in sorted(CHUNK_DIR.glob("chunk_*.parquet")):
        old = pd.read_parquet(path)
        covered.update(old["code"].astype(str).unique())
        last = pd.to_datetime(old["date"]).max()
        if pd.isna(last) or last >= end_ts:
            print(f"{path.stem} 已是最新（{last:%Y-%m-%d}），跳过", flush=True)
            continue
        begin = (last + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        batch = [_bs_code(c) for c in sorted(old["code"].astype(str).unique())]
        frames = _fetch_batch(bs, batch, begin, end, relogin)
        if frames:
            out = pd.concat([old, _finalize(frames, name6)], ignore_index=True)
            out = out.drop_duplicates(subset=["code", "date"], keep="last")
            out.to_parquet(path, index=False)
            print(f"{path.stem} 更新 {begin}~{end}（+{sum(len(f) for f in frames)} 行）",
                  flush=True)
        else:
            print(f"{path.stem} {begin}~{end} 无新数据", flush=True)

    # 新上市（不在任何 chunk 中）的股票：全量抓取写入新 chunk
    new_codes = [c for c in codes if c.split(".")[1] not in covered]
    if not new_codes:
        print("无新上市股票", flush=True)
        return
    next_id = _next_chunk_id()
    for i in range(0, len(new_codes), CHUNK_SIZE):
        batch = new_codes[i : i + CHUNK_SIZE]
        frames = _fetch_batch(bs, batch, start, end, relogin)
        if frames:
            out = _finalize(frames, name6)
            chunk_id = f"chunk_{next_id:03d}"
            out.to_parquet(CHUNK_DIR / f"{chunk_id}.parquet", index=False)
            print(f"{chunk_id} 新股 saved ({len(frames)} stocks)", flush=True)
            next_id += 1


def main() -> None:
    update_mode = "--update" in sys.argv
    args = [a for a in sys.argv[1:] if a not in ("--update", "--exclude-star")]
    start = args[0] if len(args) > 0 else "2025-09-01"  # 动态：今天往前约330天
    end = args[1] if len(args) > 1 else time.strftime("%Y-%m-%d")

    import baostock as bs

    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    bs.login()
    try:
        rs = bs.query_stock_basic()
        stocks = []
        while rs.error_code == "0" and rs.next():
            row = dict(zip(rs.fields, rs.get_row_data()))
            # type=1 股票，status=1 上市中；baostock 不含北交所
            if row.get("type") == "1" and row.get("status") == "1":
                stocks.append((row["code"], row["code_name"]))
    finally:
        bs.logout()

    codes = [c for c, _ in stocks]
    # 科创板（688）默认保留——因子计算需要。--exclude-star 时排除
    if not INCLUDE_STAR:
        before = len(codes)
        codes = [c for c in codes if not c.split(".")[1].startswith("688")]
        print(f"排除科创板: {before} -> {len(codes)}（减少 {before - len(codes)} 只）", flush=True)
    name6 = {c.split(".")[1]: n for c, n in stocks}
    print(f"total listed stocks: {len(codes)}, range {start} ~ {end}, "
          f"mode={'update' if update_mode else 'full'}", flush=True)

    bs.login()

    def relogin() -> None:
        try:
            bs.logout()
        except Exception:
            pass
        time.sleep(5)
        for i in range(5):
            lg = bs.login()
            if lg.error_code == "0":
                print("relogin ok", flush=True)
                return
            time.sleep(5 * (i + 1))
        raise RuntimeError("baostock relogin failed")

    try:
        if update_mode:
            run_update(bs, codes, name6, start, end, relogin)
        else:
            run_full(bs, codes, name6, start, end, relogin)
    finally:
        bs.logout()
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
