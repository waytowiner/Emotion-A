"""baostock 全市场 A 股日线 -> data/cache/stock_chunks/chunk_*.parquet

两种模式：
- 全量抓取（默认）：python scripts/fetch_stock_history.py [start_date] [end_date]
  已存在的 chunk 文件直接跳过（断点续跑）。首次建库用。
- 增量更新：python scripts/fetch_stock_history.py --update [end_date]
  对每个已有 chunk 追加其最后交易日之后的新数据；新上市股票写入新 chunk。

增量更新带快速路径（run_update_daily）：
- 所有 chunk 已覆盖最新交易日 → 秒级跳过（不再逐股空查 baostock）；
- 缺口恰为最新一个交易日且已收盘（>=15:05）→ 腾讯 qt.gtimg.cn 批量
  行情快照一次性追加（200 只/批，全市场约 30 秒）。东财 clist 在本机
  被 WAF 风控（间歇 RST）不可用，腾讯接口无此问题；停牌股按 baostock
  语义补 tradestatus="0" 平推行；
- 其余情形（多日补抓 / 盘中 / 快照失败）回退 baostock 逐股查询，
  约 20-40 分钟（瓶颈是 baostock 无批量日线接口，只能逐股查）。

科创板（688）默认抓取——板块因子计算需要科创板成分股，排除会导致IC反转。
如需排除：python scripts/fetch_stock_history.py --exclude-star
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

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


def _update_existing_chunks(bs, name6: dict[str, str], start: str, end: str,
                            relogin) -> set[str]:
    """baostock 慢速路径：逐 chunk 追加最后交易日之后的数据。返回已覆盖代码集。"""
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
    return covered


def _fetch_new_listings(bs, codes: list[str], name6: dict[str, str],
                        covered: set[str], start: str, end: str, relogin) -> None:
    """新上市（不在任何 chunk 中）的股票：全量抓取写入新 chunk。"""
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


def run_update(bs, codes: list[str], name6: dict[str, str],
               start: str, end: str, relogin) -> None:
    """增量更新（慢速全路径）：逐 chunk 追加；新上市股票写入新 chunk。"""
    covered = _update_existing_chunks(bs, name6, start, end, relogin)
    _fetch_new_listings(bs, codes, name6, covered, start, end, relogin)


# ---------------------------------------------------------------- 每日快速路径

def _bs_trade_calendar(bs) -> tuple[str | None, set[str]]:
    """baostock 交易日历（近 40 天）：返回 (最新交易日, 交易日集合)。"""
    start = (pd.Timestamp.today() - pd.Timedelta(days=40)).strftime("%Y-%m-%d")
    rs = bs.query_trade_dates(start_date=start, end_date=time.strftime("%Y-%m-%d"))
    days = []
    while rs.error_code == "0" and rs.next():
        row = dict(zip(rs.fields, rs.get_row_data()))
        if row.get("is_trading_day") == "1":
            days.append(row["calendar_date"])
    if not days:
        return None, set()
    return days[-1], set(days)


def _board_kline_calendar() -> tuple[str | None, set[str]]:
    """兜底日历：板块指数K线缓存（更新流程第 1 步刚刷新过）的最新交易日。"""
    cache_dir = ROOT / "data" / "cache"
    best_file = None
    best_last = None
    for f in cache_dir.glob("board_kline_*.parquet"):
        try:
            d = pd.read_parquet(f, columns=["date"])["date"]
        except Exception:  # noqa: BLE001
            continue
        if d.empty:
            continue
        mx = pd.to_datetime(d).max()
        if best_last is None or mx > best_last:
            best_last, best_file = mx, f
    if best_file is None:
        return None, set()
    dates = pd.read_parquet(best_file, columns=["date"])["date"]
    cal = set(pd.to_datetime(dates).dt.strftime("%Y-%m-%d"))
    return best_last.strftime("%Y-%m-%d"), cal


# 腾讯批量行情：200 只/批，全市场约 26 批、30 秒左右。
# 东财 clist 在本机被 WAF 风控（间歇 RST，见 emotion/transport.py 头注），不可用。
_TX_BATCH = 200
_TX_QUOTE_RE = re.compile(r'v_([a-z]{2}\d{6})="([^"]*)"')


def _fetch_tx_snapshot(codes: list[str]) -> pd.DataFrame:
    """腾讯 qt.gtimg.cn 批量收盘快照。返回 [code, name, date8, close, high,
    preclose, pctChg, amount, turn, traded]；traded=False 为停牌（成交量 0，
    腾讯给昨收平推价）。date8 为行情时间戳的日期部分，用于校验快照时效。
    """
    sess = requests.Session()
    sess.trust_env = False  # 忽略系统代理，直连

    def num(f: list[str], j: int) -> float:
        try:
            return float(f[j])
        except (ValueError, IndexError):
            return float("nan")

    rows = []
    prefixed = [("sh" if c.startswith(("6", "9")) else "sz") + c for c in codes]
    for i in range(0, len(prefixed), _TX_BATCH):
        batch = prefixed[i : i + _TX_BATCH]
        text = None
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                r = sess.get("https://qt.gtimg.cn/q=" + ",".join(batch), timeout=10)
                if r.status_code == 200 and r.text.strip():
                    text = r.content.decode("gbk", errors="ignore")
                    break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
            time.sleep(1.5 * (attempt + 1))
        if text is None:
            raise RuntimeError(f"腾讯快照第 {i // _TX_BATCH + 1} 批失败: {last_exc}")
        for m in _TX_QUOTE_RE.finditer(text):
            f = m.group(2).split("~")
            if len(f) < 39:
                continue
            parts = f[35].split("/")  # 现价/成交量(手)/成交额(元)
            amount = float(parts[2]) if len(parts) == 3 and parts[2] else float("nan")
            rows.append({
                "code": m.group(1)[2:], "name": f[1],
                "date8": f[30][:8] if len(f[30]) >= 8 else "",
                "close": num(f, 3), "high": num(f, 33), "preclose": num(f, 4),
                "pctChg": num(f, 32), "amount": amount, "turn": num(f, 38),
                "traded": num(f, 6) > 0,  # 成交量(手)>0 视为当日有交易
            })
        time.sleep(0.2)
    if not rows:
        raise RuntimeError("腾讯快照返回为空")
    snap = pd.DataFrame(rows).drop_duplicates(subset=["code"], keep="last")
    n_traded = int(snap["traded"].sum())
    if n_traded < len(snap) // 2:
        raise RuntimeError(f"快照交易中的股票仅 {n_traded}/{len(snap)} 只，疑似抓取失败")
    return snap.reset_index(drop=True)


def _fmt4(v) -> str:
    return "" if pd.isna(v) else f"{float(v):.4f}"


def _append_snapshot_day(target: str, snap: pd.DataFrame) -> tuple[int, int]:
    """把 target 日的快照追加到各 chunk（单日增量）。返回 (追加行数, 更新chunk数)。

    与 baostock 行语义对齐：正常交易股 tradestatus="1"；停牌股（traded=False）
    补平推行 close=high=preclose=昨收, pctChg/amount 空, turn="0", tradestatus="0"
    （与 chunk 中 baostock 停牌行一致）；快照中缺席的代码（退市/北交所）跳过。
    """
    by_code = snap.set_index("code")
    total_rows = 0
    touched = 0
    for path in sorted(CHUNK_DIR.glob("chunk_*.parquet")):
        old = pd.read_parquet(path)
        last = pd.to_datetime(old["date"]).max()
        if pd.notna(last) and last.strftime("%Y-%m-%d") >= target:
            continue
        old_sorted = old.sort_values("date")
        last_close = old_sorted.groupby("code")["close"].last()
        last_name = old_sorted.groupby("code")["name"].last()
        rows = []
        for code in sorted(old["code"].astype(str).unique()):
            if code not in by_code.index:
                continue  # 快照缺席：退市或北交所，不补行
            r = by_code.loc[code]
            if bool(r["traded"]) and pd.notna(r["close"]) and r["close"] > 0:
                rows.append({
                    "date": target, "code": code,
                    "close": _fmt4(r["close"]), "high": _fmt4(r["high"]),
                    "preclose": _fmt4(r["preclose"]), "pctChg": _fmt4(r["pctChg"]),
                    "amount": "" if pd.isna(r["amount"]) else f"{float(r['amount']):.2f}",
                    "turn": _fmt4(r["turn"]),
                    "tradestatus": "1",
                    "isST": "1" if "ST" in str(r["name"]).upper() else "0",
                    "name": str(r["name"]),
                })
            else:
                # 停牌：腾讯平推价为昨收；缺失时退回 chunk 内最后一根收盘价
                pc = _fmt4(r["close"]) if pd.notna(r["close"]) and r["close"] > 0 \
                    else str(last_close.get(code, ""))
                rows.append({
                    "date": target, "code": code,
                    "close": pc, "high": pc, "preclose": pc,
                    "pctChg": "", "amount": "", "turn": "0",
                    "tradestatus": "0", "isST": "",
                    "name": str(r["name"]) or str(last_name.get(code, "")),
                })
        if not rows:
            continue
        add = pd.DataFrame(rows).reindex(columns=old.columns)
        out = pd.concat([old, add], ignore_index=True)
        out = out.drop_duplicates(subset=["code", "date"], keep="last")
        out.to_parquet(path, index=False)
        total_rows += len(add)
        touched += 1
        print(f"{path.stem} 快照追加 {len(add)} 行（{target}）", flush=True)
    return total_rows, touched


def run_update_daily(bs, codes: list[str], name6: dict[str, str],
                     start: str, end: str, relogin) -> None:
    """每日增量编排：能用腾讯快照一次搞定的就不走 baostock 逐股（20-40 分钟）。

    - 全部 chunk 已覆盖最新交易日 → 秒级跳过（仅补新股）；
    - 缺口恰为最新一个交易日且已收盘 → 腾讯全市场快照一次性追加；
    - 其余（多日补抓 / 盘中 / 快照失败）回退 baostock 逐股路径。
    """
    target, cal = _bs_trade_calendar(bs)
    if target is None:
        target, cal = _board_kline_calendar()
    if target is None:
        print("无可用交易日历，走 baostock 逐股路径", flush=True)
        run_update(bs, codes, name6, start, end, relogin)
        return

    covered: set[str] = set()
    lasts: list[str | None] = []
    n_behind = 0
    one_day_gap = True
    for path in sorted(CHUNK_DIR.glob("chunk_*.parquet")):
        df = pd.read_parquet(path, columns=["code", "date"])
        covered.update(df["code"].astype(str).unique())
        last = pd.to_datetime(df["date"]).max()
        last_s = None if pd.isna(last) else last.strftime("%Y-%m-%d")
        lasts.append(last_s)
        if last_s is None or last_s < target:
            n_behind += 1
            if len([d for d in cal if (last_s or "") < d <= target]) != 1:
                one_day_gap = False

    today = time.strftime("%Y-%m-%d")
    now = time.localtime()
    after_close = (now.tm_hour, now.tm_min) >= (15, 5)

    if n_behind == 0:
        print(f"个股日线已覆盖最新交易日 {target}，跳过逐股抓取", flush=True)
    elif one_day_gap and target == today and after_close:
        try:
            snap = _fetch_tx_snapshot(sorted(covered))
            t8 = target.replace("-", "")
            mism = int(((snap["date8"] != t8) & snap["traded"]).sum())
            if mism > 50:
                raise RuntimeError(f"快照日期与 {target} 不符的已交易股票 {mism} 只")
            n_rows, n_chunks = _append_snapshot_day(target, snap)
            print(f"腾讯快照增量完成：{target} 共追加 {n_rows} 行 / {n_chunks} 个 chunk",
                  flush=True)
            n_behind = 0
        except Exception as exc:  # noqa: BLE001
            print(f"腾讯快照失败（{exc}），回退 baostock 逐股路径", flush=True)

    if n_behind > 0:
        # 可补的最晚完整交易日：盘前/盘中 baostock 尚无今日日线，
        # 逐股空查 5000+ 只会浪费 20-40 分钟（曾实际发生）
        completed = sorted(d for d in cal if d < today or (d == today and after_close))
        fetch_end = completed[-1] if completed else None
        need = fetch_end is not None and any(
            l is None or l < fetch_end for l in lasts)
        if not need:
            print(f"已覆盖最新完整交易日 {fetch_end}；"
                  + ("今日数据收盘后（15:05 起）再更新" if target == today and not after_close
                     else "无待补缺口"), flush=True)
        else:
            print(f"走 baostock 逐股路径（补至 {fetch_end}）", flush=True)
            _update_existing_chunks(bs, name6, start, fetch_end, relogin)

    _fetch_new_listings(bs, codes, name6, covered, start, end, relogin)


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
            run_update_daily(bs, codes, name6, start, end, relogin)
        else:
            run_full(bs, codes, name6, start, end, relogin)
    finally:
        bs.logout()
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
