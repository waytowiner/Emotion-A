"""数据准备管线：把原始数据装配成因子层需要的三个长表。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from .data import boards as boards_mod
from .data import quotes as quotes_mod

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict:
    p = Path(path) if path else ROOT / "config.yaml"
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_stock_chunks(cache_dir: str | Path) -> pd.DataFrame:
    """读取后台抓取的全部个股日线 chunk，合并为长表。"""
    chunk_dir = Path(cache_dir) / "stock_chunks"
    files = sorted(chunk_dir.glob("chunk_*.parquet"))
    if not files:
        raise RuntimeError(
            f"无个股日线数据：请先运行 scripts/fetch_stock_history.py（{chunk_dir}）")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df.rename(columns={"pctChg": "pct_chg", "tradestatus": "trade_status"})
    for c in ["close", "high", "preclose", "pct_chg", "amount", "turn"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "isST" in df.columns:
        df["isST"] = pd.to_numeric(df["isST"], errors="coerce").fillna(0).astype(int)
    df["date"] = pd.to_datetime(df["date"])
    df["code"] = df["code"].astype(str).str.zfill(6)
    # 跨 chunk 去重保护：抓取脚本按 chunk 隔离，正常不重复；重抓/补抓异常时
    # 同一 (code,date) 可能出现两次，会双算因子。保留最后一条，保险。
    df = df.drop_duplicates(subset=["code", "date"], keep="last")
    return df


def build_board_kline_from_stocks(stock_daily: pd.DataFrame,
                                  board_map: pd.DataFrame) -> pd.DataFrame:
    """降级：东财板块指数不可用时，用成分股等权日收益合成板块指数。"""
    m = board_map[["code", "board_code"]].drop_duplicates("code")
    d = stock_daily.merge(m, on="code", how="inner")
    g = d.groupby(["board_code", "date"]).agg(
        pct_chg=("pct_chg", "mean"), amount=("amount", "sum")).reset_index()
    g = g.sort_values(["board_code", "date"])
    g["close"] = g.groupby("board_code")["pct_chg"].transform(
        lambda s: (1 + s / 100).cumprod() * 100)
    return g[["board_code", "date", "close", "pct_chg", "amount"]]


def prepare(cfg: dict, *, refresh_boards: bool = False) -> dict:
    """返回 {stock_daily, board_map, board_kline, boards(DataFrame), board_names, market_total}。"""
    cache_dir = ROOT / cfg["paths"]["cache_dir"]
    start = cfg["history"]["start_date"]
    end = pd.Timestamp.today().strftime("%Y-%m-%d")

    # 1) 板块映射（东财行业板块优先，同花顺备用，baostock 行业兜底）
    board_source = cfg.get("board_source", "eastmoney")
    board_map = None
    boards = None
    board_kline = None
    if board_source == "ths":
        try:
            from .data import boards_ths
            boards = boards_ths.get_board_list_ths(cache_dir, refresh=refresh_boards)
            frames, cons_fail = [], 0
            for _, row in boards.iterrows():
                try:
                    cons = boards_ths.get_board_cons_ths(
                        cache_dir, row["board_code"], refresh=refresh_boards)
                    frames.append(cons.assign(board_code=row["board_code"],
                                              board_name=row["board_name"]))
                except Exception as exc:  # noqa: BLE001
                    cons_fail += 1
                    if cons_fail <= 3:
                        print(f"[warn] THS 成分缺失 {row['board_name']}: {str(exc)[:60]}")
            if frames:
                board_map = pd.concat(frames, ignore_index=True)
            if cons_fail:
                print(f"[warn] THS 成分：{cons_fail}/{len(boards)} 个板块失败（榜单将不含这些板块）")
            kframes, kline_fail = [], 0
            for _, row in boards.iterrows():
                try:
                    k = boards_ths.get_board_kline_ths(
                        cache_dir, row["board_name"], start, end, refresh=refresh_boards)
                    kframes.append(k.assign(board_code=row["board_code"]))
                except Exception as exc:  # noqa: BLE001
                    kline_fail += 1
                    if kline_fail <= 3:
                        print(f"[warn] THS K线缺失 {row['board_name']}: {str(exc)[:60]}")
            if kframes:
                board_kline = pd.concat(kframes, ignore_index=True)
            if kline_fail:
                print(f"[warn] THS K线：{kline_fail}/{len(boards)} 个板块失败")
        except Exception as exc:
            print(f"[warn] 同花顺板块不可用，降级处理：{exc}")
    if board_map is None and board_source == "eastmoney":
        try:
            boards = boards_mod.get_board_list(cache_dir, refresh=refresh_boards)
            board_map = boards_mod.get_all_cons(cache_dir, boards,
                                                refresh=refresh_boards)
        except Exception as exc:
            print(f"[warn] 东财板块不可用，降级 baostock 行业分类：{exc}")
    if board_map is None:
        board_map = boards_mod.get_board_map_baostock(cache_dir)
        boards = board_map[["board_code", "board_name"]].drop_duplicates()

    board_names = dict(zip(boards["board_code"], boards["board_name"]))

    # 2) 板块指数日线（东财 push2his；失败时成分等权合成）
    if board_kline is None and board_source == "eastmoney" and boards is not None:
        try:
            board_kline = quotes_mod.get_all_board_klines(cache_dir, boards, start, end)
        except Exception as exc:
            print(f"[warn] 东财板块指数不可用，将用成分股等权合成：{exc}")

    # 3) 个股日线（baostock chunk）
    stock_daily = load_stock_chunks(cache_dir)

    # 统一按配置截断历史：缓存可能远早于 start_date（增量缓存整包返回，
    # load_stock_chunks 也不按日期过滤），config 的 history.start_date 在此生效
    start_ts = pd.Timestamp(start)
    stock_daily = stock_daily[stock_daily["date"] >= start_ts]

    # 全市场日成交额（热度变化因子的真实分母，须在板块归属过滤前计算）
    market_total = stock_daily.groupby("date")["amount"].sum()

    # 只保留有板块归属的个股，减小计算量
    valid = set(board_map["code"])
    stock_daily = stock_daily[stock_daily["code"].isin(valid)]

    if board_kline is None:
        board_kline = build_board_kline_from_stocks(stock_daily, board_map)
    board_kline = board_kline[pd.to_datetime(board_kline["date"]) >= start_ts]

    return {
        "stock_daily": stock_daily,
        "board_map": board_map,
        "board_kline": board_kline,
        "boards": boards,
        "board_names": board_names,
        "market_total": market_total,
    }
