"""数据准备管线：把原始数据装配成因子层需要的三个长表。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from .data import boards as boards_mod
from .data import quotes as quotes_mod
from .factors import FACTOR_NAMES
from .universe import select_board_universe

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict:
    p = Path(path) if path else ROOT / "config.yaml"
    with open(p, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    validate_config(cfg, path=p)
    return cfg


def validate_config(cfg: dict, *, path: Path | None = None) -> None:
    """对会静默改变研究结论的配置错误快速失败。

    过去 ``weights.fund_flow`` 没有对应因子时会被合成层悄悄忽略；这类问题
    会让配置、报告和实际计算各说各话。这里仅校验影响核心口径的字段，仍允许
    用户在配置中加入自己的说明性字段。
    """
    if not isinstance(cfg, dict):
        raise ValueError(f"配置必须是 YAML 对象：{path or 'config.yaml'}")
    weights = cfg.get("weights")
    if not isinstance(weights, dict) or not weights:
        raise ValueError("weights 必须是非空映射")
    unknown_weights = sorted(set(weights) - set(FACTOR_NAMES))
    if unknown_weights:
        raise ValueError(
            f"weights 包含未实现的因子 {unknown_weights}；可用因子：{list(FACTOR_NAMES)}")
    if not any(float(v) > 0 for v in weights.values()):
        raise ValueError("weights 至少需要一个正权重")
    uni = cfg.get("universe", {})
    if not isinstance(uni, dict):
        raise ValueError("universe 必须是映射")
    if int(uni.get("min_constituents", 1)) < 1:
        raise ValueError("universe.min_constituents 必须 >= 1")
    threshold = float(uni.get("duplicate_jaccard", 1.0))
    if not 0 < threshold <= 1:
        raise ValueError("universe.duplicate_jaccard 必须在 (0, 1] 内")
    signal_cfg = cfg.get("signal", {})
    if not isinstance(signal_cfg, dict):
        raise ValueError("signal 必须是映射")
    for key in ("hot_quantile", "cold_quantile"):
        if key in signal_cfg and not 0 < float(signal_cfg[key]) < 1:
            raise ValueError(f"signal.{key} 必须在 (0, 1) 内")


def cache_end_date(cfg: dict) -> str:
    """数据可用的最新完整交易日：板块K线与个股日线缓存截止日的较小者。

    板块口径为"至少 95% 板块覆盖的最新日期"（0.05 分位向下取整），
    单个长期失败的板块不会把整体口径拖回旧日期，部分板块限流滞后时
    也不会虚标进度日。
    用于避免增量网络抓取（传输层可能进程级崩溃），也防止单一源的超前行
    （如中证接口凌晨预发的次日占位行）导致报告日期虚标。
    """
    cache_dir = ROOT / cfg["paths"]["cache_dir"]
    src = cfg.get("board_source", "eastmoney")
    pattern = {"eastmoney": "board_kline_BK*", "ths": "board_kline_ths_*",
               "sw": "board_kline_sw_*", "etf": "board_kline_etf_*",
               }.get(src, "board_kline_*") + ".parquet"
    lasts = []
    for f in sorted(Path(cache_dir).glob(pattern)):
        try:
            d = pd.read_parquet(f, columns=["date"])["date"]
            if not d.empty:
                lasts.append(pd.to_datetime(d).max())
        except Exception:  # noqa: BLE001
            continue
    # 板块池取"覆盖分位"口径：至少 95% 板块 K 线覆盖的最新日期（0.05 分位、
    # 向下取实际日期）。此前用 max（任一板块抓到即算），限流中止日会把
    # 大量缓存滞后的板块推入增量网络抓取（2026-10-10 实测致 main.py 在
    # 代理上无限等待）。正常日（全部板块同日）分位口径与 max 一致。
    board_last = (pd.Series(sorted(lasts)).quantile(0.05, interpolation="lower")
                  if lasts else None)
    stock_last = None
    for f in sorted(Path(cache_dir).glob("stock_chunks/chunk_*.parquet")):
        try:
            d = pd.read_parquet(f, columns=["date"])["date"]
            if not d.empty:
                mx = pd.to_datetime(d).max()
                if stock_last is None or mx > stock_last:
                    stock_last = mx
        except Exception:  # noqa: BLE001
            continue
    end = board_last
    if end is None or (stock_last is not None and stock_last < end):
        end = stock_last
    if end is None:
        return pd.Timestamp.today().strftime("%Y-%m-%d")
    return end.strftime("%Y-%m-%d")


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


def _exclude_star(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """排除科创板（688 开头，50 万门槛）。"""
    if cfg.get("history", {}).get("exclude_star", False):
        df = df[~df["code"].astype(str).str.startswith("688")]
    return df


def build_board_kline_from_stocks(stock_daily: pd.DataFrame,
                                  board_map: pd.DataFrame) -> pd.DataFrame:
    """降级：东财板块指数不可用时，用成分股等权日收益合成板块指数。

    一股多板：只按 (code, board_code) 对去重，一只股票可计入其所属的全部板块。
    """
    m = board_map[["code", "board_code"]].drop_duplicates()
    d = stock_daily.merge(m, on="code", how="inner")
    g = d.groupby(["board_code", "date"]).agg(
        pct_chg=("pct_chg", "mean"), amount=("amount", "sum")).reset_index()
    g = g.sort_values(["board_code", "date"])
    g["close"] = g.groupby("board_code")["pct_chg"].transform(
        lambda s: (1 + s / 100).cumprod() * 100)
    return g[["board_code", "date", "close", "pct_chg", "amount"]]


def prepare(cfg: dict, *, refresh_boards: bool = False,
            end_override: str | None = None,
            offline: bool = False) -> dict:
    """返回 {stock_daily, board_map, board_kline, boards(DataFrame), board_names, market_total}。

    end_override：统一截断日期（不触发增量网络抓取时用缓存截止日）。
    offline：是否强制纯离线模式（只读本地已存缓存，不发起任何网络请求）。
    """
    cache_dir = ROOT / cfg["paths"]["cache_dir"]
    start = cfg["history"]["start_date"]
    end = end_override or pd.Timestamp.today().strftime("%Y-%m-%d")

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
                        cache_dir, row["board_name"], start, end, refresh=refresh_boards,
                        offline=offline)
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
    if board_map is None and board_source == "sw":
        # 申万二级行业（131 个，全市场完整分区，经 akshare）
        try:
            from .data import boards_sw
            boards = boards_sw.get_board_list_sw(cache_dir, refresh=refresh_boards)
            all_sw_cons = cache_dir / "all_cons_sw.parquet"
            if not refresh_boards and all_sw_cons.exists():
                board_map = pd.read_parquet(all_sw_cons)
            else:
                frames, cons_fail = [], 0
                for _, row in boards.iterrows():
                    try:
                        cons = boards_sw.get_board_cons_sw(
                            cache_dir, row["board_code"], refresh=refresh_boards)
                        frames.append(cons.assign(board_code=row["board_code"],
                                                  board_name=row["board_name"]))
                    except Exception as exc:  # noqa: BLE001
                        cons_fail += 1
                        if cons_fail <= 3:
                            print(f"[warn] 申万成分缺失 {row['board_name']}: {str(exc)[:60]}")
                if frames:
                    board_map = pd.concat(frames, ignore_index=True)
                if cons_fail:
                    print(f"[warn] 申万成分：{cons_fail}/{len(boards)} 个板块失败")
            kframes, kline_fail = [], 0
            for _, row in boards.iterrows():
                try:
                    k = boards_sw.get_board_kline_sw(
                        cache_dir, row["board_code"], start, end,
                        refresh=refresh_boards, offline=offline)
                    kframes.append(k.assign(board_code=row["board_code"]))
                except Exception as exc:  # noqa: BLE001
                    kline_fail += 1
                    if kline_fail <= 3:
                        print(f"[warn] 申万K线缺失 {row['board_name']}: {str(exc)[:60]}")
            if kframes:
                board_kline = pd.concat(kframes, ignore_index=True)
            if kline_fail:
                print(f"[warn] 申万 K线：{kline_fail}/{len(boards)} 个板块失败")
        except Exception as exc:
            print(f"[warn] 申万板块不可用，降级处理：{exc}")
    if board_map is None and board_source == "etf":
        # 可交易 ETF 板块池（中证行业/主题指数，全部有跟踪产品=买得到）
        try:
            from .data import boards_etf
            boards = boards_etf.get_board_list_etf(cache_dir, refresh=refresh_boards,
                                                  offline=offline)
            all_etf_cons = cache_dir / "all_cons_etf.parquet"
            if not refresh_boards and all_etf_cons.exists():
                board_map = pd.read_parquet(all_etf_cons)
            else:
                frames, cons_fail = [], 0
                for _, row in boards.iterrows():
                    try:
                        cons = boards_etf.get_board_cons_etf(
                            cache_dir, row["board_code"], refresh=refresh_boards,
                            offline=offline)
                        frames.append(cons.assign(board_code=row["board_code"],
                                                  board_name=row["board_name"]))
                    except Exception as exc:  # noqa: BLE001
                        cons_fail += 1
                        if cons_fail <= 3:
                            print(f"[warn] ETF板块成分缺失 {row['board_name']}: {str(exc)[:60]}")
                if frames:
                    board_map = pd.concat(frames, ignore_index=True)
                if cons_fail:
                    print(f"[warn] ETF板块成分：{cons_fail}/{len(boards)} 个失败")
            kframes, kline_fail = [], 0
            for _, row in boards.iterrows():
                try:
                    k = boards_etf.get_board_kline_etf(
                        cache_dir, row["board_code"], start, end,
                        refresh=refresh_boards, offline=offline)
                    kframes.append(k.assign(board_code=row["board_code"]))
                except Exception as exc:  # noqa: BLE001
                    kline_fail += 1
                    if kline_fail <= 3:
                        print(f"[warn] ETF板块K线缺失 {row['board_name']}: {str(exc)[:60]}")
            if kframes:
                board_kline = pd.concat(kframes, ignore_index=True)
            if kline_fail:
                print(f"[warn] ETF板块 K线：{kline_fail}/{len(boards)} 个失败")
        except Exception as exc:
            print(f"[warn] ETF板块池不可用，降级处理：{exc}")
    if board_map is None:
        board_map = boards_mod.get_board_map_baostock(cache_dir)
        boards = board_map[["board_code", "board_name"]].drop_duplicates()

    # 1.5) 标的池治理：先按当前成分快照筛掉极小板块、再去掉完全重复暴露。
    # 这只影响打分/回测用的集合；原始数量及剔除明细随结果一起输出，便于审计。
    uni_cfg = cfg.get("universe", {})
    board_map, boards, universe = select_board_universe(
        board_map, boards,
        enabled=bool(uni_cfg.get("enabled", True)),
        min_constituents=int(uni_cfg.get("min_constituents", 1)),
        duplicate_jaccard=float(uni_cfg.get("duplicate_jaccard", 1.0)),
    )
    universe["board_source"] = board_source
    if board_source == "eastmoney":
        cons_path = cache_dir / "all_cons.parquet"
        if cons_path.exists():
            universe["constituent_snapshot_at"] = datetime.fromtimestamp(
                cons_path.stat().st_mtime).strftime("%Y-%m-%d")
    board_names = dict(zip(boards["board_code"], boards["board_name"]))

    # 2) 板块指数日线（东财 push2his；失败时成分等权合成）
    if board_kline is None and board_source == "eastmoney" and boards is not None:
        try:
            board_kline = quotes_mod.get_all_board_klines(cache_dir, boards, start, end,
                                                          offline=offline)
        except Exception as exc:
            print(f"[warn] 东财板块指数不可用，将用成分股等权合成：{exc}")

    # 3) 个股日线（baostock chunk）
    stock_daily = load_stock_chunks(cache_dir)
    stock_daily = _exclude_star(stock_daily, cfg)

    # 统一按配置截断历史：缓存可能远早于 start_date（增量缓存整包返回，
    # load_stock_chunks 也不按日期过滤），config 的 history.start_date 在此生效
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    stock_daily = stock_daily[stock_daily["date"].between(start_ts, end_ts)]
    # 在板块归属过滤之前确定 A 股交易日，避免跨市场指数把港股开放日、
    # 假期或异常周末行带入 RPS、平滑、前瞻收益及 IC 可知时点。
    trading_calendar = stock_trading_calendar(stock_daily)
    stock_daily = stock_daily[stock_daily["date"].isin(trading_calendar)]

    # 全市场日成交额（热度变化因子的真实分母，须在板块归属过滤前计算）
    market_total = stock_daily.groupby("date")["amount"].sum()

    # 只保留有板块归属的个股，减小计算量
    valid = set(board_map["code"])
    stock_daily = stock_daily[stock_daily["code"].isin(valid)]

    if board_kline is None:
        board_kline = build_board_kline_from_stocks(stock_daily, board_map)
    # 双向截断：缓存可能含 end 之后的行（如中证接口提前发布的次日凌晨占位行）
    board_kline = board_kline[pd.to_datetime(board_kline["date"]) >= start_ts]
    board_kline = board_kline[pd.to_datetime(board_kline["date"]) <= end_ts]
    board_kline = board_kline[board_kline["board_code"].isin(board_names)]
    board_kline, calendar_audit = align_board_calendar(board_kline, trading_calendar)

    return {
        "stock_daily": stock_daily,
        "board_map": board_map,
        "board_kline": board_kline,
        "boards": boards,
        "board_names": board_names,
        "market_total": market_total,
        "universe": universe,
        "trading_calendar": trading_calendar,
        "calendar_audit": calendar_audit,
    }


def stock_trading_calendar(stock_daily: pd.DataFrame) -> pd.DatetimeIndex:
    """采用缓存中实际有 A 股交易记录的日期；不推测缺失日或填充价格。"""
    active = stock_daily
    if "trade_status" in active:
        active = active[pd.to_numeric(active["trade_status"], errors="coerce") == 1]
    dates = pd.DatetimeIndex(pd.to_datetime(active["date"]).dropna()).normalize()
    dates = dates[dates.dayofweek < 5].unique().sort_values()
    if dates.empty:
        raise RuntimeError("个股缓存无有效 A 股交易日，无法对齐板块日历")
    return dates


def align_board_calendar(board_kline: pd.DataFrame, calendar: pd.DatetimeIndex
                         ) -> tuple[pd.DataFrame, dict]:
    """仅在分析层剔除非 A 股交易日；原始跨市场缓存保留。"""
    kline = board_kline.copy()
    kline["date"] = pd.to_datetime(kline["date"]).dt.normalize()
    keep = kline["date"].isin(calendar)
    removed_dates = sorted(kline.loc[~keep, "date"].dropna().unique())
    audit = {"source": "stock_daily_cache", "trading_days": len(calendar),
             "removed_board_rows": int((~keep).sum()),
             "removed_dates": [pd.Timestamp(d).strftime("%Y-%m-%d") for d in removed_dates]}
    return kline.loc[keep].sort_values(["date", "board_code"]).reset_index(drop=True), audit
