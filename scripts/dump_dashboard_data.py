"""导出 Web 看板数据：python scripts/dump_dashboard_data.py

跑完整管线（数据准备 -> 因子 -> 情绪分 -> 信号 -> 回测），
把结果落到 data/dashboard/，供 server.py 快速读取。全程离线（使用缓存数据）。
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from emotion import backtest, leaders, pipeline, score, signal
from emotion.factors import compute_all_factors

OUT_DIR = ROOT / "data" / "dashboard"
BACKTEST_WINDOW_FALLBACK = 180
HEALTH_DAYS = 60          # 健康度滚动窗口（与 recent_signal_health 默认一致）
HEALTH_SERIES_DAYS = 240  # 预计算健康度序列的长度上限
HEALTH_HORIZON = 5


def sanitize(obj):
    """递归把 NaN/inf 转 None，numpy 类型转原生，供 json.dumps。"""
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return obj.strftime("%Y-%m-%d")
    return obj


def dump_json(name: str, obj) -> None:
    path = OUT_DIR / name
    path.write_text(json.dumps(sanitize(obj), ensure_ascii=False), encoding="utf-8")
    print(f"    -> {path.name} ({path.stat().st_size / 1024:.0f} KB)")


def df_records(df: pd.DataFrame) -> list[dict]:
    return df.reset_index().to_dict(orient="records")


def nav_records(s: pd.Series) -> list[dict]:
    return [{"date": d.strftime("%Y-%m-%d"), "nav": float(v)}
            for d, v in s.items() if math.isfinite(v)]


def rotation_payload(rot: dict) -> dict:
    return {
        "nav": nav_records(rot["nav"]),
        "benchmark_nav": nav_records(rot["benchmark_nav"]),
        "strategy": rot["strategy"],
        "benchmark": rot["benchmark"],
        "total_cost_drag": rot["total_cost_drag"],
        "final_nav": rot["final_nav"],
        "final_benchmark": rot["final_benchmark"],
    }


def daily_ic_series(score_df: pd.DataFrame, board_close: pd.DataFrame,
                    horizon: int) -> pd.Series:
    """每日截面 Spearman IC（score vs 前瞻 horizon 日收益）。"""
    fwd = backtest.forward_returns(board_close, horizon)
    common = score_df.index.intersection(fwd.index)
    ics = {}
    for dt in common:
        pair = pd.concat([score_df.loc[dt], fwd.loc[dt]], axis=1,
                         keys=["s", "f"]).dropna()
        if len(pair) < 10:
            continue
        ics[dt] = pair["s"].corr(pair["f"], method="spearman")
    return pd.Series(ics).sort_index()


def parse_switching(items: list[str], board_names: dict[str, str]) -> list[dict]:
    """把 'BK0448↓ → BK0400,BK0401↑' 解析成结构化对象。"""
    out = []
    for raw in items:
        left, _, right = raw.partition("→")
        from_code = left.strip().rstrip("↓").strip()
        to_codes = [c.strip().rstrip("↑").strip() for c in right.split(",") if c.strip()]
        out.append({
            "raw": raw,
            "from_code": from_code,
            "from_name": board_names.get(from_code, from_code),
            "to": [{"code": c, "name": board_names.get(c, c)} for c in to_codes],
        })
    return out


def ensure_offline_fallback() -> None:
    """探测东财行情接口；不可达时把板块 K 线降级为纯缓存读取。

    pipeline 的板块 K 线默认做增量网络更新；东财 push2his 风控期间每个
    板块会串行重试多个 host，prepare 可能挂 15 分钟以上（而缓存是完整的，
    接口失败时本来就回退缓存）。探测不可达时直接用缓存离线计算——只改变
    本脚本的取数路径，不改动 emotion/ 的任何计算逻辑。
    """
    import urllib.request
    try:
        req = urllib.request.Request(
            "https://push2his.eastmoney.com/api/qt/stock/kline/get"
            "?secid=90.BK0448&klt=101&fqt=0&lmt=1&fields1=f1&fields2=f51",
            headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=6):
            return  # 接口可达，走正常增量更新
    except Exception:
        pass
    from emotion.data import cache as _cache
    from emotion.data import quotes as quotes_mod

    def _cache_only(cache_dir, board_code, start, end, **kwargs):
        cached = _cache.load(cache_dir, f"board_kline_{board_code}")
        if cached is None or cached.empty:
            raise RuntimeError(f"东财不可达且无缓存：{board_code}")
        return cached

    quotes_mod.get_board_kline = _cache_only
    print("    [offline] 东财行情接口不可达，板块 K 线使用本地缓存")


def load_etf_info(board_names: dict) -> dict[str, list[dict] | None]:
    """读取 config/etf_map.yaml + etf_list 行情，生成 board_code -> ETF 列表（或 None）。

    ETF 条目附带最新价/涨跌幅/成交额（若 etf_list 缓存存在）。
    任一输入缺失时返回空 dict（看板降级为不显示 ETF 列）。
    """
    map_path = ROOT / "config" / "etf_map.yaml"
    if not map_path.exists():
        print("    [etf] config/etf_map.yaml 不存在，跳过 ETF 信息")
        return {}
    import yaml
    with open(map_path, encoding="utf-8") as f:
        mapping = yaml.safe_load(f)
    quotes = {}
    quote_path = ROOT / "data" / "cache" / "etf_list.parquet"
    if quote_path.exists():
        q = pd.read_parquet(quote_path)
        q["code"] = q["code"].astype(str).str.zfill(6)
        quotes = {r.code: r for r in q.itertuples()}
    else:
        print("    [etf] etf_list.parquet 不存在，ETF 无行情数据")
    out: dict[str, list[dict] | None] = {}
    for code, entry in mapping.items():
        code = str(code)  # yaml 会把纯数字代码（如同花顺 881273）解析成 int
        etfs = entry.get("etf") if isinstance(entry, dict) else None
        if not etfs:
            out[code] = None
            continue
        items = []
        for e in etfs:
            ec = str(e["code"]).zfill(6)
            item = {"code": ec, "name": e["name"]}
            r = quotes.get(ec)
            if r is not None:
                item.update({
                    "price": None if pd.isna(r.price) else float(r.price),
                    "pct_chg": None if pd.isna(r.pct_chg) else float(r.pct_chg),
                    "amount": None if pd.isna(r.amount) else float(r.amount),
                })
            items.append(item)
        out[code] = items
    n = sum(1 for v in out.values() if v)
    print(f"    [etf] {n}/{len(mapping)} 个板块挂接 ETF")
    return out


def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = pipeline.load_config()

    print("[1/6] 准备数据（读取本地缓存）...")
    ensure_offline_fallback()
    data = pipeline.prepare(cfg)
    board_names: dict = data["board_names"]

    print("[2/6] 计算情绪因子 ...")
    factors = compute_all_factors(data["stock_daily"], data["board_map"],
                                  data["board_kline"],
                                  market_total=data.get("market_total"))

    print("[3/6] 合成情绪分 ...")
    _, smoothed, pct = score.build_scores(
        factors, cfg["weights"], cfg.get("smooth_window", 5),
        lookback=cfg.get("score_lookback", 120))

    board_close = data["board_kline"].pivot(
        index="date", columns="board_code", values="close").sort_index()

    print("[4/6] 生成信号与榜单 ...")
    etf_map = load_etf_info(board_names)
    leaders_map = leaders.compute_leaders(data["stock_daily"], data["board_map"])
    n_no_etf = sum(1 for c, v in etf_map.items() if not v)
    print(f"    [leaders] {len(leaders_map)} 个板块计算完成"
          f"（其中 {n_no_etf} 个无 ETF 板块将以龙头股展示）")
    sig_cfg = cfg.get("signal", {})
    signals = signal.detect_signals(
        smoothed, pct,
        consecutive_days=sig_cfg.get("consecutive_days", 3),
        hot_quantile=sig_cfg.get("hot_quantile", 0.9),
        cold_quantile=sig_cfg.get("cold_quantile", 0.1))

    latest_date = smoothed.index[-1]
    last_score = smoothed.iloc[-1].dropna().sort_values(ascending=False)
    last_pct = pct.iloc[-1]

    def rank_list(s: pd.Series) -> list[dict]:
        return [{
            "code": c,
            "name": board_names.get(c, c),
            "score": float(s[c]),
            "pct": float(last_pct[c]) if pd.notna(last_pct.get(c)) else None,
            "etf": etf_map.get(c),
        } for c in s.index]

    ranked = rank_list(last_score)
    health = backtest.recent_signal_health(smoothed, board_close, days=HEALTH_DAYS)

    signals_named = {}
    for key in ["heating", "cooling", "hot", "cold"]:
        signals_named[key] = [{"code": c, "name": board_names.get(c, c),
                               "etf": etf_map.get(c)}
                              for c in signals[key]]
    signals_named["switching"] = parse_switching(signals["switching"], board_names)
    for sw in signals_named["switching"]:
        sw["etf"] = etf_map.get(sw["from_code"])
        for t in sw["to"]:
            t["etf"] = etf_map.get(t["code"])

    dump_json("latest.json", {
        "date": latest_date,
        "board_source": cfg.get("board_source", "eastmoney"),
        "n_boards": int(smoothed.shape[1]),
        "health": health,
        "signals": signals_named,
        "top": ranked[:15],
        "bottom": ranked[-15:][::-1],
        "ranking": ranked,
        "etf_map": etf_map,
        "leaders": leaders_map,
    })

    print("[5/6] 回测（最近窗口）...")
    window = int(cfg.get("backtest_window") or BACKTEST_WINDOW_FALLBACK)
    bt_score = smoothed.tail(window)
    bt_close = board_close.loc[bt_score.index[0]:]
    bt_cfg = cfg["backtest"]
    ic_df = backtest.ic_analysis(bt_score, bt_close, bt_cfg["forward_days"])
    quant_df = backtest.quantile_test(bt_score, bt_close,
                                      horizon=bt_cfg["forward_days"][0])
    top_n = sig_cfg.get("top_n", 3)
    cost_bps = bt_cfg.get("cost_bps", 5)
    rotations = {
        d: backtest.rotation_backtest(bt_score, bt_close, top_n=top_n,
                                      cost_bps=cost_bps, direction=d)
        for d in ["top", "bottom", "avoid_top"]
    }
    dump_json("backtest.json", {
        "window": window,
        "start": bt_score.index[0],
        "end": bt_score.index[-1],
        "top_n": top_n,
        "cost_bps": cost_bps,
        "ic": df_records(ic_df),
        "quantile": df_records(quant_df),
        "rotation": {d: rotation_payload(r) for d, r in rotations.items()},
    })

    print("[6/6] 导出矩阵 / 因子快照 / 健康度序列 ...")
    scores_out = smoothed.copy()
    scores_out.index.name = "date"
    scores_out.reset_index().to_parquet(OUT_DIR / "scores.parquet", index=False)
    print(f"    -> scores.parquet {scores_out.shape}")

    pct_out = pct.copy()
    pct_out.index.name = "date"
    pct_out.reset_index().to_parquet(OUT_DIR / "pct.parquet", index=False)
    print(f"    -> pct.parquet {pct_out.shape}")

    kline = data["board_kline"].copy()
    kline["date"] = pd.to_datetime(kline["date"])
    kline.to_parquet(OUT_DIR / "board_kline.parquet", index=False)
    print(f"    -> board_kline.parquet {kline.shape}")

    # 最新日各因子原始值 + 全市场分位（横截面 rank pct）
    snap = {}
    for fname, mat in factors.items():
        if mat is None or mat.empty:
            continue
        row = mat.iloc[-1]
        rank = row.rank(pct=True)
        for c in mat.columns:
            v = row.get(c)
            if pd.isna(v):
                continue
            snap.setdefault(c, {})[fname] = {
                "value": float(v),
                "rank": float(rank[c]) if pd.notna(rank.get(c)) else None,
            }
    dump_json("factor_snapshot.json", {
        "date": factors["breadth"].index[-1],
        "boards": snap,
    })

    # 滚动健康度序列：每日 IC 的 HEALTH_DAYS 日滚动均值
    ic_daily = daily_ic_series(smoothed, board_close, HEALTH_HORIZON)
    health_roll = ic_daily.rolling(HEALTH_DAYS, min_periods=HEALTH_DAYS // 2).mean()
    hs = [{"date": d.strftime("%Y-%m-%d"), "ic": float(v)}
          for d, v in health_roll.dropna().tail(HEALTH_SERIES_DAYS).items()]
    dump_json("health_series.json", {"window": HEALTH_DAYS,
                                     "horizon": HEALTH_HORIZON, "series": hs})

    dump_json("meta.json", {
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_sec": round(time.time() - t0, 1),
        "board_source": cfg.get("board_source", "eastmoney"),
        "score_start": smoothed.index[0],
        "score_end": smoothed.index[-1],
        "n_boards": int(smoothed.shape[1]),
        "backtest_window": window,
        "score_lookback": cfg.get("score_lookback", 120),
    })

    print(f"\n完成，耗时 {time.time() - t0:.0f} 秒。输出目录：{OUT_DIR}")


if __name__ == "__main__":
    main()
