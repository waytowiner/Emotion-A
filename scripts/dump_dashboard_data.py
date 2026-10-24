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
from emotion.data import quotes
from emotion.factors import compute_all_factors
from emotion.project_status import project_status

OUT_DIR = ROOT / "data" / "dashboard"
BACKTEST_WINDOW_FALLBACK = 180
HEALTH_DAYS = 60          # 健康度滚动窗口（与 recent_signal_health 默认一致）
HEALTH_SERIES_DAYS = 240  # 预计算健康度序列的长度上限
HEALTH_HORIZON = 5
POLICY_TOP_N = 15         # 政策榜导出条数


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
    # 等权基准净值与方向/持仓数/频率无关，已提升到窗口层 benchmark_nav，不重复存
    return {
        "nav": nav_records(rot["nav"]),
        "strategy": rot["strategy"],
        "benchmark": rot["benchmark"],
        "total_cost_drag": rot["total_cost_drag"],
        "total_turnover": rot.get("total_turnover"),
        "final_nav": rot["final_nav"],
        "final_benchmark": rot["final_benchmark"],
    }


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


def load_etf_info(board_names: dict) -> dict[str, list[dict] | None]:
    """生成 board_code -> ETF 列表（或 None），ETF 条目附带最新行情。

    两个映射来源：
    - board_source=etf：data/cache/etf_trading_map.parquet（指数→规模最大 ETF，
      build_etf_trading_map.py 生成），board_code 即指数代码，直接匹配；
    - 其他口径：config/etf_map.yaml（板块→ETF 模糊映射）。
    任一输入缺失时返回空 dict（看板降级为不显示 ETF 列）。
    """
    quotes = {}
    quote_path = ROOT / "data" / "cache" / "etf_list.parquet"
    if quote_path.exists():
        q = pd.read_parquet(quote_path)
        q["code"] = q["code"].astype(str).str.zfill(6)
        quotes = {r.code: r for r in q.itertuples()}

    def _enrich(ec: str, name: str) -> dict:
        item = {"code": str(ec).zfill(6), "name": name}
        r = quotes.get(str(ec).zfill(6))
        if r is not None:
            item.update({
                "price": None if pd.isna(r.price) else float(r.price),
                "pct_chg": None if pd.isna(r.pct_chg) else float(r.pct_chg),
                "amount": None if pd.isna(r.amount) else float(r.amount),
            })
        return item

    tmap_path = ROOT / "data" / "cache" / "etf_trading_map.parquet"
    if tmap_path.exists():
        tm = pd.read_parquet(tmap_path)
        tm["board_code"] = tm["board_code"].astype(str)
        if len(set(tm["board_code"]) & {str(c) for c in board_names}) > 0:
            out: dict[str, list[dict] | None] = {str(c): None for c in board_names}
            for _, r in tm.iterrows():
                if r["board_code"] in out:
                    out[r["board_code"]] = [_enrich(r["etf_code"], r["etf_name"])]
            n = sum(1 for v in out.values() if v)
            print(f"    [etf] 交易映射：{n}/{len(out)} 个板块挂接 ETF（规模最大者）")
            return out

    map_path = ROOT / "config" / "etf_map.yaml"
    if not map_path.exists():
        print("    [etf] config/etf_map.yaml 不存在，跳过 ETF 信息")
        return {}
    import yaml
    with open(map_path, encoding="utf-8") as f:
        mapping = yaml.safe_load(f)
    out = {str(code): None for code in board_names}
    for code, entry in mapping.items():
        code = str(code)  # yaml 会把纯数字代码（如同花顺 881273）解析成 int
        if code not in out:
            continue  # 映射文件可能来自另一板块源，不能污染当前看板
        etfs = entry.get("etf") if isinstance(entry, dict) else None
        if not etfs:
            out[code] = None
            continue
        items = [_enrich(e["code"], e["name"]) for e in etfs]
        out[code] = items
    n = sum(1 for v in out.values() if v)
    print(f"    [etf] {n}/{len(out)} 个入选板块挂接 ETF")
    return out


def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = pipeline.load_config()

    print("[1/6] 准备数据（读取本地缓存）...")
    data = pipeline.prepare(cfg, offline=True, end_override=pipeline.cache_end_date(cfg))
    board_names: dict = data["board_names"]

    print("[2/6] 计算情绪因子 ...")
    factors = compute_all_factors(data["stock_daily"], data["board_map"],
                                  data["board_kline"],
                                  market_total=data.get("market_total"))
    flow_proxy = signal.compute_board_flow_proxy(data["stock_daily"], data["board_map"])

    print("[3/6] 合成情绪分 ...")
    _, smoothed, pct = score.build_scores(
        factors, cfg["weights"], cfg.get("smooth_window", 5),
        lookback=cfg.get("score_lookback", 120),
        z_clip=cfg.get("score_z_clip", 3.0))

    # 政策（时政）支柱：与情绪主线完全平行；无缓存/无词典时优雅降级为 None
    print("[3.5/6] 政策（时政）支柱 ...")
    from main import _policy_payload
    policy = _policy_payload(cfg, smoothed, pct, data)
    if policy is not None:
        print(f"    [policy] 联播 {policy['news_days']} 天（至 {policy['news_end']}），"
              f"词典 {policy['n_keyword_boards']} 板块；"
              f"共振 {len(policy['signals']['policy_resonance'])} / "
              f"潜伏 {len(policy['signals']['policy_lurking'])} 个")

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
        cold_quantile=sig_cfg.get("cold_quantile", 0.1),
        factors=factors, flow_proxy=flow_proxy,
        breakthrough_cfg=sig_cfg.get("breakthrough"))

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

    # 风格判定（regime 自适应）：滚动已实现 IC + 回撤护栏 -> 方向建议
    regime_cfg = cfg.get("regime", {})
    regime = backtest.regime_snapshot(smoothed, board_close, regime_cfg,
                                      top_n=sig_cfg.get("top_n", 3))

    signals_named = {}
    for key in ["heating", "cooling", "hot", "cold", "bottom_reversal",
                "extreme_hot", "extreme_cold", "fund_emotion_divergence"]:
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
        "project_status": project_status(),
        "board_source": cfg.get("board_source", "eastmoney"),
        "n_boards": int(smoothed.shape[1]),
        "universe": data.get("universe"),
        "health": health,
        "regime": regime,
        "signals": signals_named,
        "top": ranked[:15],
        "bottom": ranked[-15:][::-1],
        "ranking": ranked,
        "etf_map": etf_map,
        "leaders": leaders_map,
    })

    print("[5/6] 回测（多窗口）...")
    windows = cfg.get("backtest_windows")
    if not windows:
        windows = [int(cfg.get("backtest_window") or BACKTEST_WINDOW_FALLBACK)]
    windows = [int(w) for w in windows]
    bt_cfg = cfg["backtest"]
    top_n = sig_cfg.get("top_n", 3)
    top_n_list = sorted({1, 3, 5, 10, top_n})
    cost_bps = bt_cfg.get("cost_bps", 5)

    # 大盘参考：沪深300 收盘价（baostock；失败则降级为净值图不显示指数线）
    index_close = None
    try:
        idx_k = quotes.get_index_kline(
            ROOT / cfg["paths"]["cache_dir"], "sh.000300",
            str(board_close.index[0])[:10], str(board_close.index[-1])[:10])
        if not idx_k.empty:
            index_close = idx_k.set_index("date")["close"].sort_index()
            print(f"    [index] 沪深300 {len(idx_k)} 行，"
                  f"最新 {str(idx_k['date'].max())[:10]}")
    except Exception as exc:  # noqa: BLE001
        print(f"    [index] 沪深300 获取失败：{exc}（净值图降级为无指数参考线）")

    def run_window(w: int) -> dict:
        bt_score = smoothed.tail(w)
        bt_close = board_close.loc[bt_score.index[0]:]
        ic_df = backtest.ic_analysis(bt_score, bt_close, bt_cfg["forward_days"])
        quant_df = backtest.quantile_test(bt_score, bt_close,
                                          horizon=bt_cfg["forward_days"][0])
        # 频率 × 持仓数 × 方向 全组合（周频是默认口径；日频对照换手成本的代价）。
        # 等权基准净值与频率/持仓数/方向都无关，只在窗口层存一份。
        rotation = {}
        benchmark_nav = None
        for freq_key, freq in [("W", "W-FRI"), ("D", "D")]:
            by_n = {}
            for n in top_n_list:
                by_n[str(n)] = {}
                for d in ["top", "bottom", "avoid_top"]:
                    r = backtest.rotation_backtest(
                        bt_score, bt_close, top_n=n, cost_bps=cost_bps,
                        freq=freq, direction=d)
                    if benchmark_nav is None:
                        benchmark_nav = nav_records(r["benchmark_nav"])
                    by_n[str(n)][d] = rotation_payload(r)
            rotation[freq_key] = by_n
        payload = {
            "window": w,
            "start": bt_score.index[0],
            "end": bt_score.index[-1],
            "top_n": top_n,
            "top_n_list": top_n_list,
            "cost_bps": cost_bps,
            "ic": df_records(ic_df),
            "quantile": df_records(quant_df),
            "benchmark_nav": benchmark_nav,
            "rotation": rotation,
        }
        # 沪深300 参考净值：与窗口交易日对齐、归一到期初为 1
        if index_close is not None:
            s = index_close.reindex(bt_close.index).ffill().dropna()
            if len(s) > 1:
                payload["index300_nav"] = nav_records(s / s.iloc[0])
        return payload

    bt_by_window = {str(w): run_window(w) for w in windows}
    # 顶层保留首个窗口的字段（兼容旧前端），windows 携带全部档位
    dump_json("backtest.json", {
        **bt_by_window[str(windows[0])],
        "window_list": windows,
        "windows": bt_by_window,
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
    ic_daily = backtest.daily_ic_series(smoothed, board_close, HEALTH_HORIZON)
    health_roll = ic_daily.rolling(HEALTH_DAYS, min_periods=HEALTH_DAYS // 2).mean()
    hs = [{"date": d.strftime("%Y-%m-%d"), "ic": float(v)}
          for d, v in health_roll.dropna().tail(HEALTH_SERIES_DAYS).items()]
    dump_json("health_series.json", {"window": HEALTH_DAYS,
                                     "horizon": HEALTH_HORIZON, "series": hs})

    # 政策（时政）支柱导出：policy.json（榜+双因矩阵散点数据）
    # + policy_scores.parquet（date×board 政策平滑分，与 scores.parquet 平行）。
    # 无政策数据时写一份空结构，前端据此隐藏双因区。
    if policy is not None:
        from emotion.policy import DUAL_STATE_CN
        p_smoothed = policy["smoothed"]
        p_pct = policy["pct"]
        # 数据日期以情绪侧末日为准（政策分末段可能因缓存缺口比情绪侧短）
        last_date_p = smoothed.index[-1]
        # 政策分自身的最后有效行（缓存缺口时早于情绪末日，读数如实标注滞后）
        p_valid = p_smoothed.dropna(how="all")
        last_pscore = (p_valid.iloc[-1].dropna().sort_values(ascending=False)
                      if not p_valid.empty else pd.Series(dtype=float))
        last_ppct = (p_pct.loc[p_valid.index[-1]] if not p_valid.empty
                     and not p_pct.empty else pd.Series(dtype=float))
        last_epct = pct.iloc[-1]
        last_intensity = policy.get("intensity")
        # 双因矩阵散点：全部有读数的板块 (情绪分位, 政策分位)
        scatter = []
        for c in last_pscore.index:
            ep = last_epct.get(c)
            pp = last_ppct.get(c)
            if pd.isna(ep) or pd.isna(pp):
                continue
            hits = (int(last_intensity[c]) if last_intensity is not None
                    and pd.notna(last_intensity.get(c)) else None)
            scatter.append({
                "code": c, "name": board_names.get(c, c),
                "emotion_pct": float(ep), "policy_pct": float(pp),
                "policy_score": float(last_pscore[c]),
                "hits": hits,
                "etf": etf_map.get(c),
                "state": (policy["states"].get(c) if policy["states"] is not None
                          else "neutral"),
            })
        sig = policy["signals"]
        policy_payload = {
            "available": True,
            "date": last_date_p,
            "news_days": policy["news_days"],
            "news_end": policy["news_end"],
            "n_keyword_boards": policy["n_keyword_boards"],
            "state_cn": DUAL_STATE_CN,
            "signals": {k: [{"code": c, "name": board_names.get(c, c),
                             "etf": etf_map.get(c)} for c in v]
                        for k, v in sig.items()},
            "top": [{
                "code": c, "name": board_names.get(c, c),
                "score": float(last_pscore[c]),
                "pct": float(last_ppct[c]) if pd.notna(last_ppct.get(c)) else None,
                "hits": (int(last_intensity[c]) if last_intensity is not None
                         and pd.notna(last_intensity.get(c)) else None),
                "etf": etf_map.get(c),
            } for c in last_pscore.head(POLICY_TOP_N).index],
            "scatter": scatter,
        }
        p_out = p_smoothed.copy()
        p_out.index.name = "date"
        p_out.reset_index().to_parquet(OUT_DIR / "policy_scores.parquet", index=False)
        print(f"    -> policy_scores.parquet {p_out.shape}")
    else:
        policy_payload = {"available": False,
                         "state_cn": {}, "signals": {}, "top": [], "scatter": []}
    dump_json("policy.json", policy_payload)

    dump_json("meta.json", {
        "calendar_audit": data["calendar_audit"],
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_sec": round(time.time() - t0, 1),
        "board_source": cfg.get("board_source", "eastmoney"),
        "score_start": smoothed.index[0],
        "score_end": smoothed.index[-1],
        "n_boards": int(smoothed.shape[1]),
        "universe": data.get("universe"),
        "backtest_window": windows[0],
        "backtest_windows": windows,
        "score_lookback": cfg.get("score_lookback", 120),
    })

    print(f"\n完成，耗时 {time.time() - t0:.0f} 秒。输出目录：{OUT_DIR}")


if __name__ == "__main__":
    main()
