"""同窗口对比 THS / 东财两个板块口径的回测结果。

用法：python scripts/compare_board_sources.py [--window 180]

输出：两个口径的 IC、Top3 轮动年化、夏普，并给出结论建议。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from emotion import backtest, pipeline, score  # noqa: E402
from emotion.factors import compute_all_factors  # noqa: E402


def run_one(cfg: dict, source: str, window: int,
            end_override: str | None = None) -> dict:
    """用指定 board_source 跑一遍完整回测，返回关键指标。

    end_override：板块K线/个股统一按该日期截断，用于避免触发
    增量网络请求（curl_cffi 传输层在部分环境下会进程级崩溃）。
    """
    cfg = dict(cfg)
    cfg["board_source"] = source
    print(f"\n{'='*60}\n口径：{source} | 窗口：{window} 交易日\n{'='*60}")

    data = pipeline.prepare(cfg, end_override=end_override)
    factors = compute_all_factors(
        data["stock_daily"], data["board_map"], data["board_kline"],
        market_total=data.get("market_total"))
    _, smoothed, _ = score.build_scores(
        factors, cfg["weights"], cfg.get("smooth_window", 5),
        lookback=cfg.get("score_lookback", 250))
    smoothed = smoothed.tail(window)

    board_close = data["board_kline"].pivot(
        index="date", columns="board_code", values="close").sort_index()
    board_close = board_close.loc[smoothed.index[0]:]

    bt_cfg = cfg["backtest"]
    ic_df = backtest.ic_analysis(smoothed, board_close, bt_cfg["forward_days"])
    rot = backtest.rotation_backtest(
        smoothed, board_close,
        top_n=cfg["signal"].get("top_n", 3),
        cost_bps=bt_cfg.get("cost_bps", 5),
        direction="top")

    # 提取关键数字
    ic_row = ic_df[ic_df["horizon"] == bt_cfg["forward_days"][0]].iloc[0]
    final_nav = rot["final_nav"]
    strat = rot["strategy"]

    return {
        "source": source,
        "n_boards": data["board_map"]["board_code"].nunique(),
        "ic": float(ic_row["IC_mean"]),
        "ic_ir": float(ic_row["IR"]),
        "ic_win_rate": float(ic_row["IC_win_rate"]),
        "ann_return": float(strat["annual_ret"]),
        "sharpe": float(strat["sharpe"]),
        "max_dd": float(strat["max_dd"]),
        "final_nav": float(final_nav),
        "start": str(smoothed.index[0].date()),
        "end": str(smoothed.index[-1].date()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=180, help="回测交易日数")
    args = ap.parse_args()

    cfg = pipeline.load_config()
    # 板块/个股缓存截止日：避免触发 curl_cffi 增量网络请求导致进程崩溃
    import glob
    from emotion.data import cache as _cache_mod
    cache_dir = ROOT / cfg["paths"]["cache_dir"]
    last_dates = []
    for f in glob.glob(str(cache_dir / "board_kline_*.parquet"))[:200]:
        try:
            last_dates.append(pd.read_parquet(f)["date"].max())
        except Exception:
            continue
    end_override = None
    if last_dates:
        end_override = max(last_dates).strftime("%Y-%m-%d")
        print(f"[info] 用缓存截止日 {end_override} 运行（避免增量网络抓取）")

    results = []
    for source in ("ths", "eastmoney"):
        try:
            results.append(run_one(cfg, source, args.window, end_override))
        except Exception as exc:
            print(f"[error] {source} 口径失败：{exc}")
            results.append({"source": source, "error": str(exc)})

    print(f"\n{'='*60}\n对比结果（窗口 {args.window} 交易日）\n{'='*60}")
    df = pd.DataFrame(results)
    print(df.to_string(index=False))

    valid = [r for r in results if "error" not in r]
    if len(valid) == 2:
        ths, em = valid[0], valid[1]
        print("\n结论：")
        if em["ic"] >= 0.03 and ths["ic"] < 0.03:
            print("→ 东财口径有效，THS 口径无效。建议切回东财口径使用。")
        elif ths["ic"] >= 0.03 and em["ic"] < 0.03:
            print("→ THS 口径反而更好，继续使用 THS。")
        elif em["ic"] >= 0.03 and ths["ic"] >= 0.03:
            print("→ 两个口径都有效，优先用 IC 更高的。")
        else:
            print("→ 两个口径 IC 都低于 0.03，策略 alpha 可能已衰减。")
            print("  建议降级为情绪观察工具，不再用于交易决策。")

    out = ROOT / "report" / f"board_source_compare_{pd.Timestamp.today():%Y%m%d}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n结果已保存：{out}")


if __name__ == "__main__":
    main()
