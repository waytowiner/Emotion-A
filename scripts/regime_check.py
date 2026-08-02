"""分段验证：情绪分在不同市场 regime 下的有效性是否一致。

切分点：2024-09-24（政策转向，成交额/市场结构剧变）。
输出三段（9·24前 / 9·24后 / 最近180交易日）的 IC 与分层统计。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from emotion import backtest, pipeline, score
from emotion.factors import compute_all_factors

REGIME_CUT = pd.Timestamp("2024-09-24")
LOOKBACK_DAYS = 180


def summarize(label: str, smoothed: pd.DataFrame, board_close: pd.DataFrame) -> None:
    if len(smoothed) < 30:
        print(f"\n### {label}: 样本不足（{len(smoothed)} 天），跳过")
        return
    ic = backtest.ic_analysis(smoothed, board_close, [5, 10, 20])
    q = backtest.quantile_test(smoothed, board_close, horizon=5)
    print(f"\n### {label}（{len(smoothed)} 个交易日）")
    for _, r in ic.iterrows():
        print(f"  IC{int(r['horizon'])}日: {r['IC_mean']:+.4f}  "
              f"IR={r['IR']:+.2f}  胜率={r['IC_win_rate']:.0%}")
    if not q.empty:
        qs = "  ".join(f"Q{int(g) + 1}={r['mean_fwd_ret']:+.2%}" for g, r in q.iterrows())
        print(f"  分层5日收益: {qs}")


def main() -> None:
    cfg = pipeline.load_config()
    data = pipeline.prepare(cfg)
    factors = compute_all_factors(data["stock_daily"], data["board_map"],
                                  data["board_kline"],
                                  market_total=data.get("market_total"))
    _, smoothed, _ = score.build_scores(factors, cfg["weights"],
                                        cfg.get("smooth_window", 5))
    board_close = data["board_kline"].pivot(
        index="date", columns="board_code", values="close").sort_index()

    pre = smoothed[smoothed.index < REGIME_CUT]
    post = smoothed[smoothed.index >= REGIME_CUT]
    recent = smoothed.tail(LOOKBACK_DAYS)

    # 标签取实际数据范围（history.start_date 过滤后，9·24 前只剩缓冲窗口）
    pre_label = (f"9·24 之前（{pre.index.min():%Y-%m-%d} ~ "
                 f"{pre.index.max():%Y-%m-%d}）" if len(pre) else "9·24 之前（无数据）")
    post_label = (f"9·24 之后（{post.index.min():%Y-%m-%d} ~ "
                  f"{post.index.max():%Y-%m-%d}）" if len(post) else "9·24 之后（无数据）")

    summarize(pre_label, pre, board_close)
    summarize(post_label, post, board_close)
    summarize(f"最近 {LOOKBACK_DAYS} 个交易日", recent, board_close)


if __name__ == "__main__":
    main()
