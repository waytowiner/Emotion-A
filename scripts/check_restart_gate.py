"""研究重启门槛检查（观察模式，只读不改任何参数）

用途：把 `report/prospective_log.md` 表头预注册的"研究重启门槛"变成可执行的
客观判据，避免行情转折时临场判断。

门槛（2026-10-22 预注册）：
  A. 滚动 IC 回到零轴上方   —— 近 40 个交易日的日度 IC 均值 > 0
  B. 连续两个月胜率站上 50%  —— 近 40 个交易日的日度 IC 胜率 > 50%
  两项同时满足才可讨论重启研究路线。

为什么用 40 个交易日：约等于两个自然月（A股月均约 21 个交易日）。

数据来源：`data/dashboard/health_series.json` 的 `series`（由
`scripts/dump_dashboard_data.py` 导出），逐日截面 IC、60 日窗口、5 日前瞻收益。

滞后说明：日度 IC 需要 h=5 个交易日的前瞻收益，因此序列末尾天然滞后 5 个
交易日。脚本会打印 `ic_last_date` 与 `data_end`，避免把"最新值"误读成最新数据日。
以 10-21 数据为例，IC 序列止于 10-14。

用法：python scripts/check_restart_gate.py [--days 40] [--json]
输出：report/restart_gate.json（每次覆盖，记录历史判定时间）
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SERIES = ROOT / "data" / "dashboard" / "health_series.json"
META = ROOT / "data" / "dashboard" / "meta.json"
OUT = ROOT / "report" / "restart_gate.json"

# 门槛阈值（预注册，勿随手改）
IC_MEAN_GT = 0.0        # A：近 N 日 IC 均值须 > 0
WIN_RATE_GT = 0.50      # B：近 N 日 IC 胜率须 > 50%
DEFAULT_DAYS = 40       # 约两个自然月


def load_series() -> pd.DataFrame:
    if not SERIES.exists():
        raise SystemExit(f"缺少 {SERIES}，先跑 scripts/dump_dashboard_data.py")
    h = json.loads(SERIES.read_text(encoding="utf-8"))
    s = pd.DataFrame(h["series"])
    s["date"] = pd.to_datetime(s["date"])
    return s.sort_values("date").reset_index(drop=True)


def evaluate(s: pd.DataFrame, days: int) -> dict:
    seg = s.tail(days)
    ic = seg["ic"]
    ic_mean = float(ic.mean())
    win_rate = float((ic > 0).mean())
    gate_a = ic_mean > IC_MEAN_GT
    gate_b = win_rate > WIN_RATE_GT

    # 最近一次连续满足两项的起点（用于观察"是否已连续两个月达标"）
    both = (seg["ic"] > 0).astype(int)
    streak = 0
    for v in reversed(both.tolist()):
        if v == 1:
            streak += 1
        else:
            break

    data_end = None
    if META.exists():
        meta = json.loads(META.read_text(encoding="utf-8"))
        data_end = meta.get("score_end")

    return {
        "window_days": days,
        "ic_mean": ic_mean,
        "ic_win_rate": win_rate,
        "gate_a_ic_mean_above_zero": gate_a,
        "gate_b_win_rate_above_50pct": gate_b,
        "restart_allowed": bool(gate_a and gate_b),
        "consecutive_positive_ic_days": streak,
        "ic_last_date": str(s["date"].iloc[-1].date()),
        "data_end": data_end,
        "ic_series_lag_trading_days": (
            None if data_end is None else
            int(len(pd.bdate_range(s['date'].iloc[-1], data_end)) - 1)
        ),
        "n_obs": int(len(seg)),
        "ic_mean_full": float(s["ic"].mean()),
        "ic_win_rate_full": float((s["ic"] > 0).mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="研究重启门槛检查（观察模式）")
    ap.add_argument("--days", type=int, default=DEFAULT_DAYS,
                    help=f"观察窗口交易日数（默认 {DEFAULT_DAYS}，约两个自然月）")
    ap.add_argument("--json", action="store_true", help="只输出 JSON")
    args = ap.parse_args()

    s = load_series()
    r = evaluate(s, args.days)

    if args.json:
        print(json.dumps(r, ensure_ascii=False, indent=2))
    else:
        verdict = "可讨论重启" if r["restart_allowed"] else "维持观察"
        print(f"研究重启门槛检查（近 {r['window_days']} 个交易日）")
        print(f"  判据A  IC 均值 > 0        : {r['ic_mean']:+.4f}  -> "
              f"{'通过' if r['gate_a_ic_mean_above_zero'] else '未通过'}")
        print(f"  判据B  IC 胜率 > 50%     : {r['ic_win_rate']:.0%}  -> "
              f"{'通过' if r['gate_b_win_rate_above_50pct'] else '未通过'}")
        print(f"  连续为正天数             : {r['consecutive_positive_ic_days']}")
        print(f"  IC 序列止于              : {r['ic_last_date']}"
              f"（数据截至 {r['data_end']}，前瞻收益导致天然滞后）")
        print(f"  全期参考                 : IC均值 {r['ic_mean_full']:+.4f} / "
              f"胜率 {r['ic_win_rate_full']:.0%}")
        print(f"  结论                     : {verdict}")
        print("  提示：本脚本只读，不修改 config.yaml 或任何参数。")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.json:
        print(f"\n已写入：{OUT}")


if __name__ == "__main__":
    main()
