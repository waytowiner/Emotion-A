"""价格位置因子诊断（只描述，不进生产、不调权）。

动机：现有 6 个因子同为追动量方向（2026-07 风格翻转后 IC 全部转负）。
价格位置（距 N 日高点的距离）与 RPS 动量天然对冲——动量期高分组占优、
反转期低位置组占优，是高低切换行情里经典的正交方向。本脚本只做诊断：
IC / 分层 / 与现有因子的横截面相关性，回答"值不值得进入下一轮预注册"，
不据此改任何生产参数（据近期 IC 取反 = 拟合最近行情，不做）。

因子定义（值越大 = 越贴近高点 = 位置越高；≤ 0）：
- dist_high_120：close / 近 120 日最高收盘 - 1（min_periods=60）
- dist_high_250：close / 近 250 日最高收盘 - 1（min_periods=120；
  样本仅约 270 交易日，该因子前约 120 日为空，有效样本偏后段）

无前视：全部为当日收盘可得量；IC 用前瞻收益，口径同 emotion.backtest。

运行：python scripts/diag_price_position.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from emotion import backtest, pipeline
from emotion.backtest import daily_ic_series, ic_robustness, quantile_test
from emotion.factors import FACTOR_NAMES, compute_all_factors

WINDOWS = (120, 60, 30)          # 尾部窗口（全期另算）
HORIZONS = (5, 10, 20)           # 前瞻窗口
CORR_MIN_OBS = 30                # 横截面相关至少需要的有效天数


def dist_from_high(board_close: pd.DataFrame, window: int,
                   min_periods: int) -> pd.DataFrame:
    """距 window 日高点的距离：close / rolling_max(close) - 1（≤0，越大约高）。"""
    hi = board_close.rolling(window, min_periods=min_periods).max()
    return board_close / hi - 1.0


def window_row(fac: pd.DataFrame, board_close: pd.DataFrame,
               horizon: int) -> dict:
    """单因子单 horizon：全期 + 尾部窗口 IC/胜率，最短窗口附 HAC/bootstrap。"""
    ic = daily_ic_series(fac, board_close, horizon)
    rec = {"n": int(len(ic)), "ic_all": float(ic.mean()),
           "win_all": float((ic > 0).mean())}
    for w in WINDOWS:
        sub = ic.tail(w)
        rec[f"ic_{w}"] = float(sub.mean()) if len(sub) else np.nan
        rec[f"win_{w}"] = float((sub > 0).mean()) if len(sub) else np.nan
        rec[f"n_{w}"] = int(len(sub))
    rob = ic_robustness(ic.tail(min(WINDOWS)), horizon=horizon,
                        bootstrap_samples=500)
    rec["hac_t"] = rob["IC_HAC_t"]
    rec["ci_low"] = rob["IC_bootstrap_low"]
    rec["ci_high"] = rob["IC_bootstrap_high"]
    return rec


def xsection_corr_avg(a: pd.DataFrame, b: pd.DataFrame) -> tuple[float, int]:
    """两因子逐日横截面 Spearman 相关的日均值（排名口径，无需先 z 化）。"""
    days = a.index.intersection(b.index)
    vals = []
    for dt in days:
        pair = pd.concat([a.loc[dt], b.loc[dt]], axis=1, keys=["x", "y"]).dropna()
        if len(pair) < 20:
            continue
        c = pair["x"].corr(pair["y"], method="spearman")
        if pd.notna(c):
            vals.append(c)
    if len(vals) < CORR_MIN_OBS:
        return float("nan"), len(vals)
    return float(np.mean(vals)), len(vals)


def main() -> None:
    cfg = pipeline.load_config()
    print("[1/3] 准备数据（缓存口径，无网络）...")
    data = pipeline.prepare(cfg, end_override=pipeline.cache_end_date(cfg))
    board_close = data["board_kline"].pivot(
        index="date", columns="board_code", values="close").sort_index()

    new_factors = {
        "dist_high_120": dist_from_high(board_close, 120, 60),
        "dist_high_250": dist_from_high(board_close, 250, 120),
    }

    print("[2/3] IC / 分层 ...")
    ic_tables = {name: {h: window_row(f, board_close, h)
                        for h in HORIZONS}
                 for name, f in new_factors.items()}
    quant = {name: {f"全期/{h}日": quantile_test(f, board_close, horizon=h)
                    for h in (5, 10)}
             for name, f in new_factors.items()}
    for name, f in new_factors.items():
        tail_days = f.dropna(how="all").index[-60:]
        f60 = f.loc[f.index.isin(tail_days)]
        for h in (5, 10):
            quant[name][f"近60日/{h}日"] = quantile_test(f60, board_close,
                                                         horizon=h)

    print("[3/3] 与现有因子的横截面相关 ...")
    factors = compute_all_factors(data["stock_daily"], data["board_map"],
                                  data["board_kline"],
                                  market_total=data.get("market_total"))
    corr = {}
    for name, f in new_factors.items():
        corr[name] = {}
        for old in FACTOR_NAMES:
            c, n = xsection_corr_avg(f, factors[old])
            corr[name][old] = (c, n)

    # ---------- 输出 ----------
    d0, d1 = board_close.index[0], board_close.index[-1]
    lines = ["# 价格位置因子诊断（只描述，不进生产）", "",
             f"样本：{d0:%Y-%m-%d} ~ {d1:%Y-%m-%d}（{len(board_close)} 个交易日），"
             "当前成分快照。因子定义与动机见脚本 docstring。"
             "本报告不据此改任何生产参数；即便结论积极，也须先写入 config.yaml "
             "预注册、再用后续未参与时段复验。", "",
             "因子方向约定：值越大 = 越贴近 N 日高点（位置越高）。"
             "动量期预期 IC>0；反转期（高分组跑输）预期 IC<0。", ""]

    for name in new_factors:
        lines += [f"## {name}", "",
                  "### IC 汇总（5 日前瞻；最短窗口 30 日附 HAC/bootstrap）", "",
                  "| 窗口 | IC 均值 | IC 胜率 | 样本 |",
                  "|---|---|---|---|"]
        r5 = ic_tables[name][5]
        for label, k in (("全期", "all"), ("近120日", "120"),
                         ("近60日", "60"), ("近30日", "30")):
            ic_v = r5[f"ic_{k}"] if k != "all" else r5["ic_all"]
            win_v = r5[f"win_{k}"] if k != "all" else r5["win_all"]
            n_v = r5["n"] if k == "all" else r5[f"n_{k}"]
            lines.append(f"| {label} | {ic_v:+.4f} | {win_v:.0%} | {n_v} |")
        lines.append(f"| 近30日 HAC t / Bootstrap 95% CI | "
                     f"{r5['hac_t']:+.2f} | [{r5['ci_low']:+.3f}, "
                     f"{r5['ci_high']:+.3f}] | |")
        lines += ["", "### 分前瞻窗口 IC", "",
                  "| 前瞻 | 全期 IC | 全期胜率 | 近60日 IC | 近60日胜率 |",
                  "|---|---|---|---|---|"]
        for h in HORIZONS:
            r = ic_tables[name][h]
            lines.append(f"| {h} 日 | {r['ic_all']:+.4f} | {r['win_all']:.0%} "
                         f"| {r['ic_60']:+.4f} | {r['win_60']:.0%} |")
        lines += ["", "### 分层测试（平均前瞻收益，Q1=位置最低组）", "",
                  "| 口径 | Q1 | Q2 | Q3 | Q4 | Q5 |", "|---|---|---|---|---|---|"]
        for qname, qdf in quant[name].items():
            if qdf.empty:
                continue
            cells = [f"{qdf.loc[g, 'mean_fwd_ret']:+.3%}" if g in qdf.index
                     else "—" for g in range(5)]
            lines.append(f"| {qname} | " + " | ".join(cells) + " |")
        lines.append("")

    lines += ["## 与现有 6 因子的横截面相关性（逐日 Spearman 的日均）", "",
              "| 新因子 | " + " | ".join(FACTOR_NAMES) + " |",
              "|---|" + "---|" * len(FACTOR_NAMES)]
    for name in new_factors:
        cells = []
        for old in FACTOR_NAMES:
            c, n = corr[name][old]
            cells.append(f"{c:+.2f}" if pd.notna(c) else f"—(n={n})")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["",
              "> 与 rps 的相关预计偏高（同为价格动量族）；若 |相关| 接近 1，"
              "说明位置因子没有增量信息，不值得单独引入。", ""]

    lines += ["## 判读参考（描述性，不构成上线依据）", "",
              "值得进入下一轮预注册的最低门槛（三条同时满足）：",
              "① 全期 IC 与近 60 日 IC 同号（方向是结构性而非单纯风格）；"
              "② 近 30 日 Bootstrap 95% CI 排除 0（当前环境方向可信）；"
              "③ 与 rps 的横截面相关 |ρ| < 0.8（有增量信息）。",
              "对照结果请见上表；若不满足，记录为否定假设，避免重复试错。", "",
              "## 限制", "",
              "1. 样本约 270 个交易日（一个风格周期），dist_high_250 前约 120 日"
              "为空，有效样本集中在后段，结论对区间敏感。",
              "2. IC 前瞻收益彼此重叠，显著性只看 HAC t 与移动区块 bootstrap，"
              "不看朴素 IR。",
              "3. 成分股使用当前快照，存在幸存者偏差；本诊断不做任何交易成本分析——"
              "位置因子变化慢、换手低，这是它的潜在优点，但需在轮动回测中验证。", ""]
    md = "\n".join(lines)
    print(md)

    out_dir = ROOT / cfg["paths"]["report_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"price_position_factor_{pd.Timestamp.today():%Y%m%d}.md"
    path.write_text(md, encoding="utf-8")
    print(f"\n已保存：{path}")


if __name__ == "__main__":
    main()
