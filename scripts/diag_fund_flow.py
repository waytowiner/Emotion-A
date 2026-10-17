"""主力净流入因子诊断（只描述，不进生产、不调权）。

动机：现有 6 因子全部为价格/情绪衍生（非对角平均相关 0.23，风格翻转时一起
失效，见 factor_adaptive_20260913）；成交额×涨跌幅的"资金代理"与价格循环，
资金-情绪背离规则因此失效（overview.md）。主力净流入是唯一未证伪的正交
信息源。本脚本只做诊断：IC / 分层 / 与现有因子的横截面相关性，回答
"值不值得进入下一轮预注册"，不据此改任何生产参数。

因子定义（%）：
- fflow_1d：当日 Σ成分股主力净流入 / Σ成分股成交额 × 100（东财 fflow 日频）
- fflow_5d：上述分子分母的 5 日滚动和之比（降噪）

数据限制：东财 fflow/daykline 接口单次返回上限 120 条，样本仅约最近 6 个月；
缓存随每日抓取向前累积（scripts/fetch_fund_flow.py 已按增量设计），
本报告结论对区间高度敏感，只配作假设生成，不配作上线依据。

运行：python scripts/diag_fund_flow.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from emotion import pipeline
from emotion.backtest import daily_ic_series, ic_robustness, quantile_test
from emotion.factors import FACTOR_NAMES, compute_all_factors

CACHE = ROOT / "data" / "cache"
WINDOWS = (60, 30)               # 尾部窗口（样本仅 ~6 个月，不再开 120 档）
HORIZONS = (5, 10, 20)
CORR_MIN_OBS = 30


def load_fund_flow() -> pd.DataFrame:
    """读取全部 fund_flow_*.parquet，合并为 [date, code, main_net] 长表。"""
    frames = []
    for p in sorted(CACHE.glob("fund_flow_*.parquet")):
        try:
            df = pd.read_parquet(p, columns=["date", "main_net"])
        except Exception:  # noqa: BLE001
            continue
        if df.empty:
            continue
        df["code"] = p.stem.replace("fund_flow_", "")
        frames.append(df)
    if not frames:
        raise RuntimeError("无资金流缓存：请先运行 scripts/fetch_fund_flow.py")
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    return out


def build_fflow_factors(flow: pd.DataFrame, stock_daily: pd.DataFrame,
                        board_map: pd.DataFrame) -> tuple[dict, dict]:
    """板块日度主力净流入强度（%）。返回 ({fflow_1d, fflow_5d}, 覆盖度统计)。"""
    m = board_map[["code", "board_code"]].drop_duplicates()
    f = flow.merge(m, on="code", how="inner")
    main_sum = f.groupby(["date", "board_code"])["main_net"].sum()

    d = stock_daily[stock_daily["trade_status"].astype(str) == "1"]
    d = d.merge(m, on="code", how="inner")
    amt_all = d.groupby(["date", "board_code"])["amount"].sum()
    # 分母只用"有资金流数据"的成分股成交额：抓取未完/限流期间部分个股缺数据，
    # 若用全成分成交额会把覆盖率差异误当因子差异（横截面上是噪声而非偏差）
    cov_codes = set(flow["code"].unique())
    amt_cov = d[d["code"].isin(cov_codes)].groupby(["date", "board_code"])["amount"].sum()

    main_p = main_sum.unstack("board_code").sort_index()
    amt_p = amt_cov.unstack("board_code").sort_index()
    amt_p = amt_p.replace(0, np.nan)
    fflow_1d = (main_p / amt_p * 100.0).replace([np.inf, -np.inf], np.nan)
    fflow_5d = (main_p.rolling(5, min_periods=3).sum()
                / amt_p.rolling(5, min_periods=3).sum() * 100.0
                ).replace([np.inf, -np.inf], np.nan)

    # 覆盖度：有资金流数据的成分股成交额占板块成交额的比例（按板块合计）
    cov = amt_cov.groupby("board_code").sum()
    tot = amt_all.groupby("board_code").sum()
    coverage = (cov / tot.replace(0, np.nan)).describe()
    cov_stat = {"mean": float(coverage.mean()), "min": float(coverage.min())}
    return {"fflow_1d": fflow_1d, "fflow_5d": fflow_5d}, cov_stat


def window_row(fac: pd.DataFrame, board_close: pd.DataFrame,
               horizon: int) -> dict:
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
    days = a.index.intersection(b.index)
    vals = []
    for dt in days:
        pair = pd.concat([a.loc[dt], b.loc[dt]], axis=1,
                         keys=["x", "y"]).dropna()
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
    print("[1/4] 准备数据（缓存口径，无网络）...")
    data = pipeline.prepare(cfg, end_override=pipeline.cache_end_date(cfg))
    board_close = data["board_kline"].pivot(
        index="date", columns="board_code", values="close").sort_index()

    print("[2/4] 装配资金流因子 ...")
    flow = load_fund_flow()
    new_factors, cov_stat = build_fflow_factors(
        flow, data["stock_daily"], data["board_map"])

    print("[3/4] IC / 分层 ...")
    ic_tables = {name: {h: window_row(f, board_close, h) for h in HORIZONS}
                 for name, f in new_factors.items()}
    quant = {}
    for name, f in new_factors.items():
        quant[name] = {}
        for h in (5, 10):
            quant[name][f"全期/{h}日"] = quantile_test(f, board_close, horizon=h)
            tail_days = f.dropna(how="all").index[-60:]
            quant[name][f"近60日/{h}日"] = quantile_test(
                f.loc[f.index.isin(tail_days)], board_close, horizon=h)

    print("[4/4] 与现有因子的横截面相关 ...")
    factors = compute_all_factors(data["stock_daily"], data["board_map"],
                                  data["board_kline"],
                                  market_total=data.get("market_total"))
    corr = {name: {old: xsection_corr_avg(f, factors[old])
                   for old in FACTOR_NAMES}
            for name, f in new_factors.items()}

    # ---------- 输出 ----------
    d0 = min(f.dropna(how="all").index[0] for f in new_factors.values())
    d1 = max(f.dropna(how="all").index[-1] for f in new_factors.values())
    lines = ["# 主力净流入因子诊断（只描述，不进生产）", "",
             f"资金流样本：{d0:%Y-%m-%d} ~ {d1:%Y-%m-%d}（接口单次上限 120 条，"
             "缓存随每日抓取累积加长）。板块收益率与 IC 口径同 emotion.backtest；"
             "当前成分快照。本报告不据此改任何生产参数；即便结论积极，也须先写入 "
             "config.yaml 预注册、再用后续未参与时段复验。", "",
             f"资金流覆盖率（有数据成分股的成交额占板块成交额比例）："
             f"平均 {cov_stat['mean']:.0%}、最差板块 {cov_stat['min']:.0%}。", ""]

    for name in new_factors:
        lines += [f"## {name}", "",
                  "### IC 汇总（5 日前瞻；30 日窗口附 HAC/bootstrap）", "",
                  "| 窗口 | IC 均值 | IC 胜率 | 样本 |",
                  "|---|---|---|---|"]
        r5 = ic_tables[name][5]
        for label, k in (("全期", "all"), ("近60日", "60"), ("近30日", "30")):
            ic_v = r5["ic_all"] if k == "all" else r5[f"ic_{k}"]
            win_v = r5["win_all"] if k == "all" else r5[f"win_{k}"]
            n_v = r5["n"] if k == "all" else r5[f"n_{k}"]
            lines.append(f"| {label} | {ic_v:+.4f} | {win_v:.0%} | {n_v} |")
        lines.append(f"| 近30日 HAC t / Bootstrap 95% CI | {r5['hac_t']:+.2f} "
                     f"| [{r5['ci_low']:+.3f}, {r5['ci_high']:+.3f}] | |")
        lines += ["", "### 分前瞻窗口 IC", "",
                  "| 前瞻 | 全期 IC | 全期胜率 | 近60日 IC | 近60日胜率 |",
                  "|---|---|---|---|---|"]
        for h in HORIZONS:
            r = ic_tables[name][h]
            lines.append(f"| {h} 日 | {r['ic_all']:+.4f} | {r['win_all']:.0%} "
                         f"| {r['ic_60']:+.4f} | {r['win_60']:.0%} |")
        lines += ["", "### 分层测试（平均前瞻收益，Q1=净流入最弱组）", "",
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
              "> 与 turnover_heat（成交额份额变化）的相关若接近 ±1，说明资金流"
              "并没有带来成交额之外的增量信息。", ""]

    lines += ["## 判读参考（描述性，不构成上线依据）", "",
              "值得进入下一轮预注册的最低门槛（三条同时满足）：",
              "① 全期 IC 与近 60 日 IC 同号（方向是结构性而非单纯风格）；"
              "② 近 30 日 Bootstrap 95% CI 排除 0（当前环境方向可信）；"
              "③ 与全部现有因子的横截面相关 |ρ| < 0.8（有增量信息）。",
              "对照结果请见上表；若不满足，记录为否定假设，避免重复试错。", "",
              "## 限制", "",
              "1. 样本仅约 6 个月（接口上限），跨期结论极不稳定；缓存随每日抓取"
              "累积，建议积累 3 个月后再复读本报告口径。",
              "2. 「主力净流入」是东财按委托单大小的划分口径，非真实主动买卖；"
              "IC 前瞻收益彼此重叠，显著性只看 HAC t 与移动区块 bootstrap。",
              "3. 成分股使用当前快照，存在幸存者偏差；本诊断不做交易成本分析。", ""]
    md = "\n".join(lines)
    print(md)

    out_dir = ROOT / cfg["paths"]["report_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"fund_flow_factor_{pd.Timestamp.today():%Y%m%d}.md"
    path.write_text(md, encoding="utf-8")
    print(f"\n已保存：{path}")


if __name__ == "__main__":
    main()
