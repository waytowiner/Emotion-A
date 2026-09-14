"""因子层诊断 + 因子级风格自适应的预注册验证。

背景（2026-09-13）：组合层 8 个变体全部未通过预注册判定（见
report/backtest_portfolio_layer_20260913.md），说明"怎么组合"不是瓶颈——
6 个因子本身是同一个方向（追动量），组合方式的分散化只是稀释暴露。
问题在因子层：市场风格从动量切到反转后 6 因子 IC 全部转负，而权重仍然是
固定的 1.0/1.0/1.0/1.0/1.0/0.5，系统没有能力"自己发现哪个因子失灵了"。

本脚本分两部分：

  第一部分 · 因子诊断（只描述、不选择）
    - 每个因子的 IC（全期 / 近 120 / 60 / 30 日）与胜率、分层价差
    - 因子 z 的横截面相关性矩阵（判断是否同质冗余）
    - 逐因子滚动 60 日 IC 轨迹

  第二部分 · 预注册变体（查看结果前固定，不做参数搜索）
    全部变体**无前视**：t 日使用的因子权重只来自"已知时点"的滚动 IC
    （IC(d) 在 d+horizon 收盘后才可知）。

    | tag           | 机制                                        |
    |---------------|---------------------------------------------|
    | A0_fixed_eq   | 生产基线：6 因子固定权重、平滑 5            |
    | A1_sign       | 各因子权重 = sign(自身滚动 IC)              |
    | A2_ic_w       | 权重 ∝ 自身滚动 IC（连续，可为负）          |
    | A3_pos_only   | 只保留滚动 IC > 0 的因子，等权              |
    | A4_top2       | 只保留滚动 IC 最高的 2 个因子，等权         |
    | A5_sign_norps | 剔除 rps（纯动量）后再做符号自适应          |
    | A6_nosmooth   | 仅取消 5 日平滑（检验平滑是否钝化反应）     |

  滚动 IC 窗口固定 60 日、最少样本 20 —— 单一档位，**不得**做 20/40/60/90
  网格搜索。判定要求：多窗口（180/120/90/60/30）IC 胜率同向改善，且近 60 日
  的 HAC t 与移动区块 bootstrap 95% CI 不跨零。

运行：python scripts/backtest_factor_adaptive.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from emotion import backtest, pipeline, score
from emotion.factors import compute_all_factors

# ---------- 预注册变体清单（查看结果前固定） ----------
ADAPT_VARIANTS: list[dict] = [
    {"tag": "A0_fixed_eq",   "mode": "fixed",         "smooth": 5, "desc": "生产基线（等权6因子）"},
    {"tag": "A1_sign",       "mode": "sign",          "smooth": 5, "desc": "权重=sign(滚动IC)"},
    {"tag": "A2_ic_w",       "mode": "ic_weighted",   "smooth": 5, "desc": "权重∝滚动IC"},
    {"tag": "A3_pos_only",   "mode": "positive_only", "smooth": 5, "desc": "仅留IC>0因子"},
    {"tag": "A4_top2",       "mode": "top_k",         "smooth": 5, "k": 2, "desc": "仅留IC最高2因子"},
    {"tag": "A5_sign_norps", "mode": "sign",          "smooth": 5, "drop": ["rps"], "desc": "剔除rps后符号自适应"},
    {"tag": "A6_nosmooth",   "mode": "fixed",         "smooth": 1, "desc": "取消5日平滑"},
]

IC_WINDOW = 60       # 因子滚动 IC 窗口（交易日）——预注册单一档位
IC_MIN_OBS = 20
EVAL_WINDOWS = [180, 120, 90, 60, 30]
HORIZON = 5

# 补充候选：**不是**预注册变体。依据来自同一份数据的因子诊断（全期与近 60 日
# IC 同号为负的因子，其权重方向可能是系统性的，而非风格性的），因此存在样本内
# 发现偏差——单独成节展示，不进入预注册判定表，必须靠后续时段复验。
SUPPLEMENT_VARIANTS: list[dict] = [
    {"tag": "B1_heat_flip", "mode": "fixed", "smooth": 5,
     "flip": ["turnover_heat"], "desc": "仅把 turnover_heat 权重取反"},
    {"tag": "B2_heat_flip_nolim", "mode": "fixed", "smooth": 5,
     "flip": ["turnover_heat"], "drop": ["limit_up_strength"],
     "desc": "热度取反 + 剔除涨停强度（后者近60日显著为负但全期≈0）"},
]


def spec_base_w(spec: dict, base_w: pd.Series) -> pd.Series:
    """按 spec 的 flip / drop 派生基线权重（仅 fixed 模式使用）。"""
    w = base_w.copy()
    for k in spec.get("flip", []):
        if k in w.index:
            w[k] = -w[k]
    for k in spec.get("drop", []):
        if k in w.index:
            w[k] = 0.0
    return w


# ---------- 基础工具 ----------
def factor_z(factors: dict[str, pd.DataFrame], z_clip: float | None
             ) -> tuple[np.ndarray, list[str], pd.DatetimeIndex, list[str]]:
    """因子横截面 z 矩阵 (F, T, B) + 轴标签；与 score.combine_factors 同口径。"""
    avail = {k: v for k, v in factors.items() if v is not None and not v.empty}
    names = list(avail)
    zs = {}
    for k, v in avail.items():
        z = score.zscore_cross_section(v)
        if z_clip is not None and z_clip > 0:
            z = z.clip(-z_clip, z_clip)
        zs[k] = z
    idx = sorted(set().union(*[z.index for z in zs.values()]))
    cols = sorted(set().union(*[z.columns for z in zs.values()]))
    mats = np.stack([zs[k].reindex(index=idx, columns=cols).to_numpy(dtype=float)
                     for k in names])
    return mats, names, pd.DatetimeIndex(idx), cols


def daily_ic_matrix(zscores: dict[str, pd.DataFrame], fwd: pd.DataFrame,
                    min_pairs: int = 10) -> pd.DataFrame:
    """逐因子每日截面 IC（z 矩阵 vs 前瞻收益）。"""
    out = {}
    for name, z in zscores.items():
        rec = {}
        for dt in z.index.intersection(fwd.index):
            pair = pd.concat([z.loc[dt], fwd.loc[dt]], axis=1,
                             keys=["s", "f"]).dropna()
            if len(pair) < min_pairs:
                continue
            c = pair["s"].corr(pair["f"], method="spearman")
            if pd.notna(c):
                rec[dt] = c
        out[name] = pd.Series(rec)
    return pd.DataFrame(out)


def lag_to_known(ic: pd.DataFrame, calendar: pd.DatetimeIndex,
                 lag: int) -> pd.DataFrame:
    """把 IC 矩阵整体后移 lag 个交易日：IC(d) 在 d+lag 收盘后才可知。"""
    rows = {}
    for col in ic.columns:
        s = ic[col].dropna()
        if s.empty:
            continue
        pos = calendar.searchsorted(s.index)
        kp = pos + lag
        ok = kp < len(calendar)
        rows[col] = pd.Series(s.values[ok], index=calendar[kp[ok]])
    return pd.DataFrame(rows).reindex(calendar)


def weights_from_ic(ic_known: pd.DataFrame, *, mode: str, window: int,
                    min_obs: int, base_w: pd.Series,
                    drop: list[str] | None = None) -> pd.DataFrame:
    """逐日因子权重矩阵（只使用已知时点 IC）。"""
    cols = list(ic_known.columns)
    w = pd.DataFrame(0.0, index=ic_known.index, columns=cols)
    vals = ic_known.to_numpy(dtype=float)
    n, f = vals.shape
    for i in range(n):
        row = vals[max(0, i - window + 1): i + 1]
        finite = np.isfinite(row)
        counts = finite.sum(axis=0)
        means = np.where(counts >= min_obs,
                         np.nanmean(np.where(finite, row, np.nan), axis=0), np.nan)
        if not np.isfinite(means).any():
            w.iloc[i] = base_w.to_numpy()
            continue
        vec = np.zeros(f)
        for j, c in enumerate(cols):
            m = means[j]
            if drop and c in drop:
                vec[j] = 0.0
            elif mode == "sign":
                vec[j] = np.sign(m) if np.isfinite(m) else 0.0
            elif mode == "ic_weighted":
                vec[j] = m if np.isfinite(m) else 0.0
            elif mode == "positive_only":
                vec[j] = 1.0 if (np.isfinite(m) and m > 0) else 0.0
            elif mode == "top_k":
                vec[j] = 0.0
            else:
                vec[j] = base_w.get(c, 1.0)
        if mode == "top_k":
            k = min(2, f)
            order = np.argsort(np.where(np.isfinite(means), means, -np.inf))[::-1][:k]
            vec[order] = 1.0
        if mode in {"sign", "ic_weighted", "positive_only"} and np.abs(vec).sum() == 0:
            # 全部因子都没有可用 IC（或全为负）时退回等权基权重，而不是留空：
            # 留空会让该变体少掉若干日期，与基线不在同一组截面上比较。
            vec = base_w.to_numpy()
        w.iloc[i] = vec
    return w


def _nan_rolling_mean(arr: np.ndarray, window: int, min_periods: int) -> np.ndarray:
    """NaN 感知的滚动均值，口径与 score.smooth_score 一致。

    smooth_score 用 raw.rolling(window, min_periods).mean()，pandas 的 rolling
    会自动跳过 NaN 并按非 NaN 个数求均值——numpy 的 rolling 没有这个行为，
    必须显式实现，否则合成分与生产口径不一致（曾导致基线校验差 1.98）。
    """
    if window <= 1:
        return arr.copy()
    n_dates = arr.shape[0]
    out = np.full_like(arr, np.nan)
    for t in range(n_dates):
        block = arr[max(0, t - window + 1): t + 1]
        finite = np.isfinite(block)
        cnt = finite.sum(axis=0)
        total = np.where(finite, block, 0.0).sum(axis=0)
        out[t] = np.where(cnt >= min_periods,
                          total / np.where(cnt > 0, cnt, np.nan), np.nan)
    return out


def build_composite(mats: np.ndarray, wts: pd.DataFrame, smooth: int,
                    min_coverage: float = 0.5
                    ) -> tuple[np.ndarray, np.ndarray]:
    """按逐日权重合成情绪分：numer/Σ|w|，只聚合该单元格实际存在的因子。

    返回 (raw, smoothed)。**日期筛选必须用 raw**——生产路径
    `score.build_scores` 是先 rolling 再 `smoothed.loc[raw.dropna(how="all").index]`，
    即"有原始分"的日期才保留。若改用平滑后筛选，会多保留"当日原始分全空、
    但邻近日期有值"的行（本机实测多出 13 个日期），进而改变周度调仓日，
    使基线复刻结果从 -0.69% 漂到 +14.12%（曾据此误读整张表）。

    smooth <= 1 表示不做平滑（生产路径里 window=1 会让 rolling 的 min_periods
    不可能满足而产出全 NaN，因此本脚本把 1 定义为"不平滑"，这是有意与生产
    不同的语义，仅在 A6 变体中体现）。
    """
    valid = np.isfinite(mats)
    filled = np.where(valid, mats, 0.0)
    w = wts.to_numpy(dtype=float)                     # (T, F)
    w_abs = np.abs(w)
    numer = np.einsum("ftb,tf->tb", filled, w)
    denom = np.einsum("ftb,tf->tb", valid.astype(float), w_abs)
    total = w_abs.sum(axis=1)[:, None]
    coverage = np.divide(denom, np.where(total > 0, total, np.nan))
    raw = np.where(coverage >= min_coverage, numer / np.where(denom > 0, denom, np.nan),
                   np.nan)
    smoothed = raw if (not smooth or smooth <= 1) else _nan_rolling_mean(
        raw, int(smooth), max(2, int(smooth) // 2))
    return raw, smoothed


def window_ic(comp: pd.DataFrame, board_close: pd.DataFrame, window: int | None
              ) -> dict:
    """某窗口内的 IC 统计（均值/胜率/HAC/bootstrap）。"""
    ic = backtest.daily_ic_series(comp, board_close, horizon=HORIZON)
    if ic.empty:
        return {"ic_mean": np.nan, "win_rate": np.nan, "n": 0}
    sub = ic if window is None else ic.tail(window)
    if sub.empty:
        return {"ic_mean": np.nan, "win_rate": np.nan, "n": 0}
    rob = backtest.ic_robustness(sub, horizon=HORIZON, bootstrap_samples=1000)
    return {"ic_mean": float(sub.mean()), "win_rate": float((sub > 0).mean()),
            "n": int(len(sub)), **rob}


def fmt(v, pct=False, digits=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return f"{v:.{digits}%}" if pct else f"{v:.{digits}f}"


# ---------- 主流程 ----------
def main() -> None:
    ap = argparse.ArgumentParser(description="因子层诊断 + 因子级风格自适应验证")
    ap.add_argument("--end", default=None)
    args = ap.parse_args()

    cfg = pipeline.load_config()
    end = args.end or pipeline.cache_end_date(cfg)
    print(f"[1/6] 准备数据（缓存口径，无网络），end={end} ...")
    data = pipeline.prepare(cfg, end_override=end)

    print("[2/6] 计算因子 ...")
    factors = compute_all_factors(data["stock_daily"], data["board_map"],
                                 data["board_kline"],
                                 market_total=data.get("market_total"))
    z_clip = cfg.get("score_z_clip", 3.0)
    mats, names, dates, boards = factor_z(factors, z_clip)
    board_close = data["board_kline"].pivot(
        index="date", columns="board_code", values="close").sort_index()

    # 因子 z 与情绪分逐日截面相关性（诊断用）
    zs = {n: pd.DataFrame(mats[i], index=dates, columns=boards) for i, n in enumerate(names)}
    base_w = pd.Series({n: cfg["weights"].get(n, 1.0) for n in names})

    # 生产基线一致性校验：等权合成必须与 score.combine_factors 相同
    base_wts = pd.DataFrame(np.tile(base_w.to_numpy(), (len(dates), 1)),
                            index=dates, columns=names)
    base_raw, base_sm = build_composite(mats, base_wts, cfg.get("smooth_window", 5))
    base_raw = pd.DataFrame(base_raw, index=dates, columns=boards)
    prob_comp = pd.DataFrame(base_sm, index=dates, columns=boards)
    prob_comp = prob_comp.loc[base_raw.dropna(how="all").index]
    _, ref_smoothed, _ = score.build_scores(
        factors, cfg["weights"], cfg.get("smooth_window", 5),
        lookback=cfg.get("score_lookback", 250), z_clip=z_clip)
    common = prob_comp.index.intersection(ref_smoothed.index)
    diff = (prob_comp.loc[common] - ref_smoothed.loc[common]).abs().max().max()
    print(f"      基线一致性校验：与 score.build_scores 最大绝对差 = {diff:.2e}")
    baseline_ok = bool(diff < 1e-9)

    # 基线轮动交叉校验：A0 的追涨轮动必须与 backtest_regime_adaptive.py 的
    # 「固定追涨 Top3」行一致（同函数、同合成分、同 board_close）。不一致说明
    # 日期筛选口径仍有偏差，整张表不可解读。
    _probe = backtest.rotation_backtest(
        prob_comp, board_close, top_n=cfg["signal"].get("top_n", 3),
        cost_bps=cfg["backtest"].get("cost_bps", 5), direction="top")
    print(f"      基线轮动交叉校验：A0 追涨累计 "
          f"{_probe['strategy']['total_ret']:+.2%}、夏普 "
          f"{_probe['strategy']['sharpe']:.2f}、回撤 "
          f"{_probe['strategy']['max_dd']:.2%}、换手 {_probe['total_turnover']:.2f} "
          f"（生产脚本「固定追涨 Top3」应与此相同）")

    print("[3/6] 逐因子 IC 与相关性诊断 ...")
    fwd = backtest.forward_returns(board_close, HORIZON)
    ic_mat = daily_ic_matrix(zs, fwd)
    ic_known = lag_to_known(ic_mat, board_close.index, HORIZON)

    # 横截面平均相关性矩阵
    corr_acc, corr_n = pd.DataFrame(0.0, index=names, columns=names), 0
    for dt in zs[names[0]].index.intersection(board_close.index):
        sub = pd.DataFrame({n: zs[n].loc[dt] for n in names}).dropna()
        if len(sub) < 30:
            continue
        corr_acc += sub.corr(method="spearman").fillna(0.0)
        corr_n += 1
    corr_mean = corr_acc / corr_n if corr_n else corr_acc

    print("[4/6] 构造预注册变体的合成分 ...")
    comps: dict[str, pd.DataFrame] = {}
    for spec in ADAPT_VARIANTS + SUPPLEMENT_VARIANTS:
        if spec["mode"] == "fixed":
            bw = spec_base_w(spec, base_w)
            if bw.abs().sum() <= 0:
                raise ValueError(f"{spec['tag']} 的权重全为零")
            wts = pd.DataFrame(np.tile(bw.to_numpy(), (len(dates), 1)),
                               index=dates, columns=names)
        else:
            bw = base_w
            wts = weights_from_ic(ic_known, mode=spec["mode"], window=IC_WINDOW,
                                  min_obs=IC_MIN_OBS, base_w=base_w,
                                  drop=spec.get("drop"))
        arr_raw, arr_sm = build_composite(mats, wts, spec["smooth"])
        raw_df = pd.DataFrame(arr_raw, index=dates, columns=boards)
        df = pd.DataFrame(arr_sm, index=dates, columns=boards)
        # 与生产口径一致：按**原始分**筛日期，而非平滑后
        df = df.loc[raw_df.dropna(how="all").index]
        comps[spec["tag"]] = df
        print(f"      {spec['tag']:14s} 有效日期 {len(df)}")

    print("[5/6] 多窗口 IC 评估 + 轮动回测 ...")
    cost_bps = cfg["backtest"].get("cost_bps", 5)
    top_n = cfg["signal"].get("top_n", 3)
    rows_ic: dict[str, dict] = {}
    rot: dict[str, dict] = {}
    for spec in ADAPT_VARIANTS + SUPPLEMENT_VARIANTS:
        comp = comps[spec["tag"]]
        rows_ic[spec["tag"]] = {w: window_ic(comp, board_close, w)
                                for w in EVAL_WINDOWS}
        rot[spec["tag"]] = backtest.rotation_backtest(
            comp, board_close, top_n=top_n, cost_bps=cost_bps, direction="top")
        r = rows_ic[spec["tag"]]
        print(f"      {spec['tag']:14s} IC60 {r[60]['ic_mean']:+.4f} "
              f"胜率60 {r[60]['win_rate']:.0%} | 追涨累计 "
              f"{rot[spec['tag']]['strategy']['total_ret']:+.2%}")

    print("[6/6] 生成报告 ...")
    lines = ["# 因子层诊断与因子级风格自适应（预注册验证）", "",
             f"数据区间：{dates[0]:%Y-%m-%d} ~ {dates[-1]:%Y-%m-%d}"
             f"（{len(dates)} 个交易日）；前瞻窗口 {HORIZON} 日；"
             f"因子滚动 IC 窗口 {IC_WINDOW} 日、最少样本 {IC_MIN_OBS}。", "",
             "**无前视保证**：t 日使用的因子权重只来自后移 "
             f"{HORIZON} 个交易日（已走完前瞻窗口）的滚动 IC，任意变体均不使用"
             "未来信息。滚动 IC 窗口为预注册单一档位，未做网格搜索。", ""]

    lines += ["## 一、基线一致性校验", "",
              f"- 本脚本的等权合成与 `score.build_scores` 的最大绝对差 = "
              f"{diff:.2e} ⇒ **{'一致，口径无偏差' if baseline_ok else '不一致，结果不可采信'}**。",
              f"- 日期筛选按**原始分**（同生产）而非平滑后；提前筛平滑分会多留 13 个"
              f"「当日原始分全空、邻日有值」的日期并丢掉首日，使复刻基线从 -0.69% "
              f"漂到 +14.12%——这是本轮踩过的坑，已在代码注释中固化。",
              f"- 基线轮动交叉校验：A0 追涨累计 {_probe['strategy']['total_ret']:+.2%}、"
              f"夏普 {_probe['strategy']['sharpe']:.2f}、回撤 "
              f"{_probe['strategy']['max_dd']:.2%}、累计换手 "
              f"{_probe['total_turnover']:.2f}；应与 `scripts/backtest_regime_adaptive.py` "
              f"的「固定追涨 Top3」行逐项相同。",
              ""]

    lines += ["## 二、因子诊断（只描述，不选择）", "",
              "### 各因子 IC 与胜率", "",
              "| 因子 | 全期 IC | 全期胜率 | 近120日 IC | 近期胜率 | 近60日 IC "
              "| 近60胜率 | 近30日 IC | 近30胜率 |", "|---|---|---|---|---|---|---|---|---|"]
    for n in names:
        s = ic_mat[n].dropna()
        if s.empty:
            continue
        def cell(w):
            sub = s if w is None else s.tail(w)
            return (f"{sub.mean():+.4f}", f"{(sub > 0).mean():.0%}") if len(sub) else ("—", "—")
        a, b = cell(None)
        c, d = cell(120)
        e, f_ = cell(60)
        g, h = cell(30)
        lines.append(f"| {n} | {a} | {b} | {c} | {d} | {e} | {f_} | {g} | {h} |")

    lines += ["", "### 因子 z 的横截面平均相关性（Spearman，{n} 个有效日）".format(n=corr_n),
              "",
              "| — | " + " | ".join(names) + " |",
              "|" + "---|" * (len(names) + 1)]
    for a in names:
        cells = " | ".join(f"{corr_mean.loc[a, b]:.2f}" for b in names)
        lines.append(f"| {a} | {cells} |")
    off = corr_mean.to_numpy()[~np.eye(len(names), dtype=bool)]
    lines += ["", f"非对角平均相关 = **{off.mean():.2f}**（越高说明因子越同质、"
                  f"综合分越接近单一暴露）。", ""]

    lines += ["", "### 各因子稳健性（全期 vs 近 60 日）", "",
              "判断标准：**全期与近期同号**才是方向性特征（可考虑改方向）；"
              "只有近期为负、全期为正，属风格问题（不可改符号，那是拟合最近行情）。", "",
              "| 因子 | 全期 IC | 全期 HAC t | 全期 CI | 近60 IC | 近60 HAC t "
              "| 近60 CI | 全期/近期同号 |", "|---|---|---|---|---|---|---|---|"]
    for n in names:
        s_all = ic_mat[n].dropna()
        s_60 = s_all.tail(60)
        if s_all.empty or s_60.empty:
            continue
        r_all = backtest.ic_robustness(s_all, horizon=HORIZON, bootstrap_samples=1000)
        r_60 = backtest.ic_robustness(s_60, horizon=HORIZON, bootstrap_samples=1000)
        same = "是" if (np.sign(s_all.mean()) == np.sign(s_60.mean())
                        and s_all.mean() != 0) else "否"
        lines.append(
            f"| {n} | {s_all.mean():+.4f} | {fmt(r_all['IC_HAC_t'], digits=2)} "
            f"| [{r_all['IC_bootstrap_low']:.4f}, {r_all['IC_bootstrap_high']:.4f}] "
            f"| {s_60.mean():+.4f} | {fmt(r_60['IC_HAC_t'], digits=2)} "
            f"| [{r_60['IC_bootstrap_low']:.4f}, {r_60['IC_bootstrap_high']:.4f}] "
            f"| {same} |")

    lines += ["", "### 各因子近 60 日稳健性（判断该弃还是该反）", "",
              "| 因子 | 近60日 IC | 胜率 | HAC t | HAC p | Bootstrap 95% CI | 是否跨零 |",
              "|---|---|---|---|---|---|---|"]
    fac_rob = {}
    for n in names:
        s = ic_mat[n].dropna().tail(60)
        if s.empty:
            continue
        rob = backtest.ic_robustness(s, horizon=HORIZON, bootstrap_samples=1000)
        fac_rob[n] = {"ic": float(s.mean()), "win": float((s > 0).mean()), **rob}
        lo, hi = rob["IC_bootstrap_low"], rob["IC_bootstrap_high"]
        cross = "是" if (pd.notna(lo) and pd.notna(hi) and lo <= 0 <= hi) else "否"
        ci = f"[{lo:.4f}, {hi:.4f}]" if pd.notna(lo) else "—"
        lines.append(f"| {n} | {s.mean():+.4f} | {(s > 0).mean():.0%} "
                     f"| {fmt(rob['IC_HAC_t'], digits=2)} "
                     f"| {fmt(rob['IC_HAC_p'], digits=3)} | {ci} | {cross} |")
    lines += ["", "> 全期/近期同号且量级接近 = 该因子方向稳定；跨零 = 该因子在此期间"
                  "没有可辨认的方向，既不该留也不该反。", ""]

    lines += ["## 三、预注册变体定义", "",
              "| tag | 机制 | 平滑 |", "|---|---|---|"]
    for spec in ADAPT_VARIANTS:
        lines.append(f"| {spec['tag']} | {spec['desc']} | {spec['smooth']} |")

    lines += ["", "## 四、多窗口 IC 对照（胜率为核心指标）", "",
              "| 变体 | " + " | ".join(f"{w}日 IC | {w}日胜率" for w in EVAL_WINDOWS) + " |",
              "|" + "---|" * (len(EVAL_WINDOWS) * 2 + 1)]
    for spec in ADAPT_VARIANTS:
        r = rows_ic[spec["tag"]]
        cells = " | ".join(f"{fmt(r[w]['ic_mean'], digits=4)} | "
                           f"{fmt(r[w]['win_rate'], pct=True, digits=0)}"
                           for w in EVAL_WINDOWS)
        lines.append(f"| {spec['tag']} | {cells} |")

    lines += ["", "## 五、近 60 日稳健性（HAC + 移动区块 Bootstrap）", "",
              "| 变体 | IC均值 | 胜率 | 样本 | HAC t | HAC p | Bootstrap 95% CI | 是否跨零 |",
              "|---|---|---|---|---|---|---|---|"]
    for spec in ADAPT_VARIANTS:
        r = rows_ic[spec["tag"]][60]
        lo, hi = r.get("IC_bootstrap_low", np.nan), r.get("IC_bootstrap_high", np.nan)
        cross = "是" if (pd.notna(lo) and pd.notna(hi) and lo <= 0 <= hi) else "否"
        ci = f"[{lo:.4f}, {hi:.4f}]" if pd.notna(lo) else "—"
        lines.append(f"| {spec['tag']} | {fmt(r['ic_mean'], digits=4)} "
                     f"| {fmt(r['win_rate'], pct=True, digits=0)} | {r['n']} "
                     f"| {fmt(r.get('IC_HAC_t'), digits=2)} "
                     f"| {fmt(r.get('IC_HAC_p'), digits=3)} | {ci} | {cross} |")

    lines += ["", "## 六、周度轮动对照（追涨 Top3 / 抄底 Bottom3）", "",
              "| 变体 | 追涨累计 | 追涨夏普 | 追涨回撤 | 抄底累计 | 抄底夏普 "
              "| 成本损耗 | 累计换手 |", "|---|---|---|---|---|---|---|---|"]
    bot = {}
    for spec in ADAPT_VARIANTS:
        b = backtest.rotation_backtest(comps[spec["tag"]], board_close,
                                       top_n=top_n, cost_bps=cost_bps,
                                       direction="bottom")
        bot[spec["tag"]] = b
        t, tt = rot[spec["tag"]], rot[spec["tag"]]["strategy"]
        lines.append(f"| {spec['tag']} | {tt['total_ret']:+.2%} | {tt['sharpe']:.2f} "
                     f"| {tt['max_dd']:.2%} | {b['strategy']['total_ret']:+.2%} "
                     f"| {b['strategy']['sharpe']:.2f} "
                     f"| {t['total_cost_drag']:.2%} | {t['total_turnover']:.2f} |")

    # 判定：相对 A0 基线，多窗口 IC 胜率不降 且 近60日 HAC/bootstrap 不跨零 且 IC 更强
    base_tag = "A0_fixed_eq"
    b0 = rows_ic[base_tag]
    # 补充候选（样本内发现，不进入判定）
    lines += ["", "## 六之二、补充候选（样本内发现，不构成通过）", "",
              "依据来自本报告第二节的因子诊断：若某因子的 IC 在**全期与近 60 日同号为负**，"
              "其权重方向可能是系统性的（而非本轮风格造成的）。下列变体据此构造，"
              "**存在样本内发现偏差**，因此单独成节、不进入判定表，只能靠后续时段复验。", "",
              "| tag | 机制 | " + " | ".join(f"{w}日 IC | {w}日胜率"
                                            for w in EVAL_WINDOWS) + " |",
              "|" + "---|" * (len(EVAL_WINDOWS) * 2 + 2)]
    for spec in [ADAPT_VARIANTS[0]] + SUPPLEMENT_VARIANTS:
        r = rows_ic[spec["tag"]]
        cells = " | ".join(f"{fmt(r[w]['ic_mean'], digits=4)} | "
                           f"{fmt(r[w]['win_rate'], pct=True, digits=0)}"
                           for w in EVAL_WINDOWS)
        lines.append(f"| {spec['tag']} | {spec['desc']} | {cells} |")

    lines += ["", "近 60 日与轮动口径：", "",
              "| tag | 近60 IC | 近60胜率 | 近60 HAC t | 近60 CI | 追涨累计 | 追涨夏普 "
              "| 追涨回撤 | 抄底累计 |", "|---|---|---|---|---|---|---|---|---|"]
    for spec in [ADAPT_VARIANTS[0]] + SUPPLEMENT_VARIANTS:
        r = rows_ic[spec["tag"]][60]
        t = rot[spec["tag"]]["strategy"]
        b = backtest.rotation_backtest(comps[spec["tag"]], board_close,
                                       top_n=top_n, cost_bps=cost_bps,
                                       direction="bottom")
        lines.append(f"| {spec['tag']} | {fmt(r['ic_mean'], digits=4)} "
                     f"| {fmt(r['win_rate'], pct=True, digits=0)} "
                     f"| {fmt(r.get('IC_HAC_t'), digits=2)} "
                     f"| [{r['IC_bootstrap_low']:.4f}, {r['IC_bootstrap_high']:.4f}] "
                     f"| {t['total_ret']:+.2%} | {t['sharpe']:.2f} "
                     f"| {t['max_dd']:.2%} | {b['strategy']['total_ret']:+.2%} |")

    lines += ["", "## 七、判定（对照 A0_fixed_eq 基线）", "",
              "通过条件（三条同时满足）：① 全部 5 个窗口的 IC 均值都不低于基线；"
              "② 近 60 日 IC 胜率高于基线；③ 近 60 日 Bootstrap 95% CI 不跨零。", "",
              "| 变体 | 窗口IC全不低 | 近60胜率更高 | CI不跨零 | 结论 |",
              "|---|---|---|---|---|"]
    passed = []
    for spec in ADAPT_VARIANTS[1:]:
        tag = spec["tag"]
        r = rows_ic[tag]
        c1 = all(r[w]["ic_mean"] >= b0[w]["ic_mean"] for w in EVAL_WINDOWS)
        c2 = r[60]["win_rate"] > b0[60]["win_rate"]
        lo, hi = r[60].get("IC_bootstrap_low"), r[60].get("IC_bootstrap_high")
        c3 = bool(pd.notna(lo) and pd.notna(hi) and not (lo <= 0 <= hi))
        verdict = "通过" if (c1 and c2 and c3) else "未通过"
        if c1 and c2 and c3:
            passed.append(tag)
        lines.append(f"| {tag} | {'是' if c1 else '否'} | {'是' if c2 else '否'} "
                     f"| {'是' if c3 else '否'} | {verdict} |")

    # 描述性面板（非判定，仅刻画"改善方向是否一致"）
    lines += ["", "### 描述性面板（非判定，仅刻画改善方向的一致性）", "",
              "近 60 日样本仅 60 个重叠观测，判据③（Bootstrap CI 不跨零）在本窗口"
              "实际上不可能满足（IC≈0.04 时标准误约 0.06，需 IC>0.12 才能排除 0）。"
              "因此另附本面板，刻画改善方向是否跨窗口一致——它不构成通过，"
              "但用于下一轮预注册时判断值不值得再验。", "",
              "| 变体 | 相对基线的 IC 差（180/120/90/60/30） | 5窗口改善数 "
              "| 近60胜率变化 | HAC t 变化 |", "|---|---|---|---|---|"]
    for spec in ADAPT_VARIANTS[1:]:
        tag = spec["tag"]
        r = rows_ic[tag]
        diffs = [r[w]["ic_mean"] - b0[w]["ic_mean"] for w in EVAL_WINDOWS]
        n_better = sum(1 for d in diffs if d > 0)
        dwin = r[60]["win_rate"] - b0[60]["win_rate"]
        dt = (r[60].get("IC_HAC_t", np.nan) - b0[60].get("IC_HAC_t", np.nan))
        lines.append("| " + tag + " | "
                     + " / ".join(f"{d:+.4f}" for d in diffs)
                     + f" | {n_better}/5 | {dwin:+.0%} | {fmt(dt, digits=2)} |")

    lines += ["", "## 八、结论与限制", "",
              f"1. 通过预注册判定的变体：{'、'.join(passed) if passed else '无'}。",
              "2. 本报告未改动 config.yaml 与任何生产脚本；任何上线变更必须先把选定",
              "   机制写入配置，再在**未参与本轮比较的后续时段**复验。",
              "3. 禁止据本表对 IC_WINDOW / 平滑窗口 / 保留因子个数做网格搜索——",
              "   那会立刻把本轮结论退化成样本内拟合。",
              "4. 成分股使用当前快照、板块等权参考不可交易；因子 IC 前瞻收益彼此重叠，",
              "   显著性优先看 HAC t 与移动区块 Bootstrap 区间，不看朴素 IR。", ""]
    md = "\n".join(lines)
    print(md)

    out_dir = ROOT / cfg["paths"]["report_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"factor_adaptive_{pd.Timestamp.today():%Y%m%d}.md"
    path.write_text(md, encoding="utf-8")
    print(f"\n已保存：{path}")


if __name__ == "__main__":
    main()
