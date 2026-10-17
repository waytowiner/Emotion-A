"""回测层：IC 分析、分层测试、Top-N 板块轮动回测。

标的为板块指数本身（复权口径同输入数据）。ETF 与板块的映射关系
留给使用者在报告层自行对照，不在本模块内耦合。
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def equal_weight_portfolio(holdings: list[str] | tuple[str, ...] | pd.Index) -> pd.Series:
    """将持仓代码转换为等权组合权重。空持仓用空 Series 表示。"""
    index = pd.Index(holdings).drop_duplicates()
    if index.empty:
        return pd.Series(dtype=float)
    return pd.Series(1.0 / len(index), index=index, dtype=float)


def portfolio_turnover(previous: pd.Series, target: pd.Series) -> float:
    """两期组合的一次单边换手率（权重 L1 差的一半）。

    例如 Top3 等权切换到全市场 100 个板块时，若 Top3 均保留，换手率为
    97%；反向切回的换手率也必须为 97%。不能以证券只数和目标持仓数相除，
    否则变持仓数时会出现方向不对称、甚至超过 100% 的虚假成本。

    空组合代表尚未建仓；首笔建仓是否计费由调用方的执行口径决定。
    """
    cols = previous.index.union(target.index)
    if cols.empty:
        return 0.0
    old = previous.reindex(cols, fill_value=0.0)
    new = target.reindex(cols, fill_value=0.0)
    return float((new - old).abs().sum() / 2.0)


def rebalance_dates(score_index: pd.DatetimeIndex, *, freq: str = "W-FRI",
                    include_open_period: bool = False) -> pd.DatetimeIndex:
    """返回真正可执行的调仓日，默认排除最后一个尚未走完的周。

    周频信号在周内任一天运行时，最后一个自然周还没有收盘；把它当成历史
    周五调仓，会造成实盘与回测频率不一致。若最后一个可用日期正好是频率
    锚点（如周五），该周正常保留。节假日周在下一个交易周已有数据时也会
    被保留（该周真实最后交易日即为调仓日）。
    """
    days = pd.DatetimeIndex(score_index).dropna().sort_values().unique()
    if days.empty:
        return pd.DatetimeIndex([])
    dates = (pd.Series(days, index=days).groupby(pd.Grouper(freq=freq))
             .max().dropna())
    out = pd.DatetimeIndex(dates.to_numpy())
    if (not include_open_period and freq.upper().startswith("W") and len(out)
            and days[-1].normalize() < days[-1].to_period(freq).end_time.normalize()):
        # 若数据只到本周周一至周四，该组尚未完成；等周五收盘后再执行。
        out = out[:-1]
    return out


def performance_stats(nav: pd.Series) -> dict:
    """净值序列绩效；累计收益始终优先，短样本年化仅保留为参考。"""
    s = nav.dropna()
    r = s.pct_change().dropna()
    if len(s) < 2 or r.empty:
        return {
            "total_ret": np.nan, "annual_ret": np.nan, "sharpe": np.nan,
            "max_dd": np.nan, "trading_days": int(len(r)),
            "annualized_reliable": False,
        }
    years = (s.index[-1] - s.index[0]).days / 365.25
    annual = s.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
    sharpe = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan
    dd = (s / s.cummax() - 1).min()
    return {
        "total_ret": float(s.iloc[-1] / s.iloc[0] - 1),
        "annual_ret": annual,
        "sharpe": sharpe,
        "max_dd": dd,
        "trading_days": int(len(r)),
        # 约半年以前的年化很容易被几天波动放大，不应用于选择策略。
        "annualized_reliable": bool(len(r) >= 126),
    }


def annual_return_text(stats: dict) -> str:
    """展示用年化：短于半年的样本只展示累计收益。"""
    if not stats.get("annualized_reliable", True):
        return "—（样本<半年）"
    return f"{stats['annual_ret']:.2%}"


def forward_returns(board_close: pd.DataFrame, days: int) -> pd.DataFrame:
    """板块收盘价的前瞻 days 日收益（T 收盘买入 -> T+days 收盘）。"""
    return board_close.pct_change(days, fill_method=None).shift(-days)


def ic_analysis(score: pd.DataFrame, board_close: pd.DataFrame,
                horizons: list[int]) -> pd.DataFrame:
    """各前瞻窗口的 Spearman IC 序列统计，附带重叠收益的稳健性校正。

    IC = 当日情绪分截面 与 前瞻收益截面 的秩相关。
    前瞻收益会重叠（例如连续两天的 5 日收益共享 4 天），因此 ``IR`` 是
    描述性的朴素值；显著性应优先看 HAC t 值、移动区块 bootstrap 区间和
    非重叠样本均值。
    """
    rows = []
    for h in horizons:
        ic = daily_ic_series(score, board_close, h)
        robust = ic_robustness(ic, horizon=h)
        rows.append({
            "horizon": h,
            "IC_mean": ic.mean(),
            "IC_std": ic.std(),
            "IR": ic.mean() / ic.std() if ic.std() > 0 else np.nan,
            "IC_win_rate": (ic > 0).mean(),
            "n_obs": len(ic),
            **robust,
        })
    return pd.DataFrame(rows)


def quantile_test(score: pd.DataFrame, board_close: pd.DataFrame,
                  horizon: int = 5, n_groups: int = 5) -> pd.DataFrame:
    """分层测试：按情绪分分 n_groups 组，比较各组前瞻 horizon 日均收益。

    单调递增的分组收益说明情绪分具有区分度。
    """
    fwd = forward_returns(board_close, horizon)
    records = []
    for dt in score.index.intersection(fwd.index):
        pair = pd.concat([score.loc[dt], fwd.loc[dt]], axis=1, keys=["s", "f"]).dropna()
        if len(pair) < n_groups * 3:
            continue
        pair["g"] = pd.qcut(pair["s"], n_groups, labels=False, duplicates="drop")
        records.append(pair.groupby("g")["f"].mean())
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    out = df.mean().rename("mean_fwd_ret").to_frame()
    out["n_obs"] = df.count()
    out.index.name = "quantile"
    return out


def quantile_spread_series(score: pd.DataFrame, board_close: pd.DataFrame,
                           horizon: int = 5, n_groups: int = 5) -> pd.Series:
    """每日分层价差序列：最高分组的平均前瞻收益 − 最低分组（Q5−Q1）。

    正值=分组单调向上（动量结构），负值=单调倒置（反转结构）。
    与 IC 同属"决策日 d"口径：d 的值在 d+horizon 收盘后才可知，
    用于判定时必须与 IC 一样后移到可知时点，否则会前视。
    """
    fwd = forward_returns(board_close, horizon)
    out = {}
    for dt in score.index.intersection(fwd.index):
        pair = pd.concat([score.loc[dt], fwd.loc[dt]], axis=1,
                         keys=["s", "f"]).dropna()
        if len(pair) < n_groups * 3:
            continue
        pair["g"] = pd.qcut(pair["s"], n_groups, labels=False, duplicates="drop")
        means = pair.groupby("g")["f"].mean()
        if len(means) < 2:
            continue
        out[dt] = means.iloc[-1] - means.iloc[0]
    return pd.Series(out).sort_index()


def apply_confirm_hysteresis(directions: pd.Series, *, confirm: int = 1) -> pd.Series:
    """方向确认迟滞：非观望方向需连续 confirm 次同向判定才生效，未确认期间按观望
    处理；观望（neutral）即刻生效——风险降级不做迟滞。任何不同读数（含 neutral）
    都会重置连续计数。confirm <= 1 时原样返回。"""
    if confirm <= 1:
        return directions
    out, last, streak = [], None, 0
    for d in directions:
        if d == last:
            streak += 1
        else:
            last, streak = d, 1
        out.append(d if (d == "neutral" or streak >= confirm) else "neutral")
    return pd.Series(out, index=directions.index)


def rotation_backtest(score: pd.DataFrame, board_close: pd.DataFrame,
                      *, top_n: int = 3, cost_bps: float = 5.0,
                      freq: str = "W-FRI", direction: str = "top") -> dict:
    """周度轮动：每周最后交易日收盘调仓，等权持有 N 个板块。

    执行口径（T+1）：情绪分收盘后才可得，新持仓从次一交易日起享受收益，
    调仓当日收益仍归旧持仓——不用收盘后的分数赚当天的钱（前视）。
    调仓日取每周真实的最后交易日（周五遇节假日自动提前到周四），
    不会像 resample("W-FRI") 的标签那样落在非交易日导致该周静默不调仓。

    direction: "top"=买入情绪分最高 N 个（追涨），"bottom"=最低 N 个（抄底），
              "avoid_top"=等权持有除最高 N 个外的全部板块（回避亢奋）。
    成本：换仓部分按 cost_bps（万分之）计一次往返成本（卖+买合并计一次，
    口径为"每次换仓成本"而非"单边"）。
    """
    days_with_score = score.dropna(how="all").index
    rebal_dates = rebalance_dates(days_with_score, freq=freq)
    daily_ret = board_close.pct_change(fill_method=None)
    cost = cost_bps / 10000.0

    holdings: list[str] = []
    nav, bench_nav = 1.0, 1.0
    nav_series, bench_series = {}, {}
    turnover_cost_total = 0.0
    turnover_total = 0.0
    current_weights = pd.Series(dtype=float)

    all_days = daily_ret.index
    rebal_set = set(rebal_dates)

    for dt in all_days:
        # 当日收益先按调仓前的持仓结算
        if holdings:
            day_ret = daily_ret.loc[dt, holdings].mean()
            if np.isfinite(day_ret):
                nav *= 1 + day_ret
        bench_day = daily_ret.loc[dt].mean()
        if np.isfinite(bench_day):
            bench_nav *= 1 + bench_day
        # 收盘后用当日情绪分调仓，新持仓次一交易日起生效
        if dt in rebal_set:
            s = score.loc[dt].dropna()
            # 只考虑有行情数据的板块：有分无价（如 K 线抓取失败）的板块不可交易
            s = s[s.index.isin(daily_ret.columns)]
            if len(s) >= top_n:
                if direction == "top":
                    pick = s.nlargest(top_n)
                elif direction == "bottom":
                    pick = s.nsmallest(top_n)
                else:  # avoid_top
                    pick = s.drop(s.nlargest(top_n).index)
                new_holdings = pick.index.tolist()
                target_weights = equal_weight_portfolio(new_holdings)
                if holdings:
                    turnover = portfolio_turnover(current_weights, target_weights)
                    tc = cost * turnover
                    nav *= 1 - tc
                    turnover_cost_total += tc
                    turnover_total += turnover
                holdings = new_holdings
                current_weights = target_weights
        nav_series[dt] = nav
        bench_series[dt] = bench_nav

    nav_s = pd.Series(nav_series).sort_index()
    bench_s = pd.Series(bench_series).sort_index()

    return {
        "nav": nav_s,
        "benchmark_nav": bench_s,
        "strategy": performance_stats(nav_s),
        "benchmark": performance_stats(bench_s),
        "total_cost_drag": turnover_cost_total,
        "total_turnover": turnover_total,
        "final_nav": nav_s.iloc[-1] if len(nav_s) else np.nan,
        "final_benchmark": bench_s.iloc[-1] if len(bench_s) else np.nan,
    }


FACTOR_LABELS = {
    "limit_up_strength": "涨停强度",
    "money_effect": "赚钱效应",
    "breadth": "涨跌宽度",
    "rps": "相对强度",
    "turnover_heat": "热度变化",
    "promotion_rate": "晋级率",
}


def factor_ic_snapshot(factors: dict[str, pd.DataFrame], board_close: pd.DataFrame,
                       *, windows: tuple[int, ...] = (120, 60, 30),
                       horizon: int = 5,
                       bootstrap_samples: int = 500) -> pd.DataFrame:
    """逐因子 IC 快照：多窗口 IC/胜率 + 方向是否可信。

    为什么需要：日报的健康度只看**合成后**的 IC，一个因子崩掉会被其余因子
    掩盖（2026-09 实测：合成分近 60 日 IC -0.09，同期 limit_up_strength 已到
    -0.083 且 bootstrap CI 完全不跨零）。拆到因子层才能区分三件事：
      1. 该因子方向可信（CI 不跨零）——可据此调整用法；
      2. 只是与全期反向（风格切换）——**不可**据此取反，那是拟合最近行情；
      3. 各窗口都跨零——该因子本期没有可辨认的方向。

    "可信"判定用最短窗口的移动区块 bootstrap 95% CI 是否排除 0，而不是比较
    IC 均值大小——后者在 30~60 个重叠样本上极易被几天行情主导。
    """
    rows = []
    short_w = min(windows)
    for name, f in factors.items():
        if f is None or getattr(f, "empty", True):
            continue
        ic = daily_ic_series(f, board_close, horizon)
        if ic.empty:
            continue
        rec = {"factor": name, "label": FACTOR_LABELS.get(name, name),
               "ic_all": float(ic.mean()), "n": int(len(ic))}
        for w in windows:
            sub = ic.tail(w)
            rec[f"ic_{w}"] = float(sub.mean()) if len(sub) else np.nan
            rec[f"win_{w}"] = float((sub > 0).mean()) if len(sub) else np.nan
        rob_short = ic_robustness(ic.tail(short_w), horizon=horizon,
                                  bootstrap_samples=bootstrap_samples)
        rec["hac_short"] = rob_short["IC_HAC_t"]
        rec["ci_low"] = rob_short["IC_bootstrap_low"]
        rec["ci_high"] = rob_short["IC_bootstrap_high"]
        rec["window"] = short_w

        ic_short = rec.get(f"ic_{short_w}", np.nan)
        lo, hi = rec["ci_low"], rec["ci_high"]
        if pd.notna(lo) and pd.notna(hi) and hi < 0:
            rec["verdict"] = "negative_robust"
        elif pd.notna(lo) and pd.notna(hi) and lo > 0:
            rec["verdict"] = "positive_robust"
        elif (pd.notna(ic_short) and pd.notna(rec["ic_all"])
              and ic_short != 0 and rec["ic_all"] != 0
              and np.sign(ic_short) != np.sign(rec["ic_all"])):
            rec["verdict"] = "regime_flip"
        else:
            rec["verdict"] = "no_direction"
        rows.append(rec)
    return pd.DataFrame(rows)


def recent_signal_health(score: pd.DataFrame, board_close: pd.DataFrame,
                         *, days: int = 60, horizon: int = 5) -> dict:
    """最近 days 个交易日的信号健康度：滚动 IC 均值与胜率，附带分层价差。

    用于 regime 感知——IC 持续走弱/转负说明当前风格下信号正在失效。
    分层价差（Q5−Q1 平均前瞻收益）用于区分"信号反"（价差<0，分组单调倒置，
    低分组占优）与"信号死"（价差≈0，分组无结构）——IC 均值无法分辨二者。
    """
    ic = daily_ic_series(score, board_close, horizon).tail(days)
    if ic.empty:
        return {"ic_mean": float("nan"), "ic_win_rate": float("nan"), "n": 0}
    robust = ic_robustness(ic, horizon=horizon)
    out = {"ic_mean": ic.mean(), "ic_win_rate": (ic > 0).mean(), "n": len(ic),
           **robust}
    spread = quantile_spread_series(score, board_close, horizon).tail(days)
    if len(spread):
        out["spread_mean"] = float(spread.mean())
        out["spread_neg_share"] = float((spread < 0).mean())
    return out


def daily_ic_series(score: pd.DataFrame, board_close: pd.DataFrame,
                    horizon: int = 5, min_pairs: int = 10) -> pd.Series:
    """每日截面 IC 序列：当日情绪分截面 vs 前瞻 horizon 日收益截面的 Spearman 相关。"""
    fwd = forward_returns(board_close, horizon)
    ics = {}
    for dt in score.index.intersection(fwd.index):
        pair = pd.concat([score.loc[dt], fwd.loc[dt]], axis=1,
                         keys=["s", "f"]).dropna()
        if len(pair) < min_pairs:
            continue
        c = pair["s"].corr(pair["f"], method="spearman")
        if pd.notna(c):
            ics[dt] = c
    return pd.Series(ics).sort_index()


def _hac_mean_stats(values: pd.Series, lag: int) -> tuple[float, float]:
    """均值的 Newey-West(HAC) t 值与正态近似双侧 p 值。"""
    x = values.dropna().to_numpy(dtype=float)
    n = len(x)
    if n < 3:
        return float("nan"), float("nan")
    lag = min(max(int(lag), 0), n - 1)
    centered = x - x.mean()
    long_run_var = float(np.dot(centered, centered) / n)
    for k in range(1, lag + 1):
        gamma = float(np.dot(centered[k:], centered[:-k]) / n)
        long_run_var += 2 * (1 - k / (lag + 1)) * gamma
    if long_run_var <= 0:
        return float("nan"), float("nan")
    t_value = float(x.mean() / math.sqrt(long_run_var / n))
    p_value = float(math.erfc(abs(t_value) / math.sqrt(2)))
    return t_value, p_value


def _moving_block_bootstrap_interval(values: pd.Series, *, block_size: int,
                                     samples: int = 1_000, seed: int = 20260825
                                     ) -> tuple[float, float]:
    """保留时间相关性的移动区块 bootstrap 均值 95% 区间。"""
    x = values.dropna().to_numpy(dtype=float)
    n = len(x)
    if n < 3 or samples < 1:
        return float("nan"), float("nan")
    block_size = min(max(int(block_size), 1), n)
    starts = max(n - block_size + 1, 1)
    blocks_needed = math.ceil(n / block_size)
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=float)
    for i in range(samples):
        offsets = rng.integers(0, starts, size=blocks_needed)
        sample = np.concatenate([x[start:start + block_size] for start in offsets])[:n]
        means[i] = sample.mean()
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def ic_robustness(ic: pd.Series, *, horizon: int,
                  bootstrap_samples: int = 1_000) -> dict:
    """IC 均值的重叠样本稳健性统计。

    ``IC_nonoverlap_mean`` 取间隔 ``horizon`` 天的一条固定非重叠分区；
    ``IC_HAC_*`` 与区块 bootstrap 使用完整序列但校正其自相关。三项同向时，
    才比朴素 IR 更有说服力。
    """
    clean = ic.dropna()
    block = max(int(horizon), 1)
    non_overlap = clean.iloc[::block]
    hac_t, hac_p = _hac_mean_stats(clean, lag=block - 1)
    boot_low, boot_high = _moving_block_bootstrap_interval(
        clean, block_size=block, samples=bootstrap_samples)
    return {
        "IC_nonoverlap_mean": non_overlap.mean(),
        "IC_nonoverlap_n": int(len(non_overlap)),
        "IC_HAC_t": hac_t,
        "IC_HAC_p": hac_p,
        "IC_bootstrap_low": boot_low,
        "IC_bootstrap_high": boot_high,
    }


def rolling_ic_validation(ic: pd.Series, *, block_days: int = 20) -> pd.DataFrame:
    """按时间顺序切分已实现 IC，输出不可重叠的滚动验证块。

    该函数不选择参数；它只回答“此前的有效性是否在随后时间段仍出现”。
    末块不足 ``block_days`` 时保留并标注样本数，避免悄悄丢掉最近市场状态。
    """
    if block_days < 2:
        raise ValueError("block_days 必须 >= 2")
    clean = ic.dropna().sort_index()
    rows = []
    for start in range(0, len(clean), block_days):
        chunk = clean.iloc[start:start + block_days]
        if chunk.empty:
            continue
        rows.append({
            "start": chunk.index[0],
            "end": chunk.index[-1],
            "n_obs": int(len(chunk)),
            "IC_mean": chunk.mean(),
            "IC_win_rate": (chunk > 0).mean(),
            "previous_IC_mean": clean.iloc[max(0, start - block_days):start].mean(),
        })
    return pd.DataFrame(rows)


def regime_state(score: pd.DataFrame, board_close: pd.DataFrame, *,
                 ic_window: int = 15, threshold: float = 0.02,
                 horizon: int = 5, min_obs: int = 10,
                 guard_days: int = 10, guard_threshold: float | None = -0.03,
                 vol_window: int = 20, vol_threshold: float | None = None,
                 require_robust_signal: bool = False,
                 robust_ic_window: int = 60,
                 robust_bootstrap_samples: int = 500,
                 mono_days: int | None = None,
                 ic_known: pd.Series | None = None,
                 spread_known: pd.Series | None = None,
                 ew_halflife: float | None = None,
                 market_return: float | None = None,
                 market_volatility: float | None = None) -> dict:
    """当前市场风格判定（regime 自适应，无前视）。

    滚动 IC 使用"已走完前瞻窗口"的样本（IC(d) 在 d+horizon 收盘后才可知，
    序列整体后移 horizon 个交易日）；窗口内均值越阈值判方向：
      IC > +threshold  -> "top"（动量市，榜单顺势用）
      IC < -threshold  -> "bottom"（反转市，榜单反向用）
      中间             -> "neutral"（观望）
    回撤护栏：市场等权近 guard_days 日累计收益 < guard_threshold 时强制观望。
    波动闸门（可选）：近 vol_window 日日波动 > vol_threshold 时强制观望。
    稳健性闸门（可选）：当前方向的滚动已实现 IC 经移动区块 bootstrap 后，
    95% 区间必须完全位于 0 的同侧；否则只报告观察结论、强制观望。
    单调性破坏预警（可选，mono_days）：IC 均值是窗口平均、翻负慢；分层价差
    连续 mono_days 个可知日反向（追涨期 Q5<Q1 / 抄底期 Q5>Q1）说明分组结构
    已倒置，立即降级观望——通常比 IC 均值翻负早约一周。只降级、不提前给方向。

    ic_known / spread_known：可复用预计算的"可知时点"IC / 分层价差序列
    （省去重算，也便于测试注入），可选。
    参数由 config.yaml 预先声明；策略效果须用 ``scripts/validate_robustness.py``
    的重叠校正和后续时间块持续检验，不能沿用旧网格扫描的样本内最优结果。
    """
    if ic_known is None:
        ic_s = daily_ic_series(score, board_close, horizon)
        if ic_s.empty:
            ic_known = pd.Series(dtype=float)
        else:
            ic_known = realized_series(ic_s, board_close.index, lag=horizon)
    asof = board_close.index[-1]
    ic_known = ic_known.loc[:asof].dropna()
    hist = ic_known.tail(ic_window)

    eq_ret = board_close.pct_change(fill_method=None).mean(axis=1)
    mkt_recent = eq_ret.tail(guard_days).sum() if market_return is None else market_return
    guard_hit = (guard_threshold is not None and pd.notna(mkt_recent)
                 and mkt_recent < guard_threshold)

    vol_now = eq_ret.tail(vol_window).std() if market_volatility is None else market_volatility
    vol_gate = (vol_threshold is not None and pd.notna(vol_now)
                and vol_now > vol_threshold)

    if len(hist) < min_obs:
        direction, ic_mean, ic_win = "neutral", float("nan"), float("nan")
    else:
        ic_mean = (hist.ewm(halflife=ew_halflife).mean().iloc[-1]
                   if ew_halflife else hist.mean())
        ic_win = (hist > 0).mean()
        if ic_mean > threshold:
            direction = "top"
        elif ic_mean < -threshold:
            direction = "bottom"
        else:
            direction = "neutral"
    raw_direction = direction
    robust = ic_robustness(ic_known.tail(robust_ic_window), horizon=horizon,
                           bootstrap_samples=robust_bootstrap_samples)
    robust_gate = False
    if require_robust_signal and direction in {"top", "bottom"}:
        low, high = robust["IC_bootstrap_low"], robust["IC_bootstrap_high"]
        confirmed = ((direction == "top" and pd.notna(low) and low > 0)
                     or (direction == "bottom" and pd.notna(high) and high < 0))
        robust_gate = not confirmed

    # 单调性破坏预警：分层价差（Q5−Q1）连续 mono_days 个可知日与当前方向
    # 相反 → 分组结构已倒置，立即降级观望。与 IC 同样后移 horizon 到可知时点。
    mono_gate = False
    spread_last = float("nan")
    if mono_days and direction in {"top", "bottom"}:
        if spread_known is None:
            sp = quantile_spread_series(score, board_close, horizon)
            if sp.empty:
                spread_known = pd.Series(dtype=float)
            else:
                spread_known = realized_series(sp, board_close.index, lag=horizon)
        tail = spread_known.loc[:asof].dropna().tail(mono_days)
        if len(tail) >= mono_days:
            spread_last = float(tail.iloc[-1])
            counter = ((tail < 0).all() if direction == "top"
                       else (tail > 0).all())
            mono_gate = bool(counter)

    if guard_hit or vol_gate or robust_gate or mono_gate:
        direction = "neutral"

    return {
        "direction": direction,          # top / bottom / neutral
        "raw_direction": raw_direction,  # IC 阈值方向，供周度确认使用
        "ic_mean": ic_mean,              # 窗口内已实现 IC 均值
        "ic_win_rate": ic_win,
        "n_obs": len(hist),
        "guard_hit": bool(guard_hit),    # 回撤护栏触发
        "vol_gate": bool(vol_gate),      # 波动闸门触发
        "robust_gate": bool(robust_gate),  # IC 置信区间不足以确认方向
        "mono_gate": mono_gate,          # 单调性破坏预警触发（提前降级）
        "spread_last": spread_last,      # 最近一个可知日的分层价差（Q5−Q1）
        "mkt_recent": mkt_recent,        # 市场等权近 guard_days 日累计收益
        "vol_now": vol_now,              # 近 vol_window 日日波动
        "robust_ic_low": robust["IC_bootstrap_low"],
        "robust_ic_high": robust["IC_bootstrap_high"],
        "robust_ic_n": robust["IC_nonoverlap_n"],
    }


def realized_series(values: pd.Series, calendar: pd.DatetimeIndex,
                    lag: int = 5) -> pd.Series:
    """d 日前瞻统计只在同一交易日历的 d+lag 收盘后可知。"""
    if values.empty:
        return pd.Series(dtype=float)
    positions = calendar.get_indexer(values.index)
    if (positions < 0).any():
        raise ValueError("前瞻统计日期不在价格交易日历中")
    known = positions + lag
    keep = known < len(calendar)
    return pd.Series(values.to_numpy()[keep], index=calendar[known[keep]])


def regime_parameters(config: dict) -> dict:
    """日报、看板和回测共用的参数读取；不修改生产配置。"""
    defaults = {"ic_window": 15, "threshold": 0.02, "horizon": 5, "min_obs": 10,
                "guard_days": 10, "guard_threshold": -0.03, "vol_window": 20,
                "vol_threshold": None, "require_robust_signal": False,
                "robust_ic_window": 60, "robust_bootstrap_samples": 500,
                "mono_days": None}
    return {key: config.get(key, value) for key, value in defaults.items()}


def regime_decisions(score: pd.DataFrame, board_close: pd.DataFrame, *,
                     confirm: int = 1, min_pick: int = 3,
                     ic_known: pd.Series | None = None,
                     spread_known: pd.Series | None = None,
                     market_returns: pd.Series | None = None,
                     market_volatility: pd.Series | None = None,
                     **parameters) -> pd.DataFrame:
    """周度决策共用 regime_state，先确认原始方向，所有闸门即时降级。

    state_direction 是与当日报告相同的单日判定；direction 是经过连续
    confirm 次周度 IC 读数确认后的调仓方向。风控闸门不受确认迟滞影响。
    """
    horizon = parameters.get("horizon", 5)
    if ic_known is None:
        ic_known = realized_series(daily_ic_series(score, board_close, horizon),
                                   board_close.index, lag=horizon)
    if spread_known is None and parameters.get("mono_days"):
        spread_known = realized_series(quantile_spread_series(score, board_close, horizon),
                                       board_close.index, lag=horizon)
    rows = []
    for dt in rebalance_dates(score.dropna(how="all").index):
        available = score.loc[dt].reindex(board_close.columns).dropna()
        if dt not in board_close.index or len(available) < min_pick:
            continue
        state = regime_state(
            score.loc[:dt], board_close.loc[:dt], ic_known=ic_known,
            spread_known=spread_known,
            market_return=(market_returns.get(dt, np.nan) if market_returns is not None else None),
            market_volatility=(market_volatility.get(dt, np.nan) if market_volatility is not None else None),
            **parameters)
        rows.append({"date": dt, **state, "state_direction": state["direction"]})
    if not rows:
        return pd.DataFrame(columns=["direction", "raw_direction", "state_direction"],
                            index=pd.DatetimeIndex([], name="date"))
    decisions = pd.DataFrame(rows).set_index("date")
    decisions["confirmed_direction"] = apply_confirm_hysteresis(
        decisions["raw_direction"], confirm=confirm)
    blocked = decisions[["guard_hit", "vol_gate", "robust_gate", "mono_gate"]].any(axis=1)
    decisions["direction"] = decisions["confirmed_direction"].where(~blocked, "neutral")
    return decisions


def regime_snapshot(score: pd.DataFrame, board_close: pd.DataFrame,
                    config: dict, *, top_n: int = 3) -> dict:
    """当日风格与最近完成的周度调仓判定；两者明确区分。"""
    parameters = regime_parameters(config)
    horizon = parameters["horizon"]
    ic_known = realized_series(daily_ic_series(score, board_close, horizon),
                               board_close.index, lag=horizon)
    spread_known = realized_series(quantile_spread_series(score, board_close, horizon),
                                   board_close.index, lag=horizon)
    state = regime_state(score, board_close, ic_known=ic_known,
                         spread_known=spread_known, **parameters)
    confirm = config.get("confirm_readings", 1)
    decisions = regime_decisions(score, board_close, confirm=confirm, min_pick=top_n,
                                 ic_known=ic_known, spread_known=spread_known, **parameters)
    state.update({"ic_window": parameters["ic_window"], "guard_days": parameters["guard_days"],
                  "mono_days": parameters["mono_days"], "confirm_readings": confirm,
                  "rebalance_date": decisions.index[-1].strftime("%Y-%m-%d") if len(decisions) else None,
                  "rebalance_direction": decisions.iloc[-1]["direction"] if len(decisions) else "neutral"})
    return state


def _report_body(ic_df: pd.DataFrame, quant_df: pd.DataFrame,
                 rot_top: dict, rot_bottom: dict,
                 rot_avoid: dict) -> list[str]:
    """回测结果正文（不含报告标题），单窗口与多窗口报告共用。"""
    lines = ["## IC 分析（Spearman）", "",
             "| 前瞻窗口 | IC均值 | IR（重叠） | HAC t | Bootstrap 95% CI | IC胜率 | 样本数/非重叠 |",
             "|---|---|---|---|---|---|---|"]
    for _, r in ic_df.iterrows():
        ci = (f"[{r['IC_bootstrap_low']:.4f}, {r['IC_bootstrap_high']:.4f}]"
              if pd.notna(r.get("IC_bootstrap_low")) else "—")
        t = f"{r['IC_HAC_t']:.2f}" if pd.notna(r.get("IC_HAC_t")) else "—"
        non = int(r.get("IC_nonoverlap_n", 0))
        lines.append(f"| {int(r['horizon'])}日 | {r['IC_mean']:.4f} | {r['IR']:.3f} "
                     f"| {t} | {ci} | {r['IC_win_rate']:.1%} | {int(r['n_obs'])}/{non} |")
    lines += ["", "## 分层测试（5日前瞻收益，按情绪分分组）", "",
              "| 分组 | 平均前瞻收益 | 样本数 |", "|---|---|---|"]
    for g, r in quant_df.iterrows():
        lines.append(f"| Q{int(g) + 1} | {r['mean_fwd_ret']:.3%} | {int(r['n_obs'])} |")

    def rot_rows(rot: dict) -> list[str]:
        s, b = rot["strategy"], rot["benchmark"]
        return [
            f"| 累计收益 | {s['total_ret']:.2%} | {b['total_ret']:.2%} |",
            f"| 年化收益（仅供参考） | {annual_return_text(s)} | {annual_return_text(b)} |",
            f"| 夏普 | {s['sharpe']:.2f} | {b['sharpe']:.2f} |",
            f"| 最大回撤 | {s['max_dd']:.2%} | {b['max_dd']:.2%} |",
            f"| 期末净值 | {rot['final_nav']:.3f} | {rot['final_benchmark']:.3f} |",
        ]

    lines += ["", "## 周度轮动：买入情绪 Top N（追涨）", "",
              "| 指标 | 策略 | 板块等权参考（非可交易基准） |", "|---|---|---|"] + rot_rows(rot_top)
    lines += ["", "## 周度轮动：买入情绪 Bottom N（抄底）", "",
              "| 指标 | 策略 | 板块等权参考（非可交易基准） |", "|---|---|---|"] + rot_rows(rot_bottom)
    lines += ["", "## 周度轮动：回避情绪 Top N（持有其余全部）", "",
              "| 指标 | 策略 | 板块等权参考（非可交易基准） |", "|---|---|---|"] + rot_rows(rot_avoid)
    lines += ["", f"换仓成本损耗——追涨：{rot_top['total_cost_drag']:.2%}；"
              f"抄底：{rot_bottom['total_cost_drag']:.2%}；"
              f"回避：{rot_avoid['total_cost_drag']:.2%}",
              "",
              "> IC 前瞻收益彼此重叠，IR 与日频样本数会偏乐观；优先检查 HAC t、"
              "移动区块 bootstrap 区间和后续滚动验证块，不能仅凭 IC 均值挑参数。"]
    return lines


def _window_label(window_days: int | None) -> str:
    return f"最近 {window_days} 个交易日" if window_days else "全样本"


def format_backtest_report(ic_df: pd.DataFrame, quant_df: pd.DataFrame,
                           rot_top: dict, rot_bottom: dict,
                           rot_avoid: dict,
                           window_days: int | None = None) -> str:
    """回测结果 -> Markdown 文本（含追涨/抄底两个方向的轮动对照）。"""
    title = "# 板块情绪分回测报告"
    if window_days:
        title += f"（{_window_label(window_days)}）"
    return "\n".join([title, ""]
                     + _report_body(ic_df, quant_df, rot_top, rot_bottom,
                                    rot_avoid))


def format_multi_window_report(results: dict, forward_days: list[int]) -> str:
    """多档回测窗口结果 -> Markdown：先给窗口横向对照总览，再附各窗口明细。

    results: {window_days: {"ic": df, "quant": df, "top": rot, "bottom": rot,
                            "avoid": rot}}，window_days 为 None 表示全样本。
    """
    h0 = forward_days[0]
    lines = ["# 板块情绪分回测报告（多窗口验证）", "",
             "## 窗口总览（周度轮动，含 5bps 换仓成本）", "",
             "| 回测窗口 | 追涨累计 | 追涨年化 | 追涨夏普 | 追涨回撤 | 抄底累计 | 基准累计 "
             f"| IC({h0}日)均值 | IC({h0}日)胜率 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for w, res in results.items():
        t, b = res["top"], res["bottom"]
        ic_row = res["ic"].iloc[0] if len(res["ic"]) else None
        ic_mean = f"{ic_row['IC_mean']:.4f}" if ic_row is not None else "—"
        ic_win = f"{ic_row['IC_win_rate']:.1%}" if ic_row is not None else "—"
        lines.append(
            f"| {_window_label(w)} | {t['strategy']['total_ret']:.2%} "
            f"| {annual_return_text(t['strategy'])} "
            f"| {t['strategy']['sharpe']:.2f} | {t['strategy']['max_dd']:.2%} "
            f"| {b['strategy']['total_ret']:.2%} "
            f"| {t['benchmark']['total_ret']:.2%} | {ic_mean} | {ic_win} |")
    lines += ["",
              "> 多窗口口径一致（同一组情绪分），仅截取的样本区间不同；"
              "各窗口结论方向一致才说明信号稳健，单窗口拔尖多为区间运气。"]
    for w, res in results.items():
        lines += ["", "---", "", f"## 明细：{_window_label(w)}", ""]
        body = _report_body(res["ic"], res["quant"], res["top"],
                            res["bottom"], res["avoid"])
        # 明细区块内的二级标题降级为三级，保持文档层级
        lines += [ln.replace("## ", "### ", 1) if ln.startswith("## ")
                  else ln for ln in body]
    return "\n".join(lines)
