"""信号层：从情绪分矩阵生成板块切换信号。规则透明，非预测黑盒。

P0→P1 突变穿透规则（2026-08-05 升级）：
  - bottom_reversal            冰点 + 流量价格代理 z > 阈值 → 底部反转预警
  - extreme_hot                亢奋 + 单因子偏离 σ → 顺势加速度（动量市买点）
  - extreme_cold               冰点 + 单因子偏离 σ → 极端冷区异动（动量市回避/观察）
  - fund_emotion_divergence    资金连续 N 日为正 + 情绪分为负 → 资金-情绪背离
                               （regime 自适应：顺势=风险预警，逆势=反转机会）

三条规则均在信号层独立判定，**不进入合成情绪分**（不改因子体系）、**不经平滑**
（不改 smooth_window）——平滑会抹平突变，穿透信号必须用原始日度读数。
flow_proxy 为信号层穿透指标，口径为 amount * pct_chg / 100 后按板块求和。
它衡量成交额加权的涨跌方向，**不是**买卖盘意义上的资金净流入；真实资金流
数据未接入前，所有展示都必须明确标为“代理”。
"""

from __future__ import annotations

import pandas as pd

from .score import zscore_cross_section


def _consecutive_trend(s: pd.Series, days: int, direction: str) -> bool:
    """序列尾部是否连续 days 天单调上升/下降（NaN 视为中断）。"""
    tail = s.dropna().tail(days + 1)
    if len(tail) < days + 1:
        return False
    diff = tail.diff().dropna()
    if direction == "up":
        return bool((diff > 0).all())
    return bool((diff < 0).all())


def compute_board_flow_proxy(stock_daily: pd.DataFrame,
                             board_map: pd.DataFrame) -> pd.DataFrame:
    """板块日度成交额×涨跌幅代理（信号层穿透指标，不进入合成情绪分）。

    个股 flow_proxy = amount * pct_chg / 100（amount 单位为元，pct_chg 为百分比），
    按 (date, board_code) 聚合。它是成交额加权价格变动，不包含主动买卖方向，
    因而不应被称为“资金净流入”。仅计入当日交易股（trade_status=='1'），
    剔除停牌零成交。

    返回 date × board_code 矩阵；该矩阵只供 P0 穿透规则使用，不会被
    score.combine_factors 消费——故不构成对因子体系的改动。
    """
    d = stock_daily.copy()
    if "trade_status" in d.columns:
        d = d[d["trade_status"].astype(str) == "1"]
    # 一股多板：与 factors._with_board 同口径，只按 (code, board_code) 对去重，
    # 否则细分板块的资金流会整板缺失
    m = board_map[["code", "board_code"]].drop_duplicates()
    d = d.merge(m, on="code", how="inner")
    d = d.dropna(subset=["amount", "pct_chg"])
    d = d[d["amount"] > 0]
    d["flow_proxy"] = d["amount"] * d["pct_chg"] / 100.0
    g = (d.groupby(["date", "board_code"])["flow_proxy"].sum()
         .reset_index())
    return g.pivot(index="date", columns="board_code", values="flow_proxy")


def compute_board_fund_flow(stock_daily: pd.DataFrame,
                            board_map: pd.DataFrame) -> pd.DataFrame:
    """兼容旧调用的别名；返回的仍是 ``compute_board_flow_proxy`` 代理量。

    新代码请使用 :func:`compute_board_flow_proxy`，避免将价格成交代理误读为
    真实资金净流入。
    """
    return compute_board_flow_proxy(stock_daily, board_map)


def detect_signals(smoothed: pd.DataFrame, pct: pd.DataFrame,
                   *, consecutive_days: int = 3, hot_quantile: float = 0.9,
                    cold_quantile: float = 0.1, recent_top_days: int = 20,
                    recent_top_k: int = 10,
                    factors: dict[str, pd.DataFrame] | None = None,
                    flow_proxy: pd.DataFrame | None = None,
                    fund_flow: pd.DataFrame | None = None,
                   breakthrough_cfg: dict | None = None) -> dict[str, list[str]]:
    """生成信号。

    返回:
      heating   升温榜：情绪分连续 N 日上升
      cooling   降温榜：情绪分连续 N 日下降
      hot       亢奋：时序分位 > hot_quantile（不宜追高）
      cold      冰点：时序分位 < cold_quantile（观察反转机会）
      switching 切换预警：近期曾居前 K 的板块正在降温 且 有低位板块升温
      bottom_reversal          底部反转预警：冰点 + 流量价格代理 z > 阈值
      extreme_hot              顺势加速度：亢奋 + 单因子偏离 σ（动量市顺势买点）
      extreme_cold             极端冷区异动：冰点 + 单因子偏离 σ（动量市建议回避）
      fund_emotion_divergence  流量代理-情绪背离：代理连续 N 日为正 + 情绪分为负
                               （breakthrough_cfg 控制天数；report 层根据 IC regime 定语义）

    P0 穿透规则仅在传入 factors/flow_proxy 时计算；不传则三项返回空列表，
    合成情绪分与平滑窗口不受影响。breakthrough_cfg 可调阈值，缺省取 P0 值：
    flow_proxy_z_threshold=2.0、extreme_sigma=2.0、flow_positive_days=2。
    ``fund_flow`` 仅为旧调用兼容别名，不会把代理变成真实资金流。
    """
    out: dict[str, list[str]] = {
        "heating": [], "cooling": [], "hot": [], "cold": [], "switching": [],
        "bottom_reversal": [], "extreme_hot": [], "extreme_cold": [],
        "fund_emotion_divergence": [],
    }
    if smoothed.empty:
        return out

    last_pct = pct.iloc[-1]
    heating, cooling = [], []
    for board in smoothed.columns:
        s = smoothed[board]
        if _consecutive_trend(s, consecutive_days, "up"):
            heating.append(board)
        if _consecutive_trend(s, consecutive_days, "down"):
            cooling.append(board)

    hot = last_pct[last_pct > hot_quantile].index.tolist()
    cold = last_pct[last_pct < cold_quantile].index.tolist()

    # 近期（recent_top_days 内任一交易日）情绪分曾进入全市场前 K
    recent = smoothed.tail(recent_top_days)
    was_top = set()
    for _, row in recent.iterrows():
        was_top.update(row.dropna().nlargest(recent_top_k).index)
    former_leaders_cooling = sorted(was_top & set(cooling))
    low_heating = sorted(
        b for b in heating
        if pd.notna(last_pct.get(b)) and last_pct[b] < 0.5
    )
    switching = []
    if former_leaders_cooling and low_heating:
        switching = [f"{b}↓ → {','.join(low_heating[:5])}↑" for b in former_leaders_cooling]

    def _by_score(boards: list[str]) -> list[str]:
        last = smoothed.iloc[-1]
        return sorted(boards, key=lambda b: last.get(b, float("nan")),
                      reverse=True)

    out["heating"] = _by_score(heating)
    out["cooling"] = _by_score(cooling)
    out["hot"] = _by_score(hot)
    out["cold"] = _by_score(cold)
    out["switching"] = switching

    # ---- P0 突变穿透规则（信号层独立判定，不进合成情绪分、不经平滑）----
    cfg = breakthrough_cfg or {}
    if not cfg.get("enabled", True):
        return out
    if flow_proxy is None:
        flow_proxy = fund_flow
    if factors is None and flow_proxy is None:
        return out

    last_date = smoothed.index[-1]
    last_score = smoothed.iloc[-1]
    cold_set, hot_set = set(cold), set(hot)

    # 规则1：冰点 + 代理量横截面异常 → 底部反转预警
    if flow_proxy is not None and not flow_proxy.empty:
        thr1 = float(cfg.get("flow_proxy_z_threshold",
                             cfg.get("fund_flow_z_threshold", 2.0)))
        ffz = zscore_cross_section(flow_proxy)
        last_ffz = ffz.loc[last_date] if last_date in ffz.index else pd.Series(dtype=float)
        bottom_reversal = [
            b for b in cold
            if pd.notna(last_ffz.get(b)) and last_ffz[b] > thr1
        ]
        out["bottom_reversal"] = _by_score(bottom_reversal)

    # 规则2：极端分位 + 单因子偏离 σ → 拆分热端/冷端
    sigma = float(cfg.get("extreme_sigma", 2.0))
    sigma_hot = float(cfg.get("extreme_hot_sigma", sigma))
    sigma_cold = float(cfg.get("extreme_cold_sigma", sigma))
    if factors and (hot_set or cold_set):
        last_z: dict[str, pd.Series] = {}
        for name, f in factors.items():
            if f is None or f.empty:
                continue
            z = zscore_cross_section(f)
            if last_date in z.index:
                last_z[name] = z.loc[last_date]
        # 热端：亢奋 + 单因子偏离 → 顺势加速度（动量市买点）
        extreme_hot: list[str] = []
        if hot_set and last_z:
            for b in hot_set:
                for lz in last_z.values():
                    v = lz.get(b)
                    if pd.notna(v) and abs(v) > sigma_hot:
                        extreme_hot.append(b)
                        break
        out["extreme_hot"] = _by_score(extreme_hot)
        # 冷端：冰点 + 单因子偏离 → 极端冷区异动（动量市回避/观察）
        extreme_cold: list[str] = []
        if cold_set and last_z:
            for b in cold_set:
                for lz in last_z.values():
                    v = lz.get(b)
                    if pd.notna(v) and abs(v) > sigma_cold:
                        extreme_cold.append(b)
                        break
        out["extreme_cold"] = _by_score(extreme_cold)

    # 规则3：代理连续 N 日为正 + 情绪分为负 → 代理-情绪背离
    if flow_proxy is not None and not flow_proxy.empty:
        ndays = int(cfg.get("flow_positive_days", cfg.get("fund_positive_days", 2)))
        ff = flow_proxy.reindex(columns=smoothed.columns)
        tail = ff.tail(ndays)
        divergence: list[str] = []
        if len(tail) == ndays:
            pos_all = (tail > 0).all(axis=0)
            valid_all = tail.notna().all(axis=0)
            for b in smoothed.columns:
                if (pos_all.get(b, False) and valid_all.get(b, False)
                        and pd.notna(last_score.get(b)) and last_score[b] < 0):
                    divergence.append(b)
        out["fund_emotion_divergence"] = _by_score(divergence)

    return out
