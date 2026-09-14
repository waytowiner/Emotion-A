"""每日报告生成：Markdown + CSV。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd


def build_report(date: pd.Timestamp, smoothed: pd.DataFrame, pct: pd.DataFrame,
                 signals: dict[str, list[str]], board_names: dict[str, str],
                 *, top_n: int = 15, health: dict | None = None,
                 regime: dict | None = None,
                 breakthrough: dict | None = None,
                 universe: dict | None = None,
                 factor_ic: pd.DataFrame | None = None) -> str:
    """生成当日 Markdown 报告文本。board_names: board_code -> 名称。

    health: recent_signal_health 的输出（最近 N 日 IC 监控），可选。
    regime: backtest.regime_state 的输出（风格判定 + 方向建议），可选；
            提供时榜单语义按方向自动改写（Top 榜在反转市标注为回避等）。
    breakthrough: signal.breakthrough 配置，用于让报告阈值与实际信号判定一致。
    universe: {"board_source": ..., "n_boards": ...}，用于在报告头标注板块口径。
    factor_ic: backtest.factor_ic_snapshot 的输出（逐因子 IC 与可信度），可选。
    """
    last_score = smoothed.iloc[-1].dropna().sort_values(ascending=False)
    last_pct = pct.iloc[-1]

    src_label = {"ths": "同花顺行业", "eastmoney": "东财行业",
                 "sw": "申万二级行业", "etf": "中证ETF行业/主题",
                 "baostock": "证监会行业"}
    if universe:
        src = src_label.get(universe.get("board_source"), universe.get("board_source", "?"))
        raw = universe.get("raw_board_count", universe.get("n_boards", "?"))
        eligible = universe.get("eligible_board_count", universe.get("n_boards", "?"))
        min_n = universe.get("min_constituents")
        rule = (f"原始 {raw} 个，入选 {eligible} 个"
                + (f"（至少 {min_n} 只成分股）" if min_n else ""))
        snapshot = universe.get("constituent_snapshot_at")
        caveat = (f"成分为 {snapshot} 的当前快照，历史回测仍可能存在幸存者偏差。"
                  if snapshot else "历史成分变更未完整还原，回测仍可能存在幸存者偏差。")
        uni_txt = (f"\n> 板块口径：{src}（{rule}），数据截至 {date:%Y-%m-%d}。"
                   f"不同口径下榜单与回测结论不可直接对比；{caveat}")
    else:
        uni_txt = ""

    def name_of(code: str) -> str:
        return board_names.get(code, code)

    def fmt_table(codes: list[str]) -> str:
        lines = ["| 排名 | 板块 | 情绪分 | 时序分位 |", "|---|---|---|---|"]
        for i, c in enumerate(codes, 1):
            p = last_pct.get(c)
            ptxt = f"{p:.0%}" if pd.notna(p) else "—"
            lines.append(f"| {i} | {name_of(c)} | {last_score[c]:+.2f} | {ptxt} |")
        return "\n".join(lines)

    top_codes = last_score.head(top_n).index.tolist()
    bottom_codes = last_score.tail(top_n).index.tolist()[::-1]

    def fmt_list(key: str) -> str:
        items = signals.get(key) or []
        if not items:
            return "（无）"
        out = []
        for c in items[:20]:
            # 切换预警字符串内嵌板块代码，逐个替换为名称
            txt = c
            for code, nm in board_names.items():
                if code in txt:
                    txt = txt.replace(code, nm)
            out.append(txt)
        return "、".join(out)

    if health and health.get("n"):
        ic_m, ic_w = health["ic_mean"], health["ic_win_rate"]
        state = "🟢 信号有效（顺势）" if ic_m > 0.03 else (
            "🔴 信号失效/反转" if ic_m < -0.03 else "🟡 信号弱")
        health_txt = (f"近 {health['n']} 日 IC={ic_m:+.3f}，胜率={ic_w:.0%} → {state}。"
                      "IC 转负说明市场风格已切换，榜单方向需反向理解。")
        # 分层价差区分"信号反"与"信号死"：IC 均值无法分辨二者
        sp = health.get("spread_mean")
        if sp is not None and pd.notna(sp):
            if sp < 0:
                health_txt += (f" 分层价差 Q5−Q1={sp:+.2%}/5日：分组单调倒置，"
                               "属“信号反”而非“信号死”——低分组相对占优，"
                               "但反转未经 regime 确认前仍只观察、不提前抄底。")
            else:
                health_txt += (f" 分层价差 Q5−Q1={sp:+.2%}/5日：分组结构未倒置。"
                               "IC 弱而价差≈0 时是“信号死”（无结构），两榜仅观察。")
        # 规则3 regime 自适应语义
        if ic_m > 0.03:
            div_label = "风险预警（顺势期建议回避）"
        elif ic_m < -0.03:
            div_label = "反转机会（逆势期可观察抄底）"
        else:
            div_label = "背离观察（信号弱，暂不操作）"
    else:
        ic_m = None
        health_txt = "数据不足，暂无法评估。"
        div_label = "资金-情绪背离"

    rule_cfg = breakthrough or {}
    proxy_z = float(rule_cfg.get("flow_proxy_z_threshold",
                                 rule_cfg.get("fund_flow_z_threshold", 2.0)))
    sigma = float(rule_cfg.get("extreme_sigma", 2.0))
    hot_sigma = float(rule_cfg.get("extreme_hot_sigma", sigma))
    cold_sigma = float(rule_cfg.get("extreme_cold_sigma", sigma))
    proxy_days = int(rule_cfg.get("flow_positive_days",
                                  rule_cfg.get("fund_positive_days", 2)))

    # 风格判定（regime 自适应）：方向建议 + 榜单语义自动改写
    dir_meta = {
        "top": ("🟢 动量市 → 榜单顺势使用",
                "情绪分 Top（顺势关注榜）",
                "情绪分 Bottom（观察榜）"),
        "bottom": ("🔴 反转市 → 榜单反向使用",
                   "情绪分 Top（回避榜：高情绪板块倾向跑输）",
                   "情绪分 Bottom（关注榜：低情绪板块倾向补涨）"),
        "neutral": ("🟡 观望 → 不做方向性轮动",
                    "情绪分 Top（仅观察，不建议追高）",
                    "情绪分 Bottom（仅观察，不建议抄底）"),
    }
    if regime:
        d = regime.get("direction", "neutral")
        regime_head, top_label, bottom_label = dir_meta.get(
            d, dir_meta["neutral"])
        ic_txt = (f"{regime['ic_mean']:+.3f}（胜率 "
                  f"{regime['ic_win_rate']:.0%}，{regime['n_obs']} 个样本）"
                  if pd.notna(regime.get("ic_mean")) else "样本不足")
        gates = []
        if regime.get("guard_hit"):
            mkt = regime.get("mkt_recent")
            mkt_txt = f"{mkt:+.1%}" if pd.notna(mkt) else "—"
            gates.append(f"回撤护栏触发（市场近 "
                         f"{regime.get('guard_days', 10)} 日累计 {mkt_txt}）→ 强制观望")
        if regime.get("vol_gate"):
            gates.append("波动闸门触发 → 强制观望")
        if regime.get("robust_gate"):
            low, high = regime.get("robust_ic_low"), regime.get("robust_ic_high")
            ci = (f"[{low:+.3f}, {high:+.3f}]" if pd.notna(low) and pd.notna(high)
                  else "样本不足")
            gates.append(f"IC 稳健性闸门（Bootstrap 95% CI {ci} 覆盖 0）→ 强制观望")
        if regime.get("mono_gate"):
            n = regime.get("mono_days", 3)
            sp = regime.get("spread_last")
            sp_txt = f"{sp:+.2%}" if pd.notna(sp) else "—"
            gates.append(f"单调性破坏预警（连续 {n} 个可知日分层价差反向，"
                         f"最新 Q5−Q1={sp_txt}/5日）→ 提前观望")
        gate_txt = ("；".join(gates) if gates else "未触发")
        regime_txt = f"""## 风格判定（regime 自适应，自动方向）

**{regime_head}**

- 已实现 IC（近 {regime.get('ic_window', 15)} 日，无前视）：{ic_txt}
- 风控闸门：{gate_txt}
- 使用纪律：观望=等权持有/降低仓位，不做方向性轮动；方向判定与参数见
  config.yaml `regime` 段，回测依据 report/regime_adaptive_backtest_*.md

> 榜单语义已按当前风格自动标注：动量市 Top 榜顺势参考；反转市 Top 榜按回避
> 理解、Bottom 榜按补涨关注；观望期两榜仅作观察。风格判定基于已实现 IC，
> 存在约 1 周的确认滞后（方向期间若分层价差连续反向，单调性破坏预警会提前
> 降级观望）；参数来自单一风格周期的回测，需持续校验。"""
    else:
        top_label = f"情绪分 Top {top_n}"
        bottom_label = f"情绪分 Bottom {top_n}"
        regime_txt = ""

    # 因子层监控：把合成分的 IC 拆到各因子，区分"方向可信"与"风格切换"
    verdict_txt = {
        "negative_robust": "⚠️ 方向为负且可信（高值倾向跑输）",
        "positive_robust": "✅ 方向为正且可信",
        "regime_flip": "🔄 与全期反向（风格切换，勿据此取反）",
        "no_direction": "· 无显著方向（CI 跨零）",
    }
    factor_txt = ""
    if factor_ic is not None and not factor_ic.empty:
        fw = int(factor_ic["window"].iloc[0])
        rows = [f"## 因子层监控（风格拆分，主看近 {fw} 日）", "",
                "综合分的 IC 是 6 个因子加权后的结果，单个因子失灵会被其余因子"
                "掩盖。本表把风格拆到因子级：**方向可信**（移动区块 Bootstrap 95% CI "
                "排除 0）才可据此调整用法；**与全期反向**只说明风格切换——据此取反"
                "等于拟合最近行情，不做。", "",
                f"| 因子 | 全期 IC | 120日 IC | 60日 IC | {fw}日 IC | {fw}日胜率 "
                f"| {fw}日 HAC t | {fw}日 Bootstrap 95% CI | 判定 |",
                "|---|---|---|---|---|---|---|---|---|"]
        for _, r in factor_ic.iterrows():
            lo, hi = r.get("ci_low"), r.get("ci_high")
            ci = f"[{lo:+.3f}, {hi:+.3f}]" if pd.notna(lo) and pd.notna(hi) else "—"
            iw = r.get(f"ic_{fw}", float("nan"))
            ww = r.get(f"win_{fw}", float("nan"))
            rows.append(
                f"| {r['label']} | {r['ic_all']:+.4f} | {r['ic_120']:+.4f} "
                f"| {r['ic_60']:+.4f} | {iw:+.4f} "
                f"| {(f'{ww:.0%}' if pd.notna(ww) else '—')} "
                f"| {r['hac_short']:+.2f} | {ci} "
                f"| {verdict_txt.get(r['verdict'], r['verdict'])} |")
        n_neg = int((factor_ic["verdict"] == "negative_robust").sum())
        n_pos = int((factor_ic["verdict"] == "positive_robust").sum())
        n_flip = int((factor_ic["verdict"] == "regime_flip").sum())
        rows += ["", f"> 本次：{n_neg} 个因子方向可信且为负、{n_pos} 个可信且为正、"
                     f"{n_flip} 个与全期反向（风格切换）。本表是**观测指标**，不参与"
                     f"任何自动调权——因子层改动的门槛与验证见 "
                     f"report/factor_adaptive_*.md。"]
        factor_txt = "\n".join(rows)

    md = f"""# A股板块情绪日报 {date:%Y-%m-%d}

> 情绪分为多因子 z-score 加权合成（涨停强度/赚钱效应/涨跌宽度/相对强度/热度变化/晋级率），
> 经 5 日平滑。时序分位为该板块相对自身近期的位置。
> 本报告为同步/短滞后指标的客观读数，不构成投资建议。{uni_txt}

## 信号健康度（风格监控）

{health_txt}

{regime_txt}

{factor_txt}

## 信号

- **升温榜**（连续 3 日上升）：{fmt_list('heating')}
- **降温榜**（连续 3 日下降）：{fmt_list('cooling')}
- **亢奋**（分位 > 90%，不宜追高）：{fmt_list('hot')}
- **冰点**（分位 < 10%，观察反转）：{fmt_list('cold')}
- **切换预警**：{fmt_list('switching')}
- **底部反转预警**（冰点 + 成交额价格代理 z > {proxy_z:g}）：{fmt_list('bottom_reversal')}
- **顺势加速度**（亢奋 + 单因子偏离 {hot_sigma:g}σ，动量市买点）：{fmt_list('extreme_hot')}
- **极端冷区异动**（冰点 + 单因子偏离 {cold_sigma:g}σ，动量市回避）：{fmt_list('extreme_cold')}
- **代理-情绪背离**（成交额价格代理连续 {proxy_days} 日为正 + 情绪为负，{div_label}）：{fmt_list('fund_emotion_divergence')}

> 穿透规则为原始日度读数、不经平滑；上方经平滑的榜单反映趋势，穿透信号反映突变——
> 二者互补。顺势环境中 extreme_hot 为主要买点、extreme_cold 为回避信号、
> “成交额价格代理”=成交额×涨跌幅，不是主动买卖盘意义的资金净流入；代理-情绪背离
> 的语义随 regime 自适应切换。接入真实资金流前，不应将它作为资金流证据。

## {top_label}

{fmt_table(top_codes)}

## {bottom_label}

{fmt_table(bottom_codes)}

---
生成时间：{datetime.now():%Y-%m-%d %H:%M}
"""
    return md


def save_report(report_dir: str | Path, date: pd.Timestamp, md: str,
                smoothed: pd.DataFrame, board_names: dict[str, str]) -> tuple[Path, Path]:
    """落盘 Markdown 报告与当日情绪分 CSV。返回 (md_path, csv_path)。"""
    out = Path(report_dir)
    out.mkdir(parents=True, exist_ok=True)
    day = date.strftime("%Y%m%d")

    md_path = out / f"report_{day}.md"
    md_path.write_text(md, encoding="utf-8")

    last = smoothed.iloc[-1].rename("score").reset_index()
    last.columns = ["board_code", "score"]
    last["board_name"] = last["board_code"].map(board_names)
    last = last[["board_code", "board_name", "score"]].sort_values(
        "score", ascending=False)
    csv_path = out / f"scores_{day}.csv"
    last.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return md_path, csv_path
