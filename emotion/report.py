"""每日报告生成：Markdown + CSV。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from emotion.project_status import OBSERVATION_NOTICE


def build_report(date: pd.Timestamp, smoothed: pd.DataFrame, pct: pd.DataFrame,
                 signals: dict[str, list[str]], board_names: dict[str, str],
                 *, top_n: int = 15, health: dict | None = None,
                 regime: dict | None = None,
                 breakthrough: dict | None = None,
                 universe: dict | None = None,
                 factor_ic: pd.DataFrame | None = None,
                 policy: dict | None = None,
                 board_names_for_policy: dict[str, str] | None = None) -> str:
    """生成当日 Markdown 报告文本。board_names: board_code -> 名称。

    health: recent_signal_health 的输出（最近 N 日 IC 监控），可选。
    regime: 原模型读数，可选；只留作研究记录，不改变观察榜的用途。
    breakthrough: signal.breakthrough 配置，用于让报告阈值与实际信号判定一致。
    universe: {"board_source": ..., "n_boards": ...}，用于在报告头标注板块口径。
    factor_ic: backtest.factor_ic_snapshot 的输出（逐因子 IC 与可信度），可选。
    policy: 政策（时政）支柱载荷（main._policy_payload 输出），可选；
    缺省时报告不含政策段（政策层不阻断主线）。
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
        state = "历史排序相关为正" if ic_m > 0.03 else (
            "历史排序相关为负" if ic_m < -0.03 else "历史排序相关较弱")
        health_txt = (f"近 {health['n']} 日 IC={ic_m:+.3f}，胜率={ic_w:.0%} → {state}。"
                      "这是已实现收益的统计关系，不能据此确认风格切换或反向交易。")
        sp = health.get("spread_mean")
        if sp is not None and pd.notna(sp):
            health_txt += f" 分层价差 Q5−Q1={sp:+.2%}/5日；端点差不等于全部分组单调。"
        div_label = "背离观察"
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

    # 保留原模型读数供观察，不把统计方向翻译为交易动作。
    dir_meta = {
        "top": "原模型读数：正向",
        "bottom": "原模型读数：反向",
        "neutral": "原模型读数：中性",
    }
    top_label = f"情绪分 Top {top_n}（高分观察榜）"
    bottom_label = f"情绪分 Bottom {top_n}（低分观察榜）"
    if regime:
        d = regime.get("direction", "neutral")
        regime_head = dir_meta.get(d, dir_meta["neutral"])
        ic_txt = (f"{regime['ic_mean']:+.3f}（胜率 "
                  f"{regime['ic_win_rate']:.0%}，{regime['n_obs']} 个样本）"
                  if pd.notna(regime.get("ic_mean")) else "样本不足")
        gates = []
        if regime.get("guard_hit"):
            mkt = regime.get("mkt_recent")
            mkt_txt = f"{mkt:+.1%}" if pd.notna(mkt) else "—"
            gates.append(f"回撤护栏触发（市场近 "
                         f"{regime.get('guard_days', 10)} 日累计 {mkt_txt}）")
        if regime.get("vol_gate"):
            gates.append("波动闸门触发")
        if regime.get("robust_gate"):
            low, high = regime.get("robust_ic_low"), regime.get("robust_ic_high")
            ci = (f"[{low:+.3f}, {high:+.3f}]" if pd.notna(low) and pd.notna(high)
                  else "样本不足")
            gates.append(f"IC 稳健性闸门（Bootstrap 95% CI {ci} 未确认统计方向）")
        if regime.get("mono_gate"):
            n = regime.get("mono_days", 3)
            sp = regime.get("spread_last")
            sp_txt = f"{sp:+.2%}" if pd.notna(sp) else "—"
            gates.append(f"单调性破坏预警（连续 {n} 个可知日分层价差反向，"
                         f"最新 Q5−Q1={sp_txt}/5日）")
        gate_txt = ("；".join(gates) if gates else "未触发")
        regime_txt = f"""## 原模型状态（仅作观察）

**{regime_head}**

- 已实现 IC（近 {regime.get('ic_window', 15)} 日，无前视）：{ic_txt}
- 风控闸门：{gate_txt}
- 原有计算参数继续冻结在 config.yaml `regime` 段。

> 方向是旧模型的研究读数，不代表已验证的交易优势。高低分榜均只观察；
> 历史回测中的 neutral=等权是模拟规则，不是当前仓位建议。"""
        if "rebalance_direction" in regime:
            execution = {"top": "正向", "bottom": "反向", "neutral": "中性"}
            regime_txt += (f"\n\n- 最近研究信号日：{regime.get('rebalance_date') or '尚无'}；"
                           f"旧规则确认后读数：{execution[regime['rebalance_direction']]}。"
                           f"连续确认次数参数：{regime.get('confirm_readings', 1)}；不生成交易指令。")
    else:
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
                "掩盖。本表展示因子与已实现收益的关系。移动区块 Bootstrap 95% CI "
                "排除 0 也不证明可交易超额；与全期反向仅为分段差异，不据此取反或调权。", "",
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

    # ---- 政策（时政）支柱：与情绪支柱完全平行的观察层 ----
    policy_txt = ""
    if policy is not None:
        from emotion.policy import DUAL_STATE_CN
        p_smoothed = policy["smoothed"]
        p_pct = policy["pct"]
        p_names = board_names_for_policy or board_names
        last_pscore = p_smoothed.iloc[-1].dropna().sort_values(ascending=False)
        last_ppct = p_pct.iloc[-1] if not p_pct.empty else pd.Series(dtype=float)
        last_intensity = policy.get("intensity")
        # 政策热度 Top 10（观察榜，无任何买卖解读）
        top_p = last_pscore.head(10)
        rows = ["| 排名 | 板块 | 政策分 | 时序分位 | 当日命中 |", "|---|---|---|---|---|"]
        for i, c in enumerate(top_p.index, 1):
            pv = last_ppct.get(c)
            ptxt = f"{pv:.0%}" if pd.notna(pv) else "—"
            hits = (f"{int(last_intensity[c])}" if last_intensity is not None
                    and pd.notna(last_intensity.get(c)) else "—")
            rows.append(f"| {i} | {p_names.get(c, c)} | {last_pscore[c]:+.2f} "
                        f"| {ptxt} | {hits} |")
        sig = policy.get("signals") or {}
        def _fmt_policy_list(key):
            items = sig.get(key) or []
            if not items:
                return "（无）"
            return "、".join(p_names.get(c, c) for c in items[:10])
        n_res = len(sig.get("policy_resonance") or [])
        n_lurk = len(sig.get("policy_lurking") or [])
        policy_txt = f"""## 政策（时政）观察

> 数据口径：央视新闻联播文字稿（{policy.get('news_days', '?')} 天缓存，截至 {policy.get('news_end', '?')}）；
> 关键词词典 {policy.get('n_keyword_boards', '?')} 个板块，纯字符串匹配。
> 政策分 = 当日命中条数 / 近 5 日滚动热度 / 近 10 日政策广度三口径的横截面标准化合成，
> 与情绪分同 5 日平滑、同 120 日时序分位。**命中是"提及强度"的客观计数，
> 不代表政策利好或利空方向**；本段为观察层，不构成任何买卖建议。
""" + ("" if policy.get("fresh", True) else """
> ⚠️ **缓存缺口提示**：联播缓存末日与情绪数据末日相差超过 3 天，
> 本段政策读数滞后于盘面（当日命中列不展示）。请运行
> `python scripts/fetch_cctv_news.py` 补齐后再解读本段。
""") + f"""
### 政策热度榜（Top 10，仅观察）

{chr(10).join(rows)}

### 双因交叉信号（情绪分位 × 政策分位，阈值 80%/20%）

- **{DUAL_STATE_CN['resonance']}**（{n_res} 个）：{_fmt_policy_list('policy_resonance')}
- **{DUAL_STATE_CN['lurking']}**（{n_lurk} 个）：{_fmt_policy_list('policy_lurking')}

> 双因状态仅描述"情绪与政策各自处于自身历史分位的相对位置"，与已冻结的
> 旧模型方向判定完全解耦；情绪热+政策热不等于买入信号，情绪冷+政策热
> 不等于布局时机。政策支柱不参与任何回测调参与重启门槛判定。
"""

    md = f"""# A股板块情绪日报 {date:%Y-%m-%d}

> **观察模式**：{OBSERVATION_NOTICE}

> 情绪分为多因子 z-score 加权合成（涨停强度/赚钱效应/涨跌宽度/相对强度/热度变化/晋级率），
> 经 5 日平滑。时序分位为该板块相对自身近期的位置。
> 本报告为同步/短滞后指标的客观读数，不构成投资建议。{uni_txt}

## 信号健康度（风格监控）

{health_txt}

{regime_txt}

{factor_txt}

{policy_txt}

## 信号

- **升温榜**（连续 3 日上升）：{fmt_list('heating')}
- **降温榜**（连续 3 日下降）：{fmt_list('cooling')}
- **亢奋**（分位 > 90%）：{fmt_list('hot')}
- **冰点**（分位 < 10%）：{fmt_list('cold')}
- **切换预警**：{fmt_list('switching')}
- **冷区代理异动**（冰点 + 成交额价格代理 z > {proxy_z:g}）：{fmt_list('bottom_reversal')}
- **极端热区异动**（亢奋 + 单因子偏离 {hot_sigma:g}σ）：{fmt_list('extreme_hot')}
- **极端冷区异动**（冰点 + 单因子偏离 {cold_sigma:g}σ）：{fmt_list('extreme_cold')}
- **代理-情绪背离**（成交额价格代理连续 {proxy_days} 日为正 + 情绪为负，{div_label}）：{fmt_list('fund_emotion_divergence')}

> 穿透规则为原始日度读数、不经平滑；上方经平滑的榜单反映趋势，穿透信号反映突变——
> 二者用于描述市场状态，极值不代表入场或离场时机。
> “成交额价格代理”=成交额×涨跌幅，不是主动买卖盘意义的资金净流入；代理-情绪背离
> 仅记录两个指标的分歧。接入真实资金流前，不应将它作为资金流证据。

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


# ---------------------------------------------------------------------------
# 前瞻预注册台账（prospective log）
# ---------------------------------------------------------------------------
# 用途：每天生成日报时自动留痕一行"当日事前判定 + 关键证据"，只追加、
# 不回改。2026-10-13最终实验已结束交易路线；新增记录只保留观察用途。
# 回溯补录的行必须带 * 标记，与事前留痕严格区分。
PROSPECTIVE_HEADER = (
    "# 前瞻预注册台账（事前判定留痕）\n\n"
    "> main.py 每次生成日报自动追加；同日重跑不重复。回溯补录行以 * 开头。\n"
    "> 2026-10-13：最终实验未通过，当前交易路线已结束；此后台账仅为观察记录。\n\n"
    "| 日期 | 判定 | 触发闸门 | 15日IC | 15日胜率 | 60日IC | 60日胜率 "
    "| 60日价差 | 可知日价差 | 60日CI | 近10日市场 | 记录时间 |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|---|\n"
)

_DIRECTION_CN = {"top": "动量市", "bottom": "反转市", "neutral": "观望"}


def _fmt(x, fmt="{:+.3f}", na="-") -> str:
    try:
        if x is None or pd.isna(x):
            return na
        return fmt.format(float(x))
    except (TypeError, ValueError):
        return na


def append_prospective_log(report_dir: str | Path, date: pd.Timestamp,
                           regime: dict, health: dict,
                           *, backfill: bool = False) -> Path | None:
    """追加一行前瞻判定到 report/prospective_log.md。

    同一日期已存在（且非回溯覆盖场景）则跳过，幂等。backfill=True 用于
    历史回填脚本，行首日期后加 * 标记。
    """
    out = Path(report_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "prospective_log.md"
    day = date.strftime("%Y-%m-%d")

    gates = []
    if regime.get("guard_hit"):
        gates.append("回撤护栏")
    if regime.get("vol_gate"):
        gates.append("波动闸门")
    if regime.get("robust_gate"):
        gates.append("IC稳健性")
    if regime.get("mono_gate"):
        gates.append("单调性破坏")
    gate_str = "/".join(gates) if gates else "-"

    ci_low, ci_high = _fmt(health.get("IC_bootstrap_low")), _fmt(
        health.get("IC_bootstrap_high"))
    ci_str = "-" if ci_low == ci_high == "-" else f"[{ci_low}, {ci_high}]"

    mark = "*" if backfill else ""
    row = (
        f"| {day}{mark} | {_DIRECTION_CN.get(regime.get('direction'), '?')}"
        f" | {gate_str}"
        f" | {_fmt(regime.get('ic_mean'))}"
        f" | {_fmt(regime.get('ic_win_rate'), '{:.0%}')}"
        f" | {_fmt(health.get('ic_mean'))}"
        f" | {_fmt(health.get('ic_win_rate'), '{:.0%}')}"
        f" | {_fmt(health.get('spread_mean'), '{:+.2%}')}"
        f" | {_fmt(regime.get('spread_last'), '{:+.2%}')}"
        f" | {ci_str}"
        f" | {_fmt(regime.get('mkt_recent'), '{:+.1%}')}"
        f" | {datetime.now().strftime('%Y-%m-%d %H:%M')} |"
    )

    existing = ""
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if OBSERVATION_NOTICE not in existing:
            # 保留历史行及其原始预注册文字，另加有日期的状态更新。
            existing = (f"> 状态更新（2026-10-13）：{OBSERVATION_NOTICE}"
                        "原2027-01交易资格裁决安排已由本次最终实验决定取代。\n\n" + existing)
            path.write_text(existing, encoding="utf-8")
        for line in existing.splitlines():
            if line.startswith(f"| {day} ") or line.startswith(f"| {day}* "):
                return None  # 同日已留痕，幂等跳过

    if not existing:
        path.write_text(PROSPECTIVE_HEADER + row + "\n", encoding="utf-8")
    else:
        with open(path, "a", encoding="utf-8") as f:
            f.write(row + "\n")
    return path
