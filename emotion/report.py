"""每日报告生成：Markdown + CSV。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd


def build_report(date: pd.Timestamp, smoothed: pd.DataFrame, pct: pd.DataFrame,
                 signals: dict[str, list[str]], board_names: dict[str, str],
                 *, top_n: int = 15, health: dict | None = None,
                 universe: dict | None = None) -> str:
    """生成当日 Markdown 报告文本。board_names: board_code -> 名称。

    health: recent_signal_health 的输出（最近 N 日 IC 监控），可选。
    universe: {"board_source": ..., "n_boards": ...}，用于在报告头标注板块口径。
    """
    last_score = smoothed.iloc[-1].dropna().sort_values(ascending=False)
    last_pct = pct.iloc[-1]

    src_label = {"ths": "同花顺行业", "eastmoney": "东财行业",
                 "baostock": "证监会行业"}
    if universe:
        src = src_label.get(universe.get("board_source"), universe.get("board_source", "?"))
        uni_txt = (f"\n> 板块口径：{src}（{universe.get('n_boards', '?')} 个板块），"
                   f"数据截至 {date:%Y-%m-%d}。不同口径下榜单与回测结论不可直接对比。")
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
    else:
        health_txt = "数据不足，暂无法评估。"

    md = f"""# A股板块情绪日报 {date:%Y-%m-%d}

> 情绪分为多因子 z-score 加权合成（涨停强度/赚钱效应/涨跌宽度/相对强度/热度变化/晋级率），
> 经 5 日平滑。时序分位为该板块相对自身近期的位置。
> 本报告为同步/短滞后指标的客观读数，不构成投资建议。{uni_txt}

## 信号健康度（风格监控）

{health_txt}

## 信号

- **升温榜**（连续 3 日上升）：{fmt_list('heating')}
- **降温榜**（连续 3 日下降）：{fmt_list('cooling')}
- **亢奋**（分位 > 90%，不宜追高）：{fmt_list('hot')}
- **冰点**（分位 < 10%，观察反转）：{fmt_list('cold')}
- **切换预警**：{fmt_list('switching')}

## 情绪分 Top {top_n}

{fmt_table(top_codes)}

## 情绪分 Bottom {top_n}

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
