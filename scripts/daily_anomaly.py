"""一次性补丁：基于 stock_chunks 计算今日板块异动榜单。

不算情绪分（避免与原报告混淆），只展示：
- 今日板块成交额价格代理 Top/Bottom（amount * pctChg；非真实资金流）
- 今日板块涨幅 Top/Bottom（按加权平均 pctChg）
- 板块"5日连跌→今日反弹"反转榜
- 板块"5日连涨→今日回落"预警榜

输出：report/daily_anomaly_YYYYMMDD.md
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REPORT_DIR = ROOT / "report"
CHUNK_DIR = ROOT / "data" / "cache" / "stock_chunks"

# 东财 BK 板块映射（用于展示和聚合）
ALL_CONS_PATH = ROOT / "data" / "cache" / "all_cons.parquet"

# 尝试加载 board_names（用于展示板块中文名）
def load_board_mapping() -> tuple[pd.DataFrame, dict[str, str]]:
    """从 all_cons 提取 (code -> board_code) 映射和 (board_code -> board_name)。"""
    ac = pd.read_parquet(ALL_CONS_PATH)
    bm = ac[["code", "board_code"]].drop_duplicates()
    board_names = dict(zip(ac["board_code"], ac["board_name"]))
    return bm, board_names


def load_stock_daily() -> pd.DataFrame:
    parts = []
    for p in sorted(CHUNK_DIR.glob("chunk_*.parquet")):
        parts.append(pd.read_parquet(p))
    df = pd.concat(parts, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df["pctChg"] = pd.to_numeric(df["pctChg"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.dropna(subset=["pctChg", "amount", "date"])
    return df


def attach_board(df: pd.DataFrame, bm: pd.DataFrame) -> pd.DataFrame:
    m = bm[["code", "board_code"]].drop_duplicates()
    return df.merge(m, on="code", how="inner")


def compute_today_signals(df: pd.DataFrame) -> tuple[pd.Timestamp, pd.DataFrame]:
    today = df["date"].max()
    today_df = df[df["date"] == today].copy()

    # 成交额价格代理 = sum(amount * pctChg / 100)。它表达成交额加权涨跌方向，
    # 不含主动买卖盘，不能称作真实资金净流入。
    today_df["net_flow_wan"] = today_df["amount"] * today_df["pctChg"] / 100
    today_df["weighted_pct"] = today_df["pctChg"] * today_df["amount"]

    agg = today_df.groupby("board_code").agg(
        net_flow=("net_flow_wan", "sum"),
        weighted_pct=("weighted_pct", "sum"),
        total_amount=("amount", "sum"),
        n_stocks=("code", "nunique"),
    )
    agg["avg_pct"] = agg["weighted_pct"] / agg["total_amount"]

    # 反转检测：板块内"5日连跌→今日反弹"
    # 关键：用**金额加权** pctChg（而非算术平均），与成交额价格代理口径一致。
    recent15 = df[df["date"] > today - pd.Timedelta(days=15)]
    g = recent15.groupby(["board_code", "date"]).apply(
        lambda d: pd.Series({
            "pctChg": (d["pctChg"] * d["amount"]).sum() / d["amount"].sum()
            if d["amount"].sum() > 0 else 0.0
        }),
        include_groups=False,
    ).reset_index()
    bd = g.sort_values(["board_code", "date"]).reset_index(drop=True)
    bd["up"] = bd["pctChg"] > 0
    bd["down"] = bd["pctChg"] < 0
    # 连续下跌天数：用 groupby + cumsum 技巧
    bd["down_block"] = (bd["pctChg"] >= 0).astype(int)  # 1 表示非下跌（断开）
    bd["down_block"] = bd.groupby("board_code")["down_block"].cumsum()
    bd["down_streak"] = bd.groupby(["board_code", "down_block"]).cumcount()
    bd.loc[bd["pctChg"] >= 0, "down_streak"] = 0  # 上涨日 streak=0

    today_pct = bd[bd["date"] == today].set_index("board_code")["pctChg"]
    agg["today_pct_mean"] = today_pct

    # 今日之前（最后一个 < today 的交易日）的连续下跌天数
    last_before_today = bd[bd["date"] < today].sort_values("date").groupby("board_code").tail(1)
    prev_streak_map = last_before_today.set_index("board_code")["down_streak"]
    agg["prev_down_streak"] = prev_streak_map

    # 5 日平均（金额加权口径，< today）
    last5 = bd[bd["date"] < today].sort_values("date").groupby("board_code").tail(5)
    avg5 = last5.groupby("board_code")["pctChg"].mean()
    agg["avg5_pct"] = avg5

    # 反转强度 = today_pct - avg5_pct
    agg["reversal"] = agg["today_pct_mean"] - agg["avg5_pct"]

    return today, agg


def fmt_yi(v: float) -> str:
    """元 → 亿元显示；amount * pctChg / 100 的代理值单位也为元。"""
    if pd.isna(v):
        return "—"
    yi = v / 1e8
    sign = "+" if yi > 0 else ""
    if abs(yi) >= 100:
        return f"{sign}{yi:.0f}亿"
    if abs(yi) >= 10:
        return f"{sign}{yi:.1f}亿"
    return f"{sign}{yi:.2f}亿"


def build_section(title: str, df: pd.DataFrame, col: str, fmt, n: int = 10) -> list[str]:
    out = [f"### {title}", ""]
    out.append("| 排名 | 板块 | 成交额价格代理 | 加权涨幅 | 成交额 | 成分股数 |")
    out.append("|---|---|---|---|---|---|")
    rows = df.sort_values(col, ascending=False).head(n)
    for i, (code, r) in enumerate(rows.iterrows(), 1):
        nm = board_names.get(code, code)
        out.append(
            f"| {i} | {nm} | {fmt_yi(r['net_flow'])} | {r['avg_pct']:+.2f}% | "
            f"{r['total_amount']/1e8:.0f}亿 | {int(r['n_stocks'])} |"
        )
    out.append("")
    return out


def main():
    global board_names
    print("加载个股数据...")
    df = load_stock_daily()
    print(f"  {len(df)} 行，最近日期 {df['date'].max()}")

    bm, board_names = load_board_mapping()
    print(f"  板块映射：{bm['board_code'].nunique()} 个东财板块")

    df = attach_board(df, bm)
    print(f"  合并后 {len(df)} 行")

    today, agg = compute_today_signals(df)
    agg = agg.dropna(subset=["net_flow"])
    print(f"  今日 = {today.date()}，{len(agg)} 个板块有数据")

    date_str = today.strftime("%Y%m%d")
    out_path = REPORT_DIR / f"daily_anomaly_{date_str}.md"

    md = []
    md.append(f"# A股板块今日异动榜 {today.strftime('%Y-%m-%d')}")
    md.append("")
    md.append("> 基于个股日线直接计算，未走情绪分（5日平滑）系统，作为当日实时异动补丁。")
    md.append("> “成交额价格代理”=成交额×涨跌幅，不是主动买卖盘意义的资金净流入。")
    md.append("> 与日报（report_YYYYMMDD.md）配合使用：日报看趋势，今日异动看时点。")
    md.append("")
    md.append(f"覆盖板块：{agg.shape[0]} 个；个股：{int(agg['n_stocks'].sum())} 只。")
    md.append("")

    md.append("## 一、板块成交额价格代理 Top 10")
    md.append("")
    md.append("| 排名 | 板块 | 价格代理 | 加权涨幅 | 5日均涨 | 反转强度 |")
    md.append("|---|---|---|---|---|---|")
    rows = agg.sort_values("net_flow", ascending=False).head(10)
    for i, (code, r) in enumerate(rows.iterrows(), 1):
        nm = board_names.get(code, code)
        rev = r["reversal"]
        rev_s = f"{rev:+.2f}%" if pd.notna(rev) else "—"
        avg5 = r["avg5_pct"]
        avg5_s = f"{avg5:+.2f}%" if pd.notna(avg5) else "—"
        md.append(
            f"| {i} | {nm} | **{fmt_yi(r['net_flow'])}** | {r['avg_pct']:+.2f}% | "
            f"{avg5_s} | {rev_s} |"
        )
    md.append("")

    md.append("## 二、今日板块涨幅 Top 10（按加权涨幅）")
    md.append("")
    md.append("| 排名 | 板块 | 加权涨幅 | 价格代理 | 5日均涨 |")
    md.append("|---|---|---|---|---|")
    rows = agg.sort_values("avg_pct", ascending=False).head(10)
    for i, (code, r) in enumerate(rows.iterrows(), 1):
        nm = board_names.get(code, code)
        avg5 = r["avg5_pct"]
        avg5_s = f"{avg5:+.2f}%" if pd.notna(avg5) else "—"
        md.append(
            f"| {i} | {nm} | **{r['avg_pct']:+.2f}%** | {fmt_yi(r['net_flow'])} | "
            f"{avg5_s} |"
        )
    md.append("")

    md.append("## 三、板块成交额价格代理 Bottom 10")
    md.append("")
    md.append("| 排名 | 板块 | 价格代理 | 加权涨幅 | 5日均涨 | 反转强度 |")
    md.append("|---|---|---|---|---|---|")
    rows = agg.sort_values("net_flow", ascending=True).head(10)
    for i, (code, r) in enumerate(rows.iterrows(), 1):
        nm = board_names.get(code, code)
        rev = r["reversal"]
        rev_s = f"{rev:+.2f}%" if pd.notna(rev) else "—"
        avg5 = r["avg5_pct"]
        avg5_s = f"{avg5:+.2f}%" if pd.notna(avg5) else "—"
        md.append(
            f"| {i} | {nm} | **{fmt_yi(r['net_flow'])}** | {r['avg_pct']:+.2f}% | "
            f"{avg5_s} | {rev_s} |"
        )
    md.append("")

    # 反转榜：5日连跌今日反弹
    rev_strong = agg.dropna(subset=["reversal"]).sort_values("reversal", ascending=False).head(10)
    md.append("## 四、5日连跌后今日反弹 Top 10（反转预警）")
    md.append("")
    md.append("反转强度 = 今日涨幅 − 近5日均涨；正值越大代表跌势逆转越强。")
    md.append("")
    md.append("| 排名 | 板块 | 今日涨 | 5日均涨 | 反转强度 | 价格代理 | 下跌天数 |")
    md.append("|---|---|---|---|---|---|---|")
    for i, (code, r) in enumerate(rev_strong.iterrows(), 1):
        nm = board_names.get(code, code)
        streak = int(r["prev_down_streak"]) if pd.notna(r["prev_down_streak"]) else 0
        md.append(
            f"| {i} | {nm} | {r['today_pct_mean']:+.2f}% | {r['avg5_pct']:+.2f}% | "
            f"**{r['reversal']:+.2f}%** | {fmt_yi(r['net_flow'])} | {streak} |"
        )
    md.append("")

    # 5日连涨今日回落
    rev_neg = agg.dropna(subset=["reversal"]).sort_values("reversal", ascending=True).head(5)
    md.append("## 五、5日连涨后今日回落（预警）")
    md.append("")
    md.append("| 排名 | 板块 | 今日涨 | 5日均涨 | 反转强度 | 价格代理 |")
    md.append("|---|---|---|---|---|---|")
    for i, (code, r) in enumerate(rev_neg.iterrows(), 1):
        nm = board_names.get(code, code)
        md.append(
            f"| {i} | {nm} | {r['today_pct_mean']:+.2f}% | {r['avg5_pct']:+.2f}% | "
            f"**{r['reversal']:+.2f}%** | {fmt_yi(r['net_flow'])} |"
        )
    md.append("")

    # 涨跌停分布（基于个股原始字段近似）
    md.append("## 六、今日板块涨跌停分布（近似）")
    md.append("")
    today_df = df[df["date"] == today].copy()
    # 涨停阈值 9.8% 主板，19.8% 创业板/科创板 30%北交所；
    # 简化：以 pctChg >= 9.7（主板/中小板）和 19.7（创/科）判定
    def _is_lu(row):
        if pd.isna(row["pctChg"]):
            return False
        if str(row.get("code", "")).startswith(("300", "301", "688", "689")):
            return row["pctChg"] >= 19.5
        return row["pctChg"] >= 9.5
    today_df["is_lu"] = today_df.apply(_is_lu, axis=1)
    today_df["is_dn"] = today_df["pctChg"] <= -9.5
    lu = today_df[today_df["is_lu"]].groupby("board_code")["code"].count().sort_values(ascending=False).head(10)
    md.append("**涨停家数 Top 10**")
    md.append("")
    md.append("| 排名 | 板块 | 涨停数 | 净流入 |")
    md.append("|---|---|---|---|")
    for i, (code, n) in enumerate(lu.items(), 1):
        nm = board_names.get(code, code)
        nf = agg.loc[code, "net_flow"] if code in agg.index else float("nan")
        md.append(f"| {i} | {nm} | **{int(n)}** | {fmt_yi(nf)} |")
    md.append("")

    dn = today_df[today_df["is_dn"]].groupby("board_code")["code"].count().sort_values(ascending=False).head(5)
    if len(dn) > 0:
        md.append("**跌停家数 Top 5**")
        md.append("")
        md.append("| 排名 | 板块 | 跌停数 |")
        md.append("|---|---|---|")
        for i, (code, n) in enumerate(dn.items(), 1):
            nm = board_names.get(code, code)
            md.append(f"| {i} | {nm} | **{int(n)}** |")
        md.append("")

    md.append("---")
    md.append(f"生成时间：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}  | "
              f"数据截至：{today.strftime('%Y-%m-%d')}  | "
              f"补丁脚本：scripts/daily_anomaly.py")
    md.append("")

    out_path.write_text("\n".join(md), encoding="utf-8")
    print(f"\n输出：{out_path}")
    print(f"字节数：{out_path.stat().st_size}")


if __name__ == "__main__":
    main()
