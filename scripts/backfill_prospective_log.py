"""回填前瞻预注册台账（一次性工具）。

从 report/report_*.md 的「风格判定」与「信号健康度」段解析历史判定，
以 backfill=True（行首 * 标记）写入 report/prospective_log.md。
10-09 起为事前留痕（main.py 自动追加），本脚本对已有日期幂等跳过。

用法：python scripts/backfill_prospective_log.py [--dry-run]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from emotion import report  # noqa: E402

REPORT_DIR = ROOT / "report"

# 判定行：**🔴 反转市 → 榜单反向使用** 等
RE_DIR = re.compile(r"\*\*(?:🔴|🟡|🟢)\s*(动量市|反转市|观望)\s*→")
# 已实现 IC（近 15 日，无前视）：-0.151（胜率 33%，15 个样本）
RE_IC15 = re.compile(r"已实现 IC（近 15 日，无前视）：\s*([+-]?[\d.]+)（胜率\s*(\d+)%")
# 风控闸门：xxx / 未触发
RE_GATE = re.compile(r"风控闸门：(.+)")
# 近 60 日 IC=-0.158，胜率=37%
RE_IC60 = re.compile(r"近 60 日 IC=([+-]?[\d.]+)，胜率=(\d+)%")
# 分层价差 Q5−Q1=-1.38%/5日（− 兼容 U+2212 与 ASCII）
RE_SPREAD = re.compile(r"Q5[−-]Q1=([+-]?[\d.]+)%")
# 单调性破坏行内的 最新 Q5−Q1=+0.91%/5日
RE_SPREAD_LAST = re.compile(r"最新 Q5[−-]Q1=([+-]?[\d.]+)%")
# 回撤护栏触发（市场近 10 日累计 -4.1%）
RE_MKT = re.compile(r"市场近 10 日累计\s*([+-]?[\d.]+)%")

GATE_MAP = [
    ("IC 稳健性闸门", "IC稳健性"),
    ("回撤护栏", "回撤护栏"),
    ("波动闸门", "波动闸门"),
    ("单调性破坏", "单调性破坏"),
]


def parse_one(path: Path) -> dict | None:
    txt = path.read_text(encoding="utf-8")
    m_dir = RE_DIR.search(txt)
    if not m_dir:
        return None
    nan = float("nan")
    direction = {"动量市": "top", "反转市": "bottom", "观望": "neutral"}[m_dir.group(1)]

    m = RE_IC15.search(txt)
    ic15, ic15_win = (float(m.group(1)), int(m.group(2)) / 100) if m else (nan, nan)

    gates, spread_last, mkt = [], nan, nan
    m = RE_GATE.search(txt)
    if m and "未触发" not in m.group(1):
        gate_line = m.group(1)
        for key, label in GATE_MAP:
            if key in gate_line:
                gates.append(label)
        ms = RE_SPREAD_LAST.search(gate_line)
        if ms:
            spread_last = float(ms.group(1)) / 100
        mm = RE_MKT.search(gate_line)
        if mm:
            mkt = float(mm.group(1)) / 100

    m = RE_IC60.search(txt)
    ic60, ic60_win = (float(m.group(1)), int(m.group(2)) / 100) if m else (nan, nan)
    spread60 = nan
    for ms in RE_SPREAD.finditer(txt):
        spread60 = float(ms.group(1)) / 100

    day = path.stem.replace("report_", "")
    date = pd.Timestamp(f"{day[:4]}-{day[4:6]}-{day[6:]}")
    regime = {"direction": direction, "ic_mean": ic15, "ic_win_rate": ic15_win,
              "guard_hit": "回撤护栏" in gates, "vol_gate": "波动闸门" in gates,
              "robust_gate": "IC稳健性" in gates, "mono_gate": "单调性破坏" in gates,
              "spread_last": spread_last, "mkt_recent": mkt}
    health = {"ic_mean": ic60, "ic_win_rate": ic60_win, "spread_mean": spread60}
    return {"date": date, "regime": regime, "health": health}


def main() -> None:
    dry = "--dry-run" in sys.argv
    files = sorted(REPORT_DIR.glob("report_*.md"))
    n_ok, n_skip = 0, 0
    for f in files:
        parsed = parse_one(f)
        if parsed is None:
            continue
        if dry:
            r, h = parsed["regime"], parsed["health"]
            print(f"[dry] {parsed['date'].date()} {r['direction']:8s} "
                  f"ic15={r['ic_mean']:+.3f} ic60={h['ic_mean']:+.3f}")
            n_ok += 1
            continue
        p = report.append_prospective_log(
            REPORT_DIR, parsed["date"], parsed["regime"], parsed["health"],
            backfill=True)
        if p is None:
            n_skip += 1
        else:
            n_ok += 1
    print(f"回填完成：写入 {n_ok} 行，幂等跳过 {n_skip} 行（含事前留痕日）")


if __name__ == "__main__":
    main()
