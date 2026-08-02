"""一次性修复：重算所有 board_kline_ths_*.parquet 的 pct_chg。

背景：boards_ths.get_board_kline_ths 增量更新时，重叠日 pct_chg 被置 NaN
并覆盖缓存原值（已修复）。本脚本对现存 90 个缓存文件按 close 逐日重算
pct_chg，把已腐蚀的 NaN 行补回。安全幂等，可重复运行。

用法：python scripts/repair_ths_kline_pctchg.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"


def main() -> None:
    files = sorted(CACHE.glob("board_kline_ths_*.parquet"))
    if not files:
        print("无 board_kline_ths_*.parquet，无需修复")
        return
    fixed = 0
    for f in files:
        df = pd.read_parquet(f)
        before_nan = int(df["pct_chg"].isna().sum())
        df = df.sort_values("date").reset_index(drop=True)
        df["pct_chg"] = df["close"].pct_change() * 100.0
        after_nan = int(df["pct_chg"].isna().sum())
        df.to_parquet(f, index=False)
        if before_nan != after_nan:
            fixed += 1
            print(f"  修复 {f.name}: pct_chg NaN {before_nan} -> {after_nan}")
    print(f"完成：{len(files)} 个文件，其中 {fixed} 个补回了 NaN 行（首行 NaN 为正常）")


if __name__ == "__main__":
    main()
