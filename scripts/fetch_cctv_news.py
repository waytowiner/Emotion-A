"""抓取央视新闻联播文字稿：python scripts/fetch_cctv_news.py

用法：
  python scripts/fetch_cctv_news.py                    # 增量：缓存最后日期 -> 今天
  python scripts/fetch_cctv_news.py --start 2025-09-01 # 首次回填
  python scripts/fetch_cctv_news.py --date 20261023    # 单日补抓
  python scripts/fetch_cctv_news.py --refresh          # 忽略缓存强制重抓

缓存一天一文件（data/cache/cctv_news/cctv_YYYYMMDD.parquet），
可断点续跑；连续 8 次失败自动中止，直接重跑即可。
政策支柱数据源，与情绪主线完全平行。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from emotion.data import news_cctv

CACHE = ROOT / "data" / "cache"
DEFAULT_START = "2025-09-01"  # 与 config history.start_date 对齐（历史分位起点）


def main() -> None:
    ap = argparse.ArgumentParser(description="央视新闻联播文字稿抓取（政策支柱）")
    ap.add_argument("--start", help=f"回填起始日（默认 {DEFAULT_START}，仅当无缓存时使用）")
    ap.add_argument("--date", help="单日补抓（YYYYMMDD 或 YYYY-MM-DD）")
    ap.add_argument("--end", help="结束日（默认今天）")
    ap.add_argument("--refresh", action="store_true", help="忽略已有缓存强制重抓")
    args = ap.parse_args()

    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    end = args.end or today

    if args.date:
        ds = str(args.date).replace("-", "")
        ds = f"{ds[:4]}-{ds[4:6]}-{ds[6:8]}"
        print(f"[cctv] 单日抓取 {ds} ...")
        df = news_cctv.fetch_day(CACHE, ds, refresh=args.refresh)
        print(f"[cctv] {ds}：{len(df)} 条"
              + ("（已有缓存）" if not args.refresh and len(df) else ""))
        return

    last = news_cctv.last_cached_date(CACHE)
    if args.start:
        start = args.start
    elif last:
        start = last  # 增量：从缓存最后日期起（当日已缓存会自动跳过）
    else:
        start = DEFAULT_START
    print(f"[cctv] 抓取区间 {start} ~ {end}（缓存最新：{last or '无'}）")
    n = news_cctv.fetch_range(CACHE, start, end, refresh=args.refresh)
    print(f"[cctv] 完成：本次新抓 {n} 天。缓存目录：{CACHE / news_cctv.NEWS_DIR}")


if __name__ == "__main__":
    main()
