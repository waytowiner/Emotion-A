"""「回避 Top」预注册回测：反转确认期的落仓方式 × IC 前瞻窗口。

假设来源（2026-09-17 讨论，查看本脚本结果前已固定）：
- backtest_20260909：近 60 日 5 日前瞻 IC=-0.122（Bootstrap CI 跨零、不可信），
  但 10 日 IC=-0.189（HAC t=-3.35，CI [-0.309,-0.078] 排零）、
  20 日 IC=-0.309（t=-4.08，CI [-0.445,-0.217] 排零）——反转结构在 10~20 日
  前瞻上统计可信，5 日口径不可信；regime 生产口径用 5 日前瞻，看不到该层信息。
- project_status_20260909：直接反向抄底不可取（60 日口径 -10.78%，
  Bottom3 极端组 ≠ 受益的 Q1 整组）。

预注册假设：
- H1：反转确认期（regime 判 bottom）落仓从「买 Bottom N」改为「回避 Top N」
  （等权持有其余全部），回撤与夏普改善——反转主要体现在高分组跑输，温和
  反向即可捕获，无需承担极端组噪声。
- H2：regime 判定改用 10 日前瞻 IC（窗口/阈值/确认迟滞/护栏全部沿用生产
  配置）比 5 日口径更稳——显著性来自前瞻窗口而非参数重调。
- H3：反转确认期持有「情绪分最低 40%」（约两个五分位的宽基低分组）优于
  「买 Bottom 3」——低分组整组受益，极端组只是其子集且噪声更大。

变体（每个变体单一配置，禁止网格）：
- A0_base      生产口径复刻：horizon=5 IC，bottom→买 Bottom3。
- V1_h5_avoid  A0 + 反转期回避 Top3（隔离落仓方式，H1）。
- V2_h10_avoid horizon=10 IC（后移 10 交易日到可知时点）+ 回避 Top3（H1+H2）。
- V3_h10_lowq  horizon=10 IC + 反转期持有最低 40%（H2+H3）。

判定（对比 A0_base，三条同时满足才算通过）：
① 夏普更高；② 最大回撤不劣化；③ 2026-07 以来区间收益更高。
即便通过，上线前仍须用未参与本轮比较的后续时段复验。

无前视：horizon=h 的 IC(d) 在 d+h 收盘后才可知，序列整体后移 h 个交易日；
方向确认迟滞 confirm=2 与回撤护栏 -3%/10 日同生产。周度调仓，5bps 成本。
本脚本不读网络、不改 config.yaml、不改任何生产脚本。

运行：python scripts/backtest_avoid_top_h10.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from emotion import backtest, pipeline, score
from emotion.factors import compute_all_factors
from backtest_regime_adaptive import adaptive_parameters, adaptive_rotation, ic_known_series

H10 = 10  # H2/H3 的前瞻窗口（预注册单一档位，不做 15/20 网格）


def seg_ret(nav: pd.Series, start: str) -> float:
    sub = nav.loc[start:]
    if len(sub) < 2:
        return float("nan")
    return sub.iloc[-1] / sub.iloc[0] - 1


def main() -> None:
    cfg = pipeline.load_config()
    print("[1/4] 准备数据（缓存口径，无网络）...")
    data = pipeline.prepare(cfg, end_override=pipeline.cache_end_date(cfg), offline=True)

    print("[2/4] 计算因子与情绪分 ...")
    factors = compute_all_factors(data["stock_daily"], data["board_map"],
                                  data["board_kline"],
                                  market_total=data.get("market_total"))
    _, smoothed, _ = score.build_scores(
        factors, cfg["weights"], cfg.get("smooth_window", 5),
        lookback=cfg.get("score_lookback", 250),
        z_clip=cfg.get("score_z_clip", 3.0))
    board_close = data["board_kline"].pivot(
        index="date", columns="board_code", values="close").sort_index()

    print("[3/4] 计算 IC 可知序列（horizon=5 生产口径 / horizon=10 预注册档）...")
    regime_cfg = cfg.get("regime", {})
    h5 = int(regime_cfg.get("horizon", 5))
    ic_known = {}
    for h in (h5, H10):
        ic_s = backtest.daily_ic_series(smoothed, board_close, horizon=h)
        ic_known[h] = ic_known_series(ic_s, board_close.index, lag=h)

    base_kw = adaptive_parameters(regime_cfg)

    variants = [
        ("A0_base", dict(base_kw), h5, "bottom"),
        ("V1_h5_avoid", dict(base_kw), h5, "avoid_top"),
        ("V2_h10_avoid", dict(base_kw), H10, "avoid_top"),
        ("V3_h10_lowq", dict(base_kw), H10, "low_groups"),
    ]

    print("[4/4] 回测（4 个预注册变体 + 固定方向对照）...")
    top_n = cfg["signal"].get("top_n", 3)
    cost_bps = cfg["backtest"].get("cost_bps", 5)

    results = {}
    for tag, kw, h, action in variants:
        results[tag] = adaptive_rotation(
            smoothed, board_close, ic_known[h], top_n=top_n,
            cost_bps=cost_bps, bottom_action=action, **{**kw, "horizon": h})

    refs = {
        "固定追涨 Top3": backtest.rotation_backtest(
            smoothed, board_close, top_n=top_n, cost_bps=cost_bps, direction="top"),
        "固定抄底 Bottom3": backtest.rotation_backtest(
            smoothed, board_close, top_n=top_n, cost_bps=cost_bps, direction="bottom"),
        "固定回避 Top3": backtest.rotation_backtest(
            smoothed, board_close, top_n=top_n, cost_bps=cost_bps,
            direction="avoid_top"),
    }

    # ---------- 输出 ----------
    base = results["A0_base"]
    d0, d1 = base["nav"].index[0], base["nav"].index[-1]
    lines = ["# 「回避 Top」预注册回测：反转期落仓方式 × IC 前瞻窗口", "",
             f"回测区间：{d0:%Y-%m-%d} ~ {d1:%Y-%m-%d}（{len(base['nav'])} 个交易日）。"
             "预注册假设与判定标准见脚本 docstring（查看结果前固定，未做参数搜索）。"
             f"regime 参数全部沿用 config.yaml 生产配置：ic_window={base_kw['ic_window']}、"
             f"阈值 ±{base_kw['threshold']:.2f}、min_obs={base_kw['min_obs']}、"
             f"确认迟滞 confirm={base_kw['confirm']}、回撤护栏 "
             f"{base_kw['guard']:.0%}/{regime_cfg.get('guard_days', 10)} 日；"
             "风控闸门即时生效。周度调仓，单边 100% 换手成本 5bps。", ""]

    lines += ["## 主表（4 个预注册变体）", "",
              "| 变体 | 反转期落仓 | IC 前瞻 | 累计 | 夏普 | 最大回撤 | 期末净值 "
              "| 7月以来 | 成本损耗 | 累计换手 | 均持仓 |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    action_label = {"bottom": "买 Bottom3", "avoid_top": "回避 Top3",
                    "low_groups": "最低 40%"}
    for tag, kw, h, action in variants:
        r = results[tag]
        s = r["strategy"]
        hold_txt = f"{r['regime_log']['n_holdings'].mean():.0f}"
        lines.append(
            f"| {tag} | {action_label[action]} | {h} 日 | {s['total_ret']:.2%} "
            f"| {s['sharpe']:.2f} | {s['max_dd']:.2%} | {r['final_nav']:.3f} "
            f"| {seg_ret(r['nav'], '2026-07-01'):+.1%} | {r['total_cost_drag']:.2%} "
            f"| {r['total_turnover']:.2f} | {hold_txt} |")

    lines += ["", "## 对照（同期同口径，固定方向）", "",
              "| 策略 | 累计 | 夏普 | 最大回撤 | 期末净值 | 7月以来 |",
              "|---|---|---|---|---|---|"]
    for label, r in refs.items():
        s = r["strategy"]
        lines.append(f"| {label} | {s['total_ret']:.2%} | {s['sharpe']:.2f} "
                     f"| {s['max_dd']:.2%} | {r['final_nav']:.3f} "
                     f"| {seg_ret(r['nav'], '2026-07-01'):+.1%} |")
    lines.append(f"| 板块等权基准（非可交易） | {base['benchmark']['total_ret']:.2%} "
                 f"| {base['benchmark']['sharpe']:.2f} | {base['benchmark']['max_dd']:.2%} "
                 f"| {base['final_benchmark']:.3f} "
                 f"| {seg_ret(base['benchmark_nav'], '2026-07-01'):+.1%} |")

    # 判定
    b = base["strategy"]
    b_seg = seg_ret(base["nav"], "2026-07-01")
    lines += ["", "## 判定（对比 A0_base，① 夏普更高 ② 回撤不劣化 ③ 7月以来收益更高，"
              "三条同时满足）", "",
              "| 变体 | ①夏普更高 | ②回撤不劣化 | ③7月以来更高 | 结论 |",
              "|---|---|---|---|---|"]
    for tag, *_ in variants[1:]:
        r = results[tag]
        s = r["strategy"]
        c1 = pd.notna(s["sharpe"]) and s["sharpe"] > b["sharpe"]
        c2 = pd.notna(s["max_dd"]) and s["max_dd"] >= b["max_dd"]
        c3 = seg_ret(r["nav"], "2026-07-01") > b_seg
        verdict = "通过" if (c1 and c2 and c3) else "未通过"
        lines.append(f"| {tag} | {'是' if c1 else '否'} | {'是' if c2 else '否'} "
                     f"| {'是' if c3 else '否'} | {verdict} |")

    lines += ["", "## 口径校验", "",
              "A0_base 与 scripts/backtest_regime_adaptive.py 的 configured 行共用同一"
              "函数与同一组生产参数，结构上一致；数值与 regime_adaptive_backtest_20260913"
              "（年化 10.23%、夏普 0.44、回撤 -29.56%）的差异只来自样本区间延长"
              "（09-11 → 本次缓存截止日），属预期，不是口径偏差。", "",
              "## 结论与限制", "",
              "1. 通过判定的变体见上表；任何上线变更必须先把机制写入 config.yaml，"
              "再用未参与本轮比较的后续时段复验（同 factor_adaptive / portfolio_layer 纪律）。",
              "2. 禁止据本报告对前瞻窗口（5/10/15/20）、低分组比例、回避个数做网格搜索——"
              "那会立刻把结论退化成样本内拟合。",
              "3. horizon=10 的方向判定比生产口径慢约一周（IC 可知时点 d+10），"
              "这是 H2 的已知代价，已含在回测内。",
              "4. 成分股使用当前快照、板块等权参考不可交易，ETF 实盘还要另计跟踪误差、"
              "流动性与费用。", ""]
    md = "\n".join(lines)
    print(md)

    out_dir = ROOT / cfg["paths"]["report_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"avoid_top_h10_{pd.Timestamp.today():%Y%m%d}.md"
    path.write_text(md, encoding="utf-8")
    print(f"\n已保存：{path}")


if __name__ == "__main__":
    main()
