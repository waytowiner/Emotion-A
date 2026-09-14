# P0 突变穿透规则落地 + 90 日回测验证

## 做了什么

在 **不改因子体系、不改平滑窗口** 的硬约束下，为 `emotion/signal.py` 增加 3 条突变穿透规则，并完成量化验证与每日自动化接线。

### 1. signal.py 三条 P0 规则（核心）

- **底部反转预警** `bottom_reversal`：冰点（时序分位 < 0.1）+ 资金横截面 z > 2.0
- **极端值预警** `extreme_value`：极端分位（亢奋/冰点）+ 任一因子 |横截面 z| > 2σ
- **资金-情绪背离** `fund_emotion_divergence`：资金连续 2 日为正 + 情绪分（平滑）为负

关键设计：新增 `compute_board_fund_flow()` 作为**信号层穿透指标**（口径同 `daily_anomaly.py`：`amount * pct_chg / 100` 按板块求和），**不注入 `score.combine_factors`**，因此因子体系（6 因子合成）与 `smooth_window=5` 零改动。三条规则均用原始日度读数、不经平滑——平滑会抹平突变。

### 2. daily_anomaly 纳入每日自动化

`run.bat` 在 `main.py` 后插入 `scripts\daily_anomaly.py`，每日批次：main.py → daily_anomaly.py → dump_dashboard_data.py → server.py。（2026-08-07 起 run.bat 改为只启动看板服务，上述批次由看板「更新数据」按钮触发 `/api/update` 全流程执行。）

### 3. 90 日回测（量化验证，调阈值前）

`scripts/backtest_p0_signals.py` → `report/p0_signal_backtest_20260804.md`，最近 90 交易日、5/10/20 日前瞻：

| 规则 | 触发 | 20日 vs 市场 | 20日 vs 对照 | 判定 |
|---|---|---|---|---|
| 底部反转 | 少(1.2/日) | **+5.95%** | **+8.07%** | ✅ 穿透有效，优于只买冰点 |
| 极端值 | 稳(3.4/日) | **+2.75%** | **+1.62%** | ✅ 单因子偏离过滤有效 |
| 资金-情绪背离 | 多(9.6/日) | **-1.40%** | +0.07% | ⚠️ 动量市下抄负情绪是陷阱 |

## 关键结论 / 后续

- **规则 1、2**：维持 P0 缺省阈值（z>2.0、σ>2.0）即可，无需调整。
- **规则 3**：当前 IC=+0.104 顺势环境下作"买点"亏损，应作**观察/风险提示**而非买点；建议 `fund_positive_days` 2→3 收紧重测，或待 regime 切换（IC 转负）再启用为反转买点。这是调阈值的明确方向，已写入回测报告"阈值调参建议"。
- 接线：`main.py`/`report.py` 已传入 factors+fund_flow 并渲染三条新信号；今日日报已实际命中（极端值 4 板块、背离 17 板块、底部反转 0）。
- 测试：`tests/test_score_signal.py` 新增 `TestBreakthrough` 5 例，全量 **40 测试通过**，无回归。

## 改动文件

- `emotion/signal.py`（+ compute_board_fund_flow、+3 规则、扩展签名）
- `config.yaml`（+ signal.breakthrough 段）
- `main.py`（传 factors/fund_flow/breakthrough_cfg）
- `emotion/report.py`（+3 信号渲染行 + 穿透说明）
- `run.bat`（+ daily_anomaly.py）
- `scripts/backtest_p0_signals.py`（新增）
- `tests/test_score_signal.py`（+ TestBreakthrough，修 empty_input 契约）
- `report/p0_signal_backtest_20260804.md`（回测产物）
