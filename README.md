# Emotion-A：A 股板块情绪量化系统

分板块情绪量化：每日输出各板块情绪分排名与切换信号，并用历史回测验证信号有效性，
为板块 ETF 的波段操作（1-4 周）提供数据参考。

方法论借鉴：
- 国泰海通《从涨停板、"打板策略"到赚钱效应引发的情绪择时指标》的因子体系
  （参考实现：therealXiaomanChu/Sentiment-timing-report-reappear）
- vibe-astock 的指标口径（中位数优先、晋级率、"多数人体感"原则）

全部纯计算、无 LLM、数字可复现。

## 快速开始

```bash
pip install -r requirements.txt

# 1) 抓取个股历史日线（约 30-60 分钟，可断点续跑）
python scripts/fetch_stock_history.py 2023-01-01

# 2) 抓取板块列表/成分/板块指数（约 10 分钟）
#    板块源由 config.yaml 的 board_source 决定：eastmoney 用 fetch_board_data.py，ths 用 fetch_ths_boards.py
python scripts/fetch_board_data.py   # 当前默认 board_source=eastmoney

# 3) 每日情绪报告 -> report/report_YYYYMMDD.md + scores_YYYYMMDD.csv
python main.py

# 4) 回测验证 -> report/backtest_YYYYMMDD.md
python backtest.py

# 测试
python -m pytest tests/ -q
```

## 因子体系（全部按板块聚合）

| 因子 | 口径 | 方向 |
|---|---|---|
| 涨停强度 | 板块内涨停家数 / 板块内交易家数 | 正 |
| 赚钱效应 | 昨日涨停股今日涨幅**中位数** | 正 |
| 涨跌宽度 | (涨家数 − 跌家数) / 交易家数 | 正 |
| 相对强度 RPS | 板块 20 日涨幅在全板块的分位 | 正 |
| 热度变化 | 板块成交额占全市场比例的 5 日变化 | 正 |
| 晋级率 | 昨日涨停股今日继续涨停的比例（权重减半） | 正 |
| 资金流 | 东财主力净流入（接口不稳定时为可选，缺失自动剔除） | 正 |

合成：各因子每日横截面 z-score → 加权（`config.yaml: weights`）→ 5 日均值平滑为"周度情绪分"（决策用）。

## 信号规则（透明、非预测）

- **升温/降温榜**：情绪分连续 3 日单调上升/下降
- **亢奋/冰点**：情绪分时序分位 > 90% / < 10%（相对自身过去一年）
- **切换预警**：近 20 日曾居前 10 的板块正在降温，且有低位（分位<50%）板块升温

## 数据口径与已知边界

- 数据源：东财 push2his（板块指数）、baostock（个股日线）。**不含北交所**（baostock 无此数据，占比很小）。
- 涨停判定：最高价涨幅触及涨停阈值且收盘封板（距最高价 0.2% 内）。阈值：主板 9.8%、创业板/科创板 19.8%、主板 ST 4.8%（创业板/科创板 ST 仍 20%）。优先用 baostock 按日 `isST` 字段还原历史戴帽/摘帽状态，缺省回退名称含 ST。
- 东财 push2（clist）接口对请求频率敏感，会间歇性拒绝连接；传输层已做退避重试与缓存降级（`emotion/transport.py`）。
- 板块成分为**最新快照**，回测存在幸存者偏差（历史成分变化不可考）。
- 情绪指标是**同步/短滞后指标**：它更快识别正在发生的切换，不是预言。
- 回测有效 ≠ 未来有效，存在过拟合与市场结构变化风险。
- 本项目仅供研究，不构成投资建议。

## 回测结论（已扣万5换仓成本）

**关键发现：信号方向随市场风格（regime）翻转，必须短窗口使用。**

| 时段 | IC（5日） | IC胜率 | 分层形态 | 结论 |
|---|---|---|---|---|
| 9·24 之前（2023-01~2024-09） | -0.037 | 44% | 高情绪→跑输 | 反向（缩量市） |
| 9·24 之后至 2025 年末 | -0.001 | 51% | 无区分度 | 失效 |
| **最近 180 交易日** | **+0.077** | **61%** | Q1→Q5 单调递增 | **正向有效（动量市）** |

最近 180 天周度轮动（东财 100 行业板块口径，2026-08-02 修复数据重跑）：

> ⚠️ **口径敏感**：同一窗口切到同花顺 90 板块口径时，IC(5日) 约 +0.017、Top3 年化约 -13%，
> **低于 0.03 显著性阈值**，榜单仅供情绪观察、不可用于交易。
> 下方数字仅代表东财 100 口径；改 `board_source` 后必须重跑回测，结论不可直接套用。

| 策略 | 年化 | 夏普 | 最大回撤 | 对照基准 |
|---|---|---|---|---|
| 买入情绪 Top3（顺势追涨） | **+24.4%** | 0.72 | -41.8% | +0.3% |
| 买入情绪 Bottom3（抄底） | -16.8% | -0.36 | -28.8% | +0.3% |
| 回避 Top3 | -0.4% | 0.13 | -23.6% | +0.3% |

**使用纪律（重要）：**
1. 信号**只在当前动量风格下顺势使用**——每天先看日报顶部的"信号健康度"
   （近 60 日滚动 IC）：🟢 顺势有效 / 🔴 风格已切换，榜单需反向理解或停用
2. 抄底（买冰点）**在任何 regime 下都亏损**，冰点榜只作观察
3. 情绪分是短窗口指标，`score_lookback`/`backtest_window` 不宜超过 180 天；
   老数据（9·24 前）会污染信号方向
4. -33% 量级的回撤是常态，仓位需自行控制

## Web 看板

工业级深色看板（纯本地、无 AI、无外部 CDN 依赖），浏览器查看榜单、回测与板块详情。

```bash
pip install fastapi uvicorn

# 1) 首次使用：导出看板数据（离线跑完整管线，约 1-2 分钟）
python scripts/dump_dashboard_data.py

# 2) 启动看板 -> 浏览器打开 http://127.0.0.1:8000
python server.py
```

- **今日榜单**：信号健康度大灯（🟢/🟡/🔴）、升温/降温/亢奋/冰点/切换预警五张信号卡、
  情绪分 Top15/Bottom15 发散条形图、完整排名表（可排序、点行进详情）；
  **榜单与信号直接标注可交易 ETF**（如 `512480 半导体ETF`，来自 `config/etf_map.yaml`，
  板块池中大部分板块有对应 ETF，小众板块标"无直接ETF"；具体数量随 `board_source` 变化）
- **回测验证**：一句话结论（话术为"跟着前 3 名板块对应的 ETF 买"，实际回测用板块指数
  近似代替 ETF）、滚动 IC 走势、分层收益、三种轮动净值曲线、绩效指标表
- **板块详情**：对应 ETF 卡片（代码/名称/最新价/成交额）、板块龙头卡片（流动性龙头
  Top3：近 5 日日均成交额；情绪龙头：近 10 日最高连板）、情绪分 vs 板块指数双轴图、
  时序分位面积图、六因子体检卡；**无 ETF 的板块在榜单/信号卡上直接显示龙头股 pill**
- **刷新数据**：页面右上角"刷新数据"按钮会后台重跑导出脚本（等价于手动运行
  `python scripts/dump_dashboard_data.py`），完成后自动重载页面数据；
  东财行情接口不可达时脚本自动降级为纯缓存离线计算（约 1 分钟）
- 看板数据落在 `data/dashboard/`（`server.py` 只读该目录，启动快；缺失时会提示先跑导出脚本）

### ETF 清单与映射维护

```bash
python scripts/fetch_etf_list.py   # 抓全量 ETF（东财 clist，被风控时自动用新浪兜底）-> data/cache/etf_list.parquet
python scripts/build_etf_map.py    # 人工主力ETF + 关键词自动匹配 -> config/etf_map.yaml（可手工编辑）
python scripts/dump_dashboard_data.py  # 编辑映射后重跑导出即可生效
```

## 目录

```
main.py            每日情绪报告
backtest.py        回测验证
server.py          Web 看板后端（FastAPI，:8000）
config.yaml        权重/参数
static/            看板前端（index.html / app.js / vendor/echarts.min.js）
emotion/
  transport.py     弹性 HTTP 层（重试/降级/浏览器指纹）
  data/            板块、行情、缓存
  factors.py       7 个板块情绪因子
  score.py         z-score 合成 + 平滑 + 时序分位
  signal.py        升温/降温/切换信号
  report.py        Markdown/CSV 报告
  backtest.py      IC / 分层 / 周度轮动回测
  leaders.py       板块龙头股（流动性龙头 / 情绪龙头连板）
  pipeline.py      数据装配
scripts/
  fetch_stock_history.py   baostock 全市场日线（后台跑）
  fetch_board_data.py      东财板块数据
  fetch_ths_boards.py      同花顺板块数据（当前默认 board_source=ths）
  refresh_board_cons.py    刷新东财成分（补齐被 100 条上限截断的板块）
  refresh_all_boards.py    全量刷新东财板块池
  fetch_etf_list.py        全量 ETF 清单（东财 clist + 新浪兜底）
  build_etf_map.py         生成板块↔ETF 映射 config/etf_map.yaml
  dump_dashboard_data.py   导出看板数据 -> data/dashboard/
  regime_check.py          分段验证不同 regime 下信号有效性
  repair_ths_kline_pctchg.py  一次性修复 THS K线 pct_chg 缓存
tests/             因子/合成/信号单测（合成数据，不依赖网络）
```

## 后续可选扩展

- 每日定时 + 邮件/飞书推送
- 概念板块（题材）口径第二池
- 板块 ↔ ETF 跟踪指数映射表，直接输出 ETF 层面信号
