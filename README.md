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

# 2) 抓取板块列表/成分/板块指数（约 10-60 分钟，视源而定）
#    板块源由 config.yaml 的 board_source 决定：etf 用 fetch_etf_boards.py，
#    eastmoney 用 fetch_board_data.py，ths 用 fetch_ths_boards.py
python scripts/fetch_etf_boards.py   # 当前默认 board_source=etf（中证可交易ETF池）

# 3) 每日增量更新（已最新时秒级跳过；单日增量走腾讯快照约 1 分钟，
#    多日补抓回退 baostock 逐股约 20-40 分钟；启动时已不再自动执行，
#    改为看板界面「更新数据」按钮触发 /api/update 全流程，工作日 18:00 自动化亦会执行）
python scripts/fetch_etf_boards.py
python scripts/fetch_stock_history.py --update

# 4) 每日情绪报告 -> report/report_YYYYMMDD.md + scores_YYYYMMDD.csv
python main.py

# 5) 回测验证 -> report/backtest_YYYYMMDD.md
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

合成：各因子每日横截面 z-score（±3 截断，`config.yaml: score_z_clip`，防单一极端因子绑架综合分）→ 加权（`config.yaml: weights`）→ 5 日均值平滑为"周度情绪分"（决策用）。

## 信号规则（透明、非预测）

- **升温/降温榜**：情绪分连续 3 日单调上升/下降
- **亢奋/冰点**：情绪分时序分位 > 90% / < 10%（相对自身过去一年）
- **切换预警**：近 20 日曾居前 10 的板块正在降温，且有低位（分位<50%）板块升温
- **风格判定（regime 自适应，2026-08 新增）**：滚动已实现 IC（默认 15 日窗口、阈值 ±0.02，
  无前视——IC 序列后移 5 个交易日到"可知时点"）判定动量/反转/观望；回撤护栏（市场等权
  近 10 日累计收益 < -3%）强制观望。**单调性破坏预警**（`regime.mono_days`，2026-08-27 新增）：
  方向期间分层价差 Q5−Q1 连续 3 个可知日反向（高分组跑输低分组）即提前降级观望，
  比 IC 均值翻负约早一周，只降级不提前给方向。参数在 `config.yaml: regime`，日报榜单语义按判定结果
  自动标注（反转市 Top 榜=回避、Bottom 榜=关注；观望期仅观察）。
  `scripts/backtest_regime_adaptive.py` 默认只验证 `config.yaml` 的预注册配置；研究网格需显式
  `--research-grid`，结果不能直接覆盖生产参数。轮动层另有**方向确认迟滞**
  （`regime.confirm_readings`，非观望方向需连续 2 次调仓判定同向才生效，
  风控闸门不受迟滞影响）。`scripts/validate_robustness.py` 提供 HAC、
  区块 bootstrap 和时间顺序验证，避免用同一段行情反复调参。

## 数据口径与已知边界

- 数据源：东财 push2his（板块指数）、baostock（个股日线）。**不含北交所**（baostock 无此数据，占比很小）。
- 涨停判定：最高价涨幅触及涨停阈值且收盘封板（距最高价 0.2% 内）。阈值：主板 9.8%、创业板/科创板 19.8%、主板 ST 4.8%（创业板/科创板 ST 仍 20%）。优先用 baostock 按日 `isST` 字段还原历史戴帽/摘帽状态，缺省回退名称含 ST。
- 东财 push2（clist）接口对请求频率敏感，会间歇性拒绝连接；传输层已做退避重试与缓存降级（`emotion/transport.py`）。
- 板块成分为**最新快照**，回测存在幸存者偏差（历史成分变化不可考）。
- 打分/回测标的池由 `config.yaml: universe` 治理：默认剔除成分少于 20 只的极小板块、
  合并成分完全一致的重复层级板块；日报和看板会同时展示原始/入选数量，不能把它视为消除幸存者偏差。
- 信号层的“成交额价格代理”=成交额×涨跌幅，**不是真实资金净流入**，不进入情绪分；接入主动买卖盘数据前仅作辅助读数。
- 板块归属为**一股多板**：个股同时计入其所属的一/二/三级板块（如某芯片股同属电子、半导体、数字芯片设计），母板块与细分板块各自独立计分，细分热度可分辨。
- 因子体系围绕**涨停生态**构建（涨停强度/赚钱效应/晋级率占大头）：对游资连板题材敏感，对无涨停的大票机构趋势行情天然迟钝——后者往往成交额巨大但涨停家数少，需结合穿透信号与板块指数涨幅理解榜单。
- 情绪指标是**同步/短滞后指标**：它更快识别正在发生的切换，不是预言。
- 回测有效 ≠ 未来有效，存在过拟合与市场结构变化风险。
- 本项目仅供研究，不构成投资建议。

## 回测与验证纪律

运行 `python backtest.py` 生成多窗口回测。短于半年的窗口以累计收益为主，年化仅作参考；
板块等权线是**非可交易参考**，不是 ETF 实盘基准。`python scripts/validate_robustness.py`
会报告重叠收益的 HAC/Bootstrap 校正与按时间顺序的验证块。不同板块源、标的池规则、
成分快照和交易成本口径变化后，旧数值不能横向比较。

**使用纪律（重要）：**
1. 每天先看日报顶部的风格判定与“信号健康度”（近 60 日已实现 IC + 分层价差 Q5−Q1：
   价差<0 是“信号反”（分组倒置，低分组占优），价差≈0 且分层无序才是“信号死”）；
   观望时不做方向性轮动，方向翻转时 Top/Bottom 榜仅按日报标注理解。
2. 冰点、穿透和代理背离均是观察信号，不单独构成交易触发条件。
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

- **今日榜单**：信号健康度大灯（🟢/🟡/🔴）、趋势/极值/切换与穿透信号卡、
  情绪分 Top15/Bottom15 发散条形图、完整排名表（可排序、点行进详情）；
  **榜单与信号直接标注可交易 ETF**（如 `512480 半导体ETF`，来自 `config/etf_map.yaml`，
  板块池中大部分板块有对应 ETF，小众板块标"无直接ETF"；具体数量随 `board_source` 变化）
- **回测验证**：板块指数的模拟轮动（不等于 ETF 回测）、滚动 IC 走势、分层收益、三种轮动
  净值曲线、累计/年化绩效与重叠校正后的 IC 指标
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
python scripts/build_etf_map.py --source eastmoney  # 按当前东财 BK 板块代码重建映射
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
  fetch_ths_boards.py      同花顺板块数据（备选口径，仅供观察）
  fetch_sw_boards.py       申万二级行业数据（131 个，全市场分区，备选口径）
  fetch_etf_boards.py      可交易ETF板块池（中证行业/主题指数，当前主用口径）
  build_etf_trading_map.py 生成 指数→规模最大ETF 交易映射
  fetch_etf_klines.py      交易标的 ETF 日线（腾讯，前复权）
  backtest_etf_prices.py   用 ETF 真实价格复跑轮动回测
  refresh_board_cons.py    刷新东财成分（补齐被 100 条上限截断的板块）
  refresh_all_boards.py    全量刷新东财板块池
  fetch_etf_list.py        全量 ETF 清单（东财 clist + 新浪兜底）
  build_etf_map.py         生成板块↔ETF 映射 config/etf_map.yaml
  dump_dashboard_data.py   导出看板数据 -> data/dashboard/
  regime_check.py          分段验证不同 regime 下信号有效性
  diag_ic_decay.py         IC 衰减诊断：30 日分段 IC + 单因子归因（信号异常时先跑这个）
  diag_regime_flip.py      风格翻转诊断：分段涨停生态原始读数 + RPS 多空价差
                           （区分"生态退潮"与"动量→反转切换"，2026-08-27 新增）
  backtest_regime_adaptive.py  regime 自适应轮动回测（滚动 IC 方向 + 回撤护栏，无前视）
  repair_ths_kline_pctchg.py  一次性修复 THS K线 pct_chg 缓存
tests/             因子/合成/信号单测（合成数据，不依赖网络）
```

## 后续可选扩展

- 每日定时 + 邮件/飞书推送
- 概念板块（题材）口径第二池
- 板块 ↔ ETF 跟踪指数映射表，直接输出 ETF 层面信号
