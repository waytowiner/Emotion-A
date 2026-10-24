# Code Review 修复清单（2026-08-02）

对应 `CODE_REVIEW_20260802.md` 的问题项，逐条修复状态。最终 `pytest` **35 passed**，git 初始提交 `ba4d353`。

## P0 / P1 BUG

| # | 问题 | 修复 | 文件 |
|---|---|---|---|
| 1 | 红测试 `test_turnover_heat_market_total`（pandas 2.x 无名索引对齐报错） | `amt.div(total, level="date")` 显式按 level 广播 | emotion/factors.py |
| 2 | THS K线增量更新腐蚀 pct_chg（90 个缓存最新日全空） | 合并后按 close 统一重算 pct_chg + 一次性修复脚本（已跑，90 个全部补回） | emotion/data/boards_ths.py、scripts/repair_ths_kline_pctchg.py |
| 3 | ST 判定用当前名称快照回贴历史 + 创业板/科创板 ST 被错用 4.8% | limit_threshold 修正（30/68 ST 仍 20%，仅主板 ST 5%）；flag_limit_up 优先用按日 isST 字段；fetch FIELDS 加 preclose/isST | emotion/factors.py、scripts/fetch_stock_history.py、emotion/data/quotes.py |
| 4 | `--refresh-boards` 对 THS 源无效 | ths 分支透传 refresh_boards 到 list/cons/kline | emotion/pipeline.py |
| 5 | 成本口径"单边"注释与实现差 2 倍 | 注释统一为"往返换仓成本"（实现不变，口径自洽） | config.yaml、emotion/backtest.py |

## P2 工程问题

| # | 问题 | 修复 | 文件 |
|---|---|---|---|
| 6 | 无 git 版本控制 | git init + .gitignore（排除 data/report/.workbuddy/__pycache__）+ 初始提交 | .gitignore |
| 7 | requirements 无上限、隐式传递依赖 | 钉版本上限 + 显式声明 requests/py_mini_racer/lxml | requirements.txt |
| 8 | flag_limit_up 行级 apply 性能（~400 万行） | 向量化（np.where 阈值 + preclose 优先） | emotion/factors.py |
| 9 | promotion_rate 未过滤当日停牌（与赚钱效应口径不一致） | 加 trade_status=="1" 过滤 | emotion/factors.py |
| 10 | load_stock_chunks 缺 (code,date) 去重保护 | drop_duplicates(keep="last") | emotion/pipeline.py |
| 11 | pipeline ths 分支静默吞异常 | cons/kline 失败计数 + 前 3 条告警 + 汇总 | emotion/pipeline.py |
| 12 | build_etf_map ALLOW_QDII 空操作死代码 + "100 板块池"硬编码 | 删死代码、文案通用化 | scripts/build_etf_map.py |
| 13 | server on_event 弃用告警 | 改 lifespan 上下文 | server.py |
| 14 | 产物无口径标签（东财/THS 混存易误读） | latest.json/meta.json 加 board_source/n_boards；报告头标注板块口径 | scripts/dump_dashboard_data.py、emotion/report.py、main.py |
| 15 | README 多处过期 | ST 阈值说明、回测表加口径敏感警告、ETF 数量口径化、fetch 脚本说明、目录补全 5 个脚本 | README.md |

## 新增测试
- `tests/test_factors.py::test_st_on_chinext_star_still_20`：锁定创业板/科创板 ST 阈值仍为 20%。

## 待后续处理（需用户决策或耗时操作）
- **重抓个股数据**：preclose/isST 字段需重跑 `fetch_stock_history.py`（约 30-60 分钟）才真正生效；flag_limit_up 已做缺省回退，旧数据行为不变。
- **每日更新自动化**：Code Review 建议配 automation，但属定时资源消耗型任务，待你确认是否创建。
- **THS 口径下的回测结论**：当前 THS 口径 IC=0.017 低于阈值，建议在 THS 口径下重新校准使用纪律（README 已加警告，具体结论待你定夺）。
