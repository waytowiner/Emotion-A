# Emotion-A 项目工程评估报告

评估日期：2026-08-02 ｜ 范围：全部 14 个 emotion/ 模块、9 个 scripts、4 个测试、server/static、data 缓存实物核查
评估方式：逐文件代码审查 + 运行测试套件 + 用本机 Python 对疑点逐一复现验证（非纸面推测）

---

## 一、总体评价

这是一个**完成度相当高的个人量化项目**，不是玩具代码：

- 分层清晰：transport → data → factors → score → signal → report/backtest，依赖方向单向
- 回测执行口径严谨：T+1 调仓不前视、调仓日取每周真实最后交易日，且有测试锁定这两个不变式
- 工程韧性用心：三通道传输层（curl/curl_cffi/requests）、缓存降级、断点续跑、离线兜底
- README 诚实披露了幸存者偏差、regime 依赖、同步指标属性，没有夸大
- 有 regime 感知设计（信号健康度大灯），这比绝大多数同类项目强

但存在 **1 个动摇结论可信度的关键发现**、**若干真实 BUG** 和一批工程卫生问题。

---

## 二、关键发现（P0）：策略有效性对板块口径极度敏感，README 头部结论在当前配置下不成立

实测对比（同一窗口 2025-11-04 ~ 2026-07-30，同一套因子与合成逻辑）：

| 口径 | IC(5日) | IC胜率 | Top3 轮动年化 | 夏普 | 期末净值 | 来源 |
|---|---|---|---|---|---|---|
| 东财 100 板块 | +0.072 | 60.8% | **+101.7%** | 1.86 | 1.677 | report/backtest_20260731.md |
| 同花顺 90 板块（当前 config） | **+0.017** | 55.1% | **+6.2%** | 0.35 | 1.045 | data/dashboard/backtest.json |

- 当前 `config.yaml` 为 `board_source: ths`，在该口径下 IC(5日)=0.017，**低于项目自己定的 |0.03| 显著性阈值**，分层单调性也大概率不再成立；
- README 头部的 "+101.7% 年化 / 正向有效" 结论来自东财 100 板块口径（细粒度，含"学历教育""电商服务"等小板块），切换口径后结论垮塌——**策略的 alpha 很大程度上来自板块宇宙的切分方式，而非稳健的规律**；
- 当前看板的"信号健康度"显示 🟢（近 56 日 IC=+0.050），与 180 日回测 IC=0.017 并存——说明近 60 日是近期最强区间，健康度大灯本身也偏乐观；
- report/ 目录里的 7-30 日报（东财口径 100 板块）与 data/dashboard 的同日榜单（THS 口径 90 板块，Top 排名完全不同）**混存且无口径标注**，容易误读。

**建议**：
1. 把"板块宇宙"当作一等公民纳入配置与产物命名（报告/看板显著标注口径与板块数）；
2. 结论性数字（README、报告头）必须附带口径标签，过期口径的结论应撤下或标注；
3. 把 regime_check 扩展为"口径×窗口"二维稳健性检查，作为每周体检；
4. 在 THS 口径下重新撰写使用纪律（当前纪律是基于东财口径得出的）。

---

## 三、确认的 BUG（按严重度）

### P0-1 测试套件当前是红的（已在 pandas 2.3.0 复现）
`pytest tests/` → **1 failed, 33 passed**：
`test_turnover_heat_market_total` 抛 `ValueError: cannot join with no overlapping index names`。

机制：`factors.factor_turnover_heat` 中 `amt / total`，amt 是 (date, board_code) MultiIndex Series，total 是 date 索引 Series。pandas 2.3 下该除法**依赖 total 的索引名恰好叫 "date"** 才能广播。生产路径因 `groupby("date")` 继承了索引名而侥幸工作；测试中构造的 Series 索引无名 → 直接炸。

- 修复一行：`amt.div(total, level="date")`（对无名索引同样健壮）；
- 同时建议给 requirements 加版本上限（见 P2），这个错就是 pandas 升级后行为变化引入的。

### P1-1 THS 板块 K 线增量更新腐蚀 pct_chg（90/90 缓存文件全部中招）
`boards_ths.get_board_kline_ths`：增量时 `new["pct_chg"] = new["close"].pct_change()*100`，new 首行（重叠日）pct_chg 必为 NaN；`concat + drop_duplicates(keep="last")` 让 NaN 行**覆盖**缓存里原来的有效值。实测全部 90 个 `board_kline_ths_*.parquet` 最新交易日的 pct_chg 均为 NaN。

- 影响面：回测/RPS 用 close 自算，不受影响；但 server `/api/board/{code}` 把该列透传给看板，**板块详情最新一天的涨跌幅恒为空**；
- 修复：concat 之后按 date 排序统一重算 `pct_chg = close.groupby.pct_change()`，或合并前丢掉 new 的首行 NaN。

### P1-2 ST 判定用"当前名称快照"回贴全部历史
`fetch_stock_history.py` 只保存当前 `code_name`，`limit_threshold` 按名称含 ST 判 4.8%。后果：

- 历史上戴帽、现已摘帽的股：历史涨停（5%）全部漏判；
- 历史上正常、现被 ST 的股：历史中 +5%~+9.8% 的日子被误判为涨停（假阳性）；
- 另外 **创业板/科创板 ST 股涨跌停仍为 20%**，当前实现对所有 ST 一刀切 4.8%（30/68 开头的 ST 股阈值错误）。

实测 chunk_000 中有 18 只 ST 股、15570 行全部回贴当前名。三年窗口内 ST 状态变动的股票估计上百只，属于系统性（虽小幅）污染涨停类因子（涨停强度/赚钱效应/晋级率）。

- 修复：baostock `query_history_k_data_plus` 支持 `isST` 字段，按日存储并按日判定；
- 同时修正 ST 阈值逻辑：仅主板 ST 用 4.8%。

### P1-3 `--refresh-boards` 对 THS 源完全无效
`pipeline.prepare` 的 ths 分支调用 `get_board_list_ths(cache_dir)` / `get_board_cons_ths(cache_dir, code)` **均未透传 `refresh=refresh_boards`**，main.py 的命令行参数成了摆设。成分股是静态快照，长期不刷新会让新纳入股票缺席（叠加 README 已声明的幸存者偏差，口径更漂）。

### P1-4 换仓成本口径自相矛盾（2 倍偏差）
`config.yaml` 注释 `cost_bps: 5  # 单边成本`，但 `rotation_backtest` 对换仓只扣一次 `cost*changed/N`（卖旧+买新实际应为 2×）。README 宣称"已扣万5换仓成本"——按实现实际是"往返合计万5"。经济结果影响不大（全周期拖累约 1.3%），但**标签与实现相差 2 倍**，且测试 `test_switch_charges_cost` 把错误口径锁成了"正确"。建议统一为"往返成本"或实现双边扣费。

---

## 四、工程问题（P2）

**版本与依赖**
- 项目没有 git 版本控制（`git status` 确认非仓库）——对方法论持续迭代的量化项目风险很高，无法回溯"结论是用哪版代码算的"；
- `requirements.txt` 全部只有下限无上限（pandas>=2.0 即本次红测试的成因）；`py_mini_racer`、`lxml/html5lib`（pd.read_html 需要）、`requests` 依赖 akshare 传递安装，未显式声明。建议生成 lock（pip-tools/uv）。

**数据管线**
- THS 成分股匿名抓取最多 5 页=100 只/板块，大板块被截断；代码注释写"口径见 README"，**但 README 实际没写**（文档 bug）；
- THS 板块 K 线缓存以**板块名称**为文件名键，板块改名即产生孤儿缓存；
- `load_stock_chunks` 合并不做 (code,date) 去重，依赖抓取脚本自律；
- `flag_limit_up` 用 `close/(1+pct_chg/100)` 反推 prev_close，baostock pctChg 仅两位小数，阈值边缘会误判——baostock 本身提供 `preclose` 字段，应直接取；
- `flag_limit_up` 的 `df.apply(axis=1)` 对约 400 万行逐行跑 Python，可 `np.select` 向量化（当前管线 ~79s 可接受，列为优化项）；
- pipeline ths 分支对单板块 cons/kline 失败 `except: continue`，无失败计数告警——板块静默缺失时榜单照常出；
- `promotion_rate` 未过滤当日停牌股（`money_effect` 过滤了），口径不一致；
- 数据新鲜度：个股 chunk 止于 2026-07-30（缺周五 7-31），无定时调度，每日更新靠手动（README 已列为后续扩展，建议配 automation）。

**Web/脚本**
- `server.py` 的 `@app.on_event("startup")` 已弃用（日志有 DeprecationWarning），应改 lifespan；
- `build_etf_map.py` 的 `ALLOW_QDII` 判断是 `if ...: pass` 空操作死代码；抽查文案硬编码"100 板块池"（THS 口径下误导）；
- 东财/THS 两套口径的脚本、缓存、产物混在同一目录无命名隔离；
- README 多处过期：仍写"100 个板块中 87 个有对应 ETF"（现为 THS 90 个）；快速开始只提 `fetch_board_data.py`（东财）未提 `fetch_ths_boards.py`（当前主力）；目录清单缺 `fetch_ths_boards/refresh_*/regime_check` 三个脚本。

**测试**
- 覆盖因子/合成/信号/回测核心口径，质量不错；缺 pipeline、transport、server 的任何测试（可接受，但建议至少给 `combine_factors` 的 min_coverage 边界、`parse_switching` 补用例）；
- 建议加一个"测试环境构造的 Series 索引命名与生产一致"的约定，或直接在代码里不依赖索引名。

---

## 五、改进路线图（建议按序执行）

1. **先修红测试**：`amt.div(total, level="date")`，并把 requirements 钉上上限（pandas<2.4 或实测范围）；
2. **修 THS K 线 pct_chg 腐蚀**：合并后统一重算；顺手写个一次性脚本修复 90 个现存缓存；
3. **给所有产物打口径标签**：报告头/看板顶部标注 board_source、板块数、数据截止日；README 头部结论注明"东财 100 口径"，并补充 THS 口径的实测结论（当前的 +6.2%）；
4. **接入 git**，此后每次改因子口径都留痕；
5. **修 ST 时变性**：抓取加 `isST` 字段 + 阈值逻辑修正（30/68 ST 仍 20%）；`preclose` 直接取用；
6. **透传 refresh 参数**（ths 分支）+ pipeline 失败计数告警；
7. **统一成本口径**（注释/实现/测试三者对齐）；
8. 工程收尾：向量化 flag_limit_up、lifespan 替换 on_event、清理死代码、README 同步、每日更新自动化。

---

## 六、结论

代码功底和工程意识在个人项目里属于上游（传输层、T+1 回测、离线降级、测试锁不变式都是加分项）。**真正需要警惕的不是代码质量，而是结论的稳健性**：同一个信号体系在两个板块口径下从 +101.7% 掉到 +6.2%，说明 README 头部的惊艳数字不具备跨口径稳健性；当前 THS 配置下信号强度低于项目自定阈值。建议先完成"口径标签化 + 红测试修复 + K线腐蚀修复"三件事，再谈因子层面的迭代。
