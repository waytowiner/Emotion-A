# 临时与历史诊断脚本归档目录 (Diagnostics Archive)

本目录用于存放特定时点（如 `_0911`）的一过性诊断、市场结构验算与专项排查脚本。

## 归档脚本说明

- **`diag_guard_0911.py`**：2026-09-11 验证市场等权指数近 10 日累计收益与回撤护栏触发阈值。
- **`diag_ic_realized_0911.py`**：2026-09-11 验证历史情绪分 CSV 与后续 3/5 日板块实际收益实现情况。
- **`diag_market_regime_0911.py`**：2026-09-11 验证板块 K 线的等权净值走势、回撤、新高新低宽度分布。

> 注：常态化生产与诊断工具（如 `diag_ic_decay.py`、`diag_regime_flip.py`、`validate_robustness.py`）保留在 `scripts/` 主目录中。
