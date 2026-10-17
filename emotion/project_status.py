"""项目用途声明，与冻结的情绪计算/历史回测参数分离。"""

OBSERVATION_NOTICE = (
    "项目已于2026-10-13结束当前情绪交易路线，保留为市场观察工具。"
    "情绪分、冷热榜与风格读数仅描述历史和当前状态，不生成买卖建议。"
)


def project_status():
    return {
        "mode": "observation_only",
        "effective_date": "2026-10-13",
        "trading_eligible": False,
        "notice": OBSERVATION_NOTICE,
        "reason": "最后一轮去重与情绪增量实验六项退出条件仅通过一项",
        "assessment": "report/final_portfolio_20261009/report.md",
        "decision": "report/final_portfolio_20261009/decision.json",
        "plan": "config/final_portfolio_experiment.json",
    }
