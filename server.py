"""板块情绪雷达 Web 看板后端：python server.py

启动后浏览器打开 http://127.0.0.1:8000
数据来自 scripts/dump_dashboard_data.py 生成的 data/dashboard/。
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent
DASH_DIR = ROOT / "data" / "dashboard"
DUMP_SCRIPT = ROOT / "scripts" / "dump_dashboard_data.py"

# ---------------- 数据加载（内存缓存，refresh 后重载） ----------------

_cache: dict = {}


def _read_json(name: str):
    with open(DASH_DIR / name, encoding="utf-8") as f:
        return json.load(f)


def load_all() -> None:
    """把 data/dashboard/ 全部读入内存。"""
    global _cache
    if not DASH_DIR.exists() or not (DASH_DIR / "meta.json").exists():
        _cache = {}
        return
    _cache = {
        "latest": _read_json("latest.json"),
        "meta": _read_json("meta.json"),
        "backtest": _read_json("backtest.json"),
        "factor_snapshot": _read_json("factor_snapshot.json"),
        "health_series": _read_json("health_series.json"),
        "scores": pd.read_parquet(DASH_DIR / "scores.parquet"),
        "pct": pd.read_parquet(DASH_DIR / "pct.parquet"),
        "kline": pd.read_parquet(DASH_DIR / "board_kline.parquet"),
    }


def require_data() -> dict:
    if not _cache:
        raise HTTPException(
            status_code=503,
            detail="看板数据缺失：请先运行 python scripts/dump_dashboard_data.py")
    return _cache


def _matrix_long(df: pd.DataFrame, value_name: str, days: int,
                 board: str | None = None) -> list[dict]:
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.set_index("date").sort_index().tail(days)
    if board:
        if board not in d.columns:
            raise HTTPException(status_code=404, detail=f"未知板块代码：{board}")
        d = d[[board]]
    out = d.reset_index().melt(id_vars="date", var_name="board_code",
                               value_name=value_name).dropna()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    return out.to_dict(orient="records")


# ---------------- API ----------------

@asynccontextmanager
async def lifespan(_app: FastAPI):
    """启动时加载看板数据；缺失时提示先跑导出脚本。"""
    load_all()
    if not _cache:
        print("!" * 60)
        print("! 看板数据缺失：请先运行  python scripts/dump_dashboard_data.py")
        print("!" * 60)
    yield


app = FastAPI(title="板块情绪雷达", docs_url=None, redoc_url=None, lifespan=lifespan)


@app.middleware("http")
async def no_cache(request, call_next):
    """禁止浏览器启发式缓存：本地看板文件常更新，避免 HTML/JS 新旧混载。"""
    resp = await call_next(request)
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    return resp


@app.get("/api/overview")
def api_overview():
    c = require_data()
    return {"latest": c["latest"], "meta": c["meta"]}


@app.get("/api/scores")
def api_scores(days: int = Query(120, ge=1, le=1500)):
    c = require_data()
    return {"days": days, "data": _matrix_long(c["scores"], "score", days)}


@app.get("/api/board/{code}")
def api_board(code: str, days: int = Query(180, ge=1, le=1500)):
    c = require_data()
    code = code.strip()
    score_s = _matrix_long(c["scores"], "score", days, board=code)
    pct_s = _matrix_long(c["pct"], "pct", days, board=code)

    k = c["kline"]
    k = k[k["board_code"] == code]
    if k.empty and not score_s:
        raise HTTPException(status_code=404, detail=f"未知板块代码：{code}")
    k = k.sort_values("date").tail(days)
    kline = [{"date": pd.Timestamp(r.date).strftime("%Y-%m-%d"),
              "close": None if pd.isna(r.close) else float(r.close),
              "pct_chg": None if pd.isna(r.pct_chg) else float(r.pct_chg),
              "amount": None if pd.isna(r.amount) else float(r.amount)}
             for r in k.itertuples()]

    snap = c["factor_snapshot"]["boards"].get(code, {})
    name = next((r["name"] for r in c["latest"]["ranking"] if r["code"] == code), code)
    rank_row = next((r for r in c["latest"]["ranking"] if r["code"] == code), None)
    etf = c["latest"].get("etf_map", {}).get(code)
    leaders_info = c["latest"].get("leaders", {}).get(code,
                                                      {"liquidity": [], "momentum": []})

    # 该板块当前命中的信号
    badges = []
    for key in ["heating", "cooling", "hot", "cold"]:
        if any(x["code"] == code for x in c["latest"]["signals"][key]):
            badges.append(key)
    if any(x["from_code"] == code for x in c["latest"]["signals"]["switching"]):
        badges.append("switching_from")

    return {
        "code": code,
        "name": name,
        "latest": rank_row,
        "badges": badges,
        "etf": etf,
        "leaders": leaders_info,
        "score": score_s,
        "pct": pct_s,
        "kline": kline,
        "factors": snap,
        "factor_date": c["factor_snapshot"]["date"],
    }


@app.get("/api/backtest")
def api_backtest():
    return require_data()["backtest"]


@app.get("/api/health_series")
def api_health_series(days: int = Query(120, ge=1, le=1000)):
    c = require_data()
    hs = c["health_series"]
    return {"window": hs["window"], "horizon": hs["horizon"],
            "series": hs["series"][-days:]}


# ---------------- 数据刷新 ----------------

_refresh_state = {"running": False, "started_at": None, "finished_at": None,
                  "ok": None, "log_tail": ""}


def _run_dump():
    _refresh_state.update(running=True, started_at=time.strftime("%H:%M:%S"),
                          ok=None, log_tail="")
    try:
        proc = subprocess.run([sys.executable, str(DUMP_SCRIPT)],
                              cwd=str(ROOT), capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=900)
        tail = (proc.stdout + proc.stderr).strip().splitlines()[-8:]
        _refresh_state["log_tail"] = "\n".join(tail)
        _refresh_state["ok"] = proc.returncode == 0
        if proc.returncode == 0:
            load_all()
    except Exception as exc:  # noqa: BLE001
        _refresh_state["ok"] = False
        _refresh_state["log_tail"] = f"{type(exc).__name__}: {exc}"
    finally:
        _refresh_state["running"] = False
        _refresh_state["finished_at"] = time.strftime("%H:%M:%S")


@app.post("/api/refresh")
def api_refresh():
    if _refresh_state["running"]:
        return {"started": False, "message": "正在刷新中，请稍候"}
    if _update_state["running"]:
        return {"started": False, "message": "数据更新进行中，请稍候"}
    threading.Thread(target=_run_dump, daemon=True).start()
    return {"started": True, "message": "已开始重新计算，约需 1-2 分钟"}


@app.get("/api/refresh_status")
def api_refresh_status():
    return _refresh_state


# ---------------- 数据更新（联网抓取全流程） ----------------
# 原 run.bat 启动时串行的 5 步。瓶颈已治理：板块步走中证接口增量（缓存命中
# 零请求），个股日线走腾讯快照（单日 42 秒），baostock 仅在多日补抓时启用。
# 已从启动流程移除，改为看板界面「更新数据」按钮触发本接口。

UPDATE_STEPS = [
    # (步骤名, 脚本, 额外参数, 超时秒)
    ("板块成分/指数", "scripts/fetch_etf_boards.py", [], 1800),  # 中证可交易ETF池
    ("个股日线增量", "scripts/fetch_stock_history.py", ["--update"], 3600),
    ("情绪日报", "main.py", [], 900),
    ("异动榜", "scripts/daily_anomaly.py", [], 900),
    ("看板数据导出", "scripts/dump_dashboard_data.py", [], 900),
]

_update_state = {"running": False, "started_at": None, "finished_at": None,
                 "ok": None, "step": None, "steps": [], "log_tail": ""}


def _run_update():
    _update_state.update(running=True, started_at=time.strftime("%H:%M:%S"),
                         finished_at=None, ok=None, step=None, log_tail="",
                         steps=[{"label": label, "status": "pending", "secs": None}
                                for label, _, _, _ in UPDATE_STEPS])
    ok = True
    try:
        for i, (label, script, args, timeout) in enumerate(UPDATE_STEPS):
            _update_state["step"] = label
            _update_state["steps"][i]["status"] = "running"
            t0 = time.time()
            try:
                proc = subprocess.run([sys.executable, str(ROOT / script), *args],
                                      cwd=str(ROOT), capture_output=True, text=True,
                                      encoding="utf-8", errors="replace",
                                      timeout=timeout)
            except Exception as exc:  # noqa: BLE001
                _update_state["steps"][i]["status"] = "fail"
                _update_state["log_tail"] = f"[{label}] {type(exc).__name__}: {exc}"
                ok = False
                break
            _update_state["steps"][i]["secs"] = round(time.time() - t0)
            tail = (proc.stdout + proc.stderr).strip().splitlines()
            if tail:
                _update_state["log_tail"] = f"[{label}] " + "\n".join(tail[-8:])
            if proc.returncode != 0:
                _update_state["steps"][i]["status"] = "fail"
                ok = False
                break
            _update_state["steps"][i]["status"] = "ok"
        if ok:
            load_all()
    finally:
        _update_state.update(running=False, ok=ok,
                             finished_at=time.strftime("%H:%M:%S"))


@app.post("/api/update")
def api_update():
    if _update_state["running"]:
        return {"started": False, "message": "数据更新进行中，请稍候"}
    if _refresh_state["running"]:
        return {"started": False, "message": "正在刷新中，请稍候"}
    threading.Thread(target=_run_update, daemon=True).start()
    return {"started": True,
            "message": "已开始更新数据（抓取→日报→看板）：日常增量通常 1-5 分钟"
                       "（多日补抓或快照不可用时 20-40 分钟），期间正常显示旧数据"}


@app.get("/api/update_status")
def api_update_status():
    return _update_state


# ---------------- 静态页面 ----------------

@app.get("/", include_in_schema=False)
def index():
    return FileResponse(ROOT / "static" / "index.html")


app.mount("/", StaticFiles(directory=ROOT / "static"), name="static")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
