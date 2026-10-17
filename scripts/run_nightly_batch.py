"""夜间/盘后全量数据更新批次（5步串行）
步骤 1: scripts/fetch_etf_boards.py
步骤 2: scripts/fetch_stock_history.py --update
步骤 3: main.py
步骤 4: scripts/daily_anomaly.py
步骤 5: scripts/dump_dashboard_data.py
"""

import os
import sys
import time
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)

STEPS = [
    ("Step 1: 板块日线增量(ETF)", [sys.executable, "-X", "utf8", str(ROOT / "scripts" / "fetch_etf_boards.py")], "auto_20260924_night_step1.log"),
    ("Step 2: 个股日线增量(快照/逐股)", [sys.executable, "-X", "utf8", str(ROOT / "scripts" / "fetch_stock_history.py"), "--update"], "auto_20260924_night_step2.log"),
    ("Step 3: 情绪日报生成", [sys.executable, "-X", "utf8", str(ROOT / "main.py")], "auto_20260924_night_step3.log"),
    ("Step 4: 异动榜计算", [sys.executable, "-X", "utf8", str(ROOT / "scripts" / "daily_anomaly.py")], "auto_20260924_night_step4.log"),
    ("Step 5: 看板数据导出", [sys.executable, "-X", "utf8", str(ROOT / "scripts" / "dump_dashboard_data.py")], "auto_20260924_night_step5.log"),
]

def log(msg: str):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{timestamp}] {msg}"
    print(formatted, flush=True)
    with open(LOGS_DIR / "auto_20260924_night.log", "a", encoding="utf-8") as f:
        f.write(formatted + "\n")

def main():
    log("=== 开始执行夜间全流程数据更新批次 ===")
    total_t0 = time.time()
    
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    
    for idx, (label, cmd, logfile_name) in enumerate(STEPS, start=1):
        log(f"[{idx}/{len(STEPS)}] 开始执行: {label}")
        log_path = LOGS_DIR / logfile_name
        t0 = time.time()
        
        with open(log_path, "w", encoding="utf-8") as log_file:
            log_file.write(f"=== {label} 开始于 {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
            log_file.flush()
            
            proc = subprocess.Popen(
                cmd,
                cwd=str(ROOT),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace"
            )
            
            for line in proc.stdout:
                log_file.write(line)
                log_file.flush()
            
            proc.wait()
            elapsed = time.time() - t0
            
            if proc.returncode == 0:
                log(f"[{idx}/{len(STEPS)}] 成功完成: {label} (耗时: {elapsed:.1f} 秒)")
            else:
                log(f"[{idx}/{len(STEPS)}] 失败: {label} (退出码: {proc.returncode}, 耗时: {elapsed:.1f} 秒)")
                log("更新批次中止，请检查对应日志: " + str(log_path))
                sys.exit(proc.returncode)
                
    total_elapsed = time.time() - total_t0
    log(f"=== 全部 5 步更新成功完成！总耗时: {total_elapsed:.1f} 秒 ===")

if __name__ == "__main__":
    main()
