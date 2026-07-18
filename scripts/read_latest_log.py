import os
import glob
import sys

# Resolve project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

log_dir = os.path.join(PROJECT_ROOT, "artifacts", "logs")
if not os.path.exists(log_dir):
    print(f"[ERROR] Log directory does not exist: {log_dir}")
    sys.exit(1)

log_files = sorted(glob.glob(os.path.join(log_dir, "tune_models_*.log")))
if not log_files:
    print("[ERROR] No log files found.")
    sys.exit(1)

latest_log = log_files[-1]
print(f"Reading last 50 lines of latest log: {latest_log}\n")

with open(latest_log, "r", encoding="utf-8", errors="ignore") as f:
    lines = f.readlines()

for line in lines[-50:]:
    print(line, end="")
