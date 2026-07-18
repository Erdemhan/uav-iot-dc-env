import os
import glob
import sys

# Resolve project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

print("Sweeping Ray session logs in /tmp...")
try:
    log_pattern = "/tmp/ray/session_*/logs/"
    log_dirs = sorted(glob.glob(log_pattern))
    if not log_dirs:
        print("[ERROR] No Ray log directories found in /tmp.")
        sys.exit(1)
        
    latest_log_dir = log_dirs[-1]
    print(f"Reading from latest session log directory: {latest_log_dir}\n")
    
    # Check important log files
    log_files = ["gcs_server.out", "raylet.out", "dashboard.log", "python-core-driver-*.log"]
    
    for lf_pattern in log_files:
        matching_files = sorted(glob.glob(os.path.join(latest_log_dir, lf_pattern)))
        if not matching_files:
            continue
            
        lf = matching_files[-1]
        print(f"=== LAST 20 LINES OF {os.path.basename(lf)} ===")
        try:
            with open(lf, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            for line in lines[-20:]:
                print("  " + line.strip())
        except Exception as fe:
            print(f"  [ERROR] Could not read file: {fe}")
        print()
        
except Exception as e:
    print(f"[ERROR] Failed to read Ray logs: {e}")
