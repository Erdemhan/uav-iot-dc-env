import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
from scripts.dashboard_server import start_dashboard, stop_dashboard, get_active_run_dir

def main():
    run_dir = get_active_run_dir()
    if not run_dir or not os.path.exists(run_dir):
        # Find latest run dir in artifacts
        import glob
        runs = sorted(glob.glob(os.path.join("artifacts", "202*-*")), reverse=True)
        if runs:
            run_dir = runs[0]
        else:
            print("[DASHBOARD] Error: No run directory found in artifacts.")
            return

    timestamp = os.path.basename(run_dir)
    print(f"[DASHBOARD] Launching dashboard pointing to: {run_dir}")
    
    start_dashboard(run_dir, timestamp)
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("[DASHBOARD] Stopping dashboard...")
        stop_dashboard()

if __name__ == "__main__":
    main()
