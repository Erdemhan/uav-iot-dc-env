import os
import glob
import sys

# Resolve project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

tune_dir = os.path.join(PROJECT_ROOT, "artifacts", "tune")
ppo_lstm_runs = sorted(glob.glob(os.path.join(tune_dir, "tune_ppo_lstm_phase1_2026-07-*")), reverse=True)
if not ppo_lstm_runs:
    print("[ERROR] No PPO-LSTM run folder found.")
    sys.exit(1)

run_dir = ppo_lstm_runs[0]
print(f"Listing all files recursively under: {run_dir}\n")

for root, dirs, files in os.walk(run_dir):
    for f in files:
        rel_path = os.path.relpath(os.path.join(root, f), run_dir)
        size = os.path.getsize(os.path.join(root, f))
        print(f"  {rel_path} ({size} bytes)")
