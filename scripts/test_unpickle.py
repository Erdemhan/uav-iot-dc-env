import pickle
import os
import sys

# Resolve project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

path = os.path.join(PROJECT_ROOT, "artifacts", "tune", "tune_ppo_lstm_phase1_2026-07-17_10-29-00", "tune_results", "optuna_study", "searcher-state-2026-07-17_10-29-00.pkl")

if not os.path.exists(path):
    # Try alternate path
    path = os.path.join(PROJECT_ROOT, "artifacts", "tune_ppo_lstm_phase1_2026-07-17_10-29-00", "tune_results", "optuna_study", "searcher-state-2026-07-17_10-29-00.pkl")

if not os.path.exists(path):
    print(f"[ERROR] Pickle file not found at: {path}")
    sys.exit(1)

print(f"Attempting to load pickle file: {path}...")
try:
    with open(path, "rb") as f:
        s = pickle.load(f)
    print(f"[SUCCESS] Searcher loaded successfully!")
    print(f"Class: {type(s)}")
    if hasattr(s, "study"):
        print(f"Optuna Study name: {s.study.study_name}")
        print(f"Number of trials in study: {len(s.study.trials)}")
except Exception as e:
    print(f"[ERROR] Failed during unpickling: {e}")
