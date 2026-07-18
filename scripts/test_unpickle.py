import pickle
import os
import sys

# Resolve project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

path = os.path.join(PROJECT_ROOT, "artifacts", "tune", "tune_ppo_lstm_phase1_2026-07-17_10-29-00", "tune_results", "optuna_study", "searcher-state-2026-07-17_10-29-00.pkl")

if not os.path.exists(path):
    path = os.path.join(PROJECT_ROOT, "artifacts", "tune_ppo_lstm_phase1_2026-07-17_10-29-00", "tune_results", "optuna_study", "searcher-state-2026-07-17_10-29-00.pkl")

with open(path, "rb") as f:
    s = pickle.load(f)

print(f"[SUCCESS] Searcher loaded successfully!")
print(f"Class: {type(s)}")
if isinstance(s, dict):
    print("Keys of the state dictionary:", list(s.keys()))
    for k, v in s.items():
        print(f"  {k}: type={type(v)}")
