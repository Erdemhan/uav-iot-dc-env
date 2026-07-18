import pickle
import os
import glob
import sys
import json

# Resolve project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

def inspect_pkl(path):
    print(f"\n[PKL] Inspecting: {path}")
    try:
        with open(path, "rb") as f:
            state_dict = pickle.load(f)
        completed_set = state_dict.get("_completed_trials", set())
        ot_trials = state_dict.get("_ot_trials", {})
        print(f"  _completed_trials (len={len(completed_set)}): {list(completed_set)[:5]}...")
        print(f"  _ot_trials (len={len(ot_trials)}): {list(ot_trials.keys())[:5]}...")
        active = set(ot_trials.keys()) - completed_set
        print(f"  Active trials registered in searcher: {list(active)}")
    except Exception as e:
        print(f"  [ERROR] {e}")

def inspect_json(path):
    print(f"\n[JSON] Inspecting: {path}")
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        limiter = data.get("name:ConcurrencyLimiter", {})
        print(f"  live_trials: {limiter.get('live_trials')} (type={type(limiter.get('live_trials'))})")
        print(f"  num_unfinished_live_trials: {limiter.get('num_unfinished_live_trials')} (type={type(limiter.get('num_unfinished_live_trials'))})")
    except Exception as e:
        print(f"  [ERROR] {e}")

def main():
    print("=== INSPECTING CURRENT STATES ===")
    
    # Local paths
    tune_dir = os.path.join(PROJECT_ROOT, "artifacts", "tune")
    ppo_lstm_runs = sorted(glob.glob(os.path.join(tune_dir, "tune_ppo_lstm_phase1_2026-07-*")), reverse=True)
    if ppo_lstm_runs:
        run_dir = ppo_lstm_runs[0]
        opt_study_dir = os.path.join(run_dir, "tune_results", "optuna_study")
        pkls = glob.glob(os.path.join(opt_study_dir, "searcher-state-*.pkl"))
        jsons = glob.glob(os.path.join(opt_study_dir, "search_gen_state-*.json"))
        
        for pkl in pkls: inspect_pkl(pkl)
        for js in jsons: inspect_json(js)
        
    # /tmp paths
    tmp_pkls = glob.glob("/tmp/ray/session_2026-07-17_10-05-50_996511_3860/artifacts/2026-07-17_10-29-00/optuna_study/driver_artifacts/searcher-state-*.pkl")
    tmp_jsons = glob.glob("/tmp/ray/session_2026-07-17_10-05-50_996511_3860/artifacts/2026-07-17_10-29-00/optuna_study/driver_artifacts/search_gen_state-*.json")
    
    for pkl in tmp_pkls: inspect_pkl(pkl)
    for js in tmp_jsons: inspect_json(js)

if __name__ == "__main__":
    main()
