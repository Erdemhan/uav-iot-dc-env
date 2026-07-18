import json
import os
import glob
import sys

# Resolve project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

print("=== INSPECTING JSON TYPE AND STRUCTURE ===")
try:
    # Local path
    run_dir = os.path.join(PROJECT_ROOT, "artifacts", "tune", "tune_ppo_lstm_phase1_2026-07-17_10-29-00")
    opt_study_dir = os.path.join(run_dir, "tune_results", "optuna_study")
    state_files = glob.glob(os.path.join(opt_study_dir, "experiment_state-*.json"))
    
    if not state_files:
        print("No experiment state files found locally.")
        sys.exit(1)
        
    state_path = state_files[-1]
    print(f"Reading: {state_path}")
    
    with open(state_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    print(f"Type of root element: {type(data)}")
    if isinstance(data, list):
        print(f"List length: {len(data)}")
        if len(data) > 0:
            print(f"Type of first element: {type(data[0])}")
            if isinstance(data[0], dict):
                print(f"First element keys: {list(data[0].keys())}")
            else:
                print(f"First element preview: {str(data[0])[:200]}")
    elif isinstance(data, dict):
        print(f"Dictionary keys: {list(data.keys())}")
        for k in list(data.keys()):
            val = data[k]
            print(f"  Key '{k}': type={type(val)}")
            if isinstance(val, str):
                print(f"    String preview: {val[:100]}...")
            elif isinstance(val, list):
                print(f"    List length: {len(val)}")
                if len(val) > 0:
                    print(f"    First list element type: {type(val[0])}")
                    
except Exception as e:
    print(f"[ERROR] failed: {e}")
