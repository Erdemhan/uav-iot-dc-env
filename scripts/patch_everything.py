import pickle
import os
import glob
import sys
import shutil
import optuna

# Standalone dummy function definition in __main__ namespace to satisfy pickle load/dump
def short_trial_dirname_creator(trial):
    pass

# Resolve project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

def patch_searcher_pkl(path):
    print(f"\n[PKL] Processing searcher state file: {path}...")
    try:
        with open(path, "rb") as f:
            state_dict = pickle.load(f)
            
        # 1. Patch Optuna study
        study = state_dict.get("_ot_study")
        if study is not None:
            running_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.RUNNING]
            print(f"  [PKL] Found {running_trials} running trials in Optuna study.")
            for t in running_trials:
                print(f"    Patching Trial #{t.number} (ID: {t._trial_id}) state -> FAIL")
                try:
                    study._storage.set_trial_state_values(t._trial_id, optuna.trial.TrialState.FAIL)
                except AttributeError:
                    study._storage.set_trial_state(t._trial_id, optuna.trial.TrialState.FAIL)
                    
        # 2. Patch completed trials set
        completed_set = state_dict.get("_completed_trials", set())
        ot_trials_dict = state_dict.get("_ot_trials", {})
        active_ids = set(ot_trials_dict.keys()) - completed_set
        if active_ids:
            print(f"  [PKL] Adding active trial IDs to _completed_trials: {list(active_ids)}")
            completed_set.update(active_ids)
            state_dict["_completed_trials"] = completed_set
            
        # Save back
        with open(path, "wb") as f:
            pickle.dump(state_dict, f)
        print(f"  [PKL SUCCESS] Patched: {path}")
    except Exception as e:
        print(f"  [PKL ERROR] Failed to patch {path}: {e}")

def patch_search_gen_json(path):
    print(f"\n[JSON] Processing search gen state file: {path}...")
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
            
        # 1. Patch limiter state
        limiter_state = data.get("name:ConcurrencyLimiter")
        if limiter_state is not None:
            print(f"  [JSON] Current live_trials: {limiter_state.get('live_trials')} | unfinished: {limiter_state.get('num_unfinished_live_trials')}")
            limiter_state["live_trials"] = "set()"
            limiter_state["num_unfinished_live_trials"] = 0
            
        # 2. Patch Experiment object to bypass PicklingErrors
        exp = data.get("experiment")
        if exp is not None and hasattr(exp, "spec") and isinstance(exp.spec, dict):
            print("  [JSON] Nullifying 'trial_dirname_creator' in Experiment spec...")
            exp.spec["trial_dirname_creator"] = None
            
        # Safe save (atomic write)
        temp_file = path + ".tmp"
        with open(temp_file, "wb") as f:
            pickle.dump(data, f)
        shutil.move(temp_file, path)
        print(f"  [JSON SUCCESS] Patched: {path}")
    except Exception as e:
        print(f"  [JSON ERROR] Failed to patch {path}: {e}")

def main():
    print("=== SWEEPING AND PATCHING ALL TUNE STATE LOCATIONS ===")
    
    # 1. Sweep local workspace paths
    tune_dir = os.path.join(PROJECT_ROOT, "artifacts", "tune")
    ppo_lstm_runs = sorted(glob.glob(os.path.join(tune_dir, "tune_ppo_lstm_phase1_2026-07-*")), reverse=True)
    
    local_pkls = []
    local_jsons = []
    
    if ppo_lstm_runs:
        run_dir = ppo_lstm_runs[0]
        opt_study_dir = os.path.join(run_dir, "tune_results", "optuna_study")
        local_pkls = glob.glob(os.path.join(opt_study_dir, "searcher-state-*.pkl"))
        local_jsons = glob.glob(os.path.join(opt_study_dir, "search_gen_state-*.json"))
        
    # 2. Sweep /tmp ray session paths
    tmp_pkls = glob.glob("/tmp/ray/session_*/artifacts/*/optuna_study/driver_artifacts/searcher-state-*.pkl")
    tmp_jsons = glob.glob("/tmp/ray/session_*/artifacts/*/optuna_study/driver_artifacts/search_gen_state-*.json")
    
    all_pkls = list(set(local_pkls + tmp_pkls))
    all_jsons = list(set(local_jsons + tmp_jsons))
    
    print(f"Found {len(all_pkls)} searcher state PKL files to patch.")
    print(f"Found {len(all_jsons)} search generator JSON files to patch.")
    
    for pkl in all_pkls:
        patch_searcher_pkl(pkl)
        
    for js in all_jsons:
        patch_search_gen_json(js)
        
    print("\n[COMPLETE] All locations swept and patched successfully!")

if __name__ == "__main__":
    main()
