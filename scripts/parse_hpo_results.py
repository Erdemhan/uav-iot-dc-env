import os
import json
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Parse HPO trial results and extract best parameters.")
    parser.add_argument("--run-dir", type=str, required=True, 
                        help="Path to the HPO run directory containing 'tune_results' (e.g. artifacts/tune/tune_dqn_...)")
    parser.add_argument("--algo", type=str, default="dqn", choices=["ppo", "dqn", "ppo_lstm", "qjc"],
                        help="Algorithm key to update in tuned_configs.json (default: dqn)")
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    tune_results_dir = os.path.join(run_dir, "tune_results", "optuna_study")
    
    if not os.path.exists(tune_results_dir):
        # Fallback to scanning trial folders directly under run_dir
        tune_results_dir = run_dir
        
    print(f"Scanning directory: {tune_results_dir}")
    
    best_objective = -float('inf')
    best_config = None
    best_trial_num = None
    completed_trials_count = 0
    
    # Walk through the directories to find trial_* folders
    for item in os.listdir(tune_results_dir):
        item_path = os.path.join(tune_results_dir, item)
        if os.path.isdir(item_path) and (item.startswith("trial_") or item.startswith("t_") or item.startswith("train_rllib_trial")):
            result_file = os.path.join(item_path, "result.json")
            if os.path.exists(result_file):
                try:
                    # Read the last line of result.json which contains the latest metrics
                    last_result = None
                    with open(result_file, "r", encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                last_result = json.loads(line)
                    
                    if last_result is not None:
                        completed_trials_count += 1
                        objective = last_result.get("objective")
                        config = last_result.get("config")
                        
                        if objective is not None and config is not None:
                            # We want to maximize the objective
                            if objective > best_objective:
                                best_objective = objective
                                best_config = config
                                best_trial_num = item
                except Exception as e:
                    print(f"Warning: Could not parse {result_file}: {e}")
                    
    print(f"\nScan completed. Found {completed_trials_count} completed trials.")
    
    if best_config is None:
        print("Error: Could not find any valid trial results with an objective and config.")
        return
        
    print(f"Best Trial: {best_trial_num}")
    print(f"Best Objective: {best_objective:.4f}")
    
    # Filter config to only include the tuned hyper-parameters (remove static ones)
    tuned_keys = ["lr", "gamma", "architecture", "target_network_update_freq", "lstm_cell_size", "max_seq_len", "tau_0", "temp_xi", "mu_offset"]
    cleaned_config = {k: v for k, v in best_config.items() if k in tuned_keys}
    print(f"Best Hyperparameters: {cleaned_config}")
    
    # Save/Update in confs/tuned_configs.json
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tuned_cfg_path = os.path.join(project_root, "confs", "tuned_configs.json")
    
    tuned_configs = {}
    if os.path.exists(tuned_cfg_path):
        try:
            with open(tuned_cfg_path, "r", encoding="utf-8") as f:
                tuned_configs = json.load(f)
        except Exception as e:
            print(f"Warning: Could not read existing tuned_configs.json: {e}")
            
    # Update the specific algorithm key
    algo_key = args.algo.lower().replace("-", "_")
    tuned_configs[algo_key] = cleaned_config
    
    with open(tuned_cfg_path, "w", encoding="utf-8") as f:
        json.dump(tuned_configs, f, indent=4)
        
    print(f"\n[SUCCESS] Successfully saved best parameters for '{algo_key}' to: {tuned_cfg_path}")

if __name__ == "__main__":
    main()
