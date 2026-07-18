import os
import json
import argparse
import sys
import glob
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Resolve project root dynamically (works on both Windows and Linux/WSL)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from confs.opt_config import OptConfig

def main():
    parser = argparse.ArgumentParser(description="Reconstruct Optuna study in correct order and save visualization plots.")
    parser.add_argument("--run-dir", type=str, required=True, 
                        help="Path to the HPO run directory containing 'tune_results' (e.g. artifacts/tune_ppo_...)")
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "dqn", "ppo_lstm", "qjc"],
                        help="Algorithm name (default: ppo)")
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    tune_results_dir = os.path.join(run_dir, "tune_results", "optuna_study")
    if not os.path.exists(tune_results_dir):
        tune_results_dir = run_dir
        
    optuna_dir = os.path.join(run_dir, "optuna")
    os.makedirs(optuna_dir, exist_ok=True)
    
    print(f"Reading trial logs from: {tune_results_dir}")
    
    # 1. Build chronological mapping from experiment_state-*.json
    trial_id_to_index = {}
    state_files = glob.glob(os.path.join(tune_results_dir, "experiment_state-*.json"))
    if state_files:
        state_file = state_files[0]
        print(f"Loading chronological trial order from state metadata: {os.path.basename(state_file)}")
        try:
            with open(state_file, "r", encoding="utf-8") as f:
                state_data = json.load(f)
            trial_list = state_data.get("trial_data", [])
            for idx, item in enumerate(trial_list):
                if isinstance(item, list) and len(item) > 0:
                    elem = item[0]
                    if isinstance(elem, str):
                        try:
                            elem = json.loads(elem)
                        except Exception:
                            pass
                    if isinstance(elem, dict) and "trial_id" in elem:
                        trial_id_to_index[elem["trial_id"]] = idx
            print(f"Successfully mapped {len(trial_id_to_index)} trial IDs to chronological order.")
        except Exception as e:
            print(f"Warning: Could not parse experiment state file: {e}")
    else:
        print("Warning: No experiment_state-*.json file found. Order will be fallback sequential.")

    import optuna
    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
    
    # Create study
    study = optuna.create_study(direction="maximize")
    
    # Define distributions based on algorithm
    distributions = {
        "lr": optuna.distributions.FloatDistribution(OptConfig.RL_LR_MIN, OptConfig.RL_LR_MAX, log=True),
        "gamma": optuna.distributions.FloatDistribution(OptConfig.RL_GAMMA_MIN, OptConfig.RL_GAMMA_MAX)
    }
    
    arch_choices = OptConfig.ARCH_CHOICES
    scanned_trials = []
    
    # Scan trial folders and extract results
    for item in os.listdir(tune_results_dir):
        item_path = os.path.join(tune_results_dir, item)
        if os.path.isdir(item_path) and (item.startswith("trial_") or item.startswith("t_") or item.startswith("train_rllib_trial")):
            trial_hash = item.split("_")[-1]
            result_file = os.path.join(item_path, "result.json")
            if os.path.exists(result_file):
                try:
                    last_result = None
                    with open(result_file, "r", encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                last_result = json.loads(line)
                                
                    if last_result is not None:
                        objective = last_result.get("objective")
                        config = last_result.get("config")
                        iteration = last_result.get("training_iteration", 0)
                        
                        # Only consider trials that finished at least 100 iterations (or 1000)
                        if objective is not None and config is not None and iteration >= 100:
                            original_index = trial_id_to_index.get(trial_hash, len(scanned_trials))
                            
                            params = {}
                            trial_dists = {}
                            
                            # 1. Base params (lr, gamma)
                            if "lr" in config:
                                params["lr"] = config["lr"]
                                trial_dists["lr"] = distributions["lr"]
                            if "gamma" in config:
                                params["gamma"] = config["gamma"]
                                trial_dists["gamma"] = distributions["gamma"]
                                
                            # 2. Architecture
                            if "architecture" in config:
                                arch_val = config["architecture"]
                                if isinstance(arch_val, list):
                                    arch_val = ",".join(map(str, arch_val))
                                params["architecture"] = arch_val
                                choices = [str(x) if isinstance(x, list) else x for x in arch_choices]
                                trial_dists["architecture"] = optuna.distributions.CategoricalDistribution(choices)
                                
                            # 3. DQN target update freq
                            if args.algo == "dqn" and "target_network_update_freq" in config:
                                params["target_network_update_freq"] = config["target_network_update_freq"]
                                trial_dists["target_network_update_freq"] = optuna.distributions.CategoricalDistribution(OptConfig.DQN_TARGET_UPDATE_FREQ)
                                
                            # 4. PPO-LSTM specific params
                            if args.algo == "ppo_lstm":
                                if "lstm_cell_size" in config:
                                    params["lstm_cell_size"] = config["lstm_cell_size"]
                                    trial_dists["lstm_cell_size"] = optuna.distributions.CategoricalDistribution(OptConfig.PPOLSTM_CELL_SIZE)
                                if "max_seq_len" in config:
                                    params["max_seq_len"] = config["max_seq_len"]
                                    trial_dists["max_seq_len"] = optuna.distributions.CategoricalDistribution(OptConfig.PPOLSTM_MAX_SEQ_LEN)
                                    
                            frozen = optuna.trial.create_trial(
                                params=params,
                                distributions=trial_dists,
                                value=objective
                            )
                            scanned_trials.append((original_index, frozen))
                except Exception as e:
                    print(f"Warning: Could not load {result_file}: {e}")
                    
    # Sort trials by original Ray chronological index to preserve historical order
    scanned_trials.sort(key=lambda x: x[0])
    for _, frozen in scanned_trials:
        study.add_trial(frozen)
        
    print(f"Reconstructed study in correct chronological order with {len(study.trials)} trials.")
    
    if len(study.trials) == 0:
        print("Error: No trials found to reconstruct.")
        return
        
    # Draw and Save Matplotlib Plots
    print(f"Generating Optuna visualization plots in: {optuna_dir}")
    
    from scripts.tune_models import save_optuna_visualizations
    save_optuna_visualizations(study, optuna_dir)
    
    print("[SUCCESS] Plots recreated successfully!")

if __name__ == "__main__":
    main()
