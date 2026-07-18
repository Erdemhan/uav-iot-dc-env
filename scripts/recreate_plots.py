import os
import json
import argparse
import sys
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root to sys.path
PROJECT_ROOT = "c:/Users/Erdemhan/Desktop/OneDrive - erciyes.edu.tr/okul_msi/Projeler/DR TEZ/uav-iot-dc-env"
sys.path.append(PROJECT_ROOT)

from confs.opt_config import OptConfig

def main():
    parser = argparse.ArgumentParser(description="Reconstruct Optuna study and save all HPO visualization plots.")
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
    
    import optuna
    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
    
    # Create study
    study = optuna.create_study(direction="maximize")
    
    # Define distributions based on algorithm
    # PPO and DQN have different parameters, so we dynamically assign the right distributions
    distributions = {
        "lr": optuna.distributions.FloatDistribution(OptConfig.RL_LR_MIN, OptConfig.RL_LR_MAX, log=True),
        "gamma": optuna.distributions.FloatDistribution(OptConfig.RL_GAMMA_MIN, OptConfig.RL_GAMMA_MAX)
    }
    
    # Add architecture choice to distributions (shared by PPO, DQN, PPO-LSTM)
    arch_choices = OptConfig.ARCH_CHOICES
    
    scanned_trials = []
    
    # Scan trial folders and extract results
    for item in os.listdir(tune_results_dir):
        item_path = os.path.join(tune_results_dir, item)
        if os.path.isdir(item_path) and (item.startswith("trial_") or item.startswith("t_") or item.startswith("train_rllib_trial")):
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
                            trial_id = int(item.split("_")[-1]) if item.split("_")[-1].isdigit() else len(scanned_trials)
                            
                            # Filter config to only include the searched parameters
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
                                # Convert lists to strings if needed
                                if isinstance(arch_val, list):
                                    arch_val = ",".join(map(str, arch_val))
                                params["architecture"] = arch_val
                                # Ensure the choices contains strings
                                choices = [str(x) if isinstance(x, list) else x for x in arch_choices]
                                trial_dists["architecture"] = optuna.distributions.CategoricalDistribution(choices)
                                
                            # 3. DQN specific target update freq
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
                                    
                            # Create Optuna FrozenTrial
                            frozen = optuna.trial.create_trial(
                                params=params,
                                distributions=trial_dists,
                                value=objective
                            )
                            scanned_trials.append((trial_id, frozen))
                except Exception as e:
                    print(f"Warning: Could not load {result_file}: {e}")
                    
    # Sort trials by ID to maintain chronological order in Optimization History
    scanned_trials.sort(key=lambda x: x[0])
    for _, frozen in scanned_trials:
        study.add_trial(frozen)
        
    print(f"Reconstructed study with {len(study.trials)} trials.")
    
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
