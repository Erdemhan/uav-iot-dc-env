import os
import sys
import argparse
import json
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

os.environ["RAY_DISABLE_METRICS_EXPORT"] = "1"
os.environ["RAY_OVERRIDE_JOB_RUNTIME_ENV"] = "1"

import ray
from confs.model_config import GlobalConfig, PPOConfig, DQNConfig, PPOLSTMConfig
from confs.env_config import EnvConfig

# Direct import of HPO's exact trainable function!
from scripts.tune_models import train_rllib_trial

@ray.remote(num_gpus=1)
def run_hpo_trainable_remote(config):
    """Remote actor wrapper executing HPO's exact train_rllib_trial function on Worker GPU."""
    return train_rllib_trial(config)

def main():
    parser = argparse.ArgumentParser(description="Unified Test Trainer directly executing HPO's exact train_rllib_trial code.")
    parser.add_argument("--scenario", type=str, default="1-A", choices=["1-A", "1-B", "2-A", "2-B"], help="Scenario ID")
    parser.add_argument("--algo", type=str, default="DQN", choices=["PPO", "DQN", "PPO-LSTM", "dqn", "ppo", "ppo_lstm"], help="Algorithm name")
    parser.add_argument("--iterations", type=int, default=1000, help="Training iterations")
    parser.add_argument("--num-workers", type=int, default=10, help="Rollout workers")
    args = parser.parse_args()

    algo_upper = args.algo.upper()
    if algo_upper == "PPO_LSTM":
        algo_upper = "PPO-LSTM"

    print(f"\n==================================================")
    print(f" DIRECT HPO PIPELINE EXECUTOR: {algo_upper} | Scenario {args.scenario}")
    print(f"==================================================\n")

    # Set Scenario EnvConfig
    if args.scenario == "1-A":
        EnvConfig.NUM_NODES = 15; EnvConfig.NUM_UAVS = 1; EnvConfig.AREA_SIZE = 500.0; EnvConfig.W_COST = 0.03
    elif args.scenario == "1-B":
        EnvConfig.NUM_NODES = 15; EnvConfig.NUM_UAVS = 1; EnvConfig.AREA_SIZE = 500.0; EnvConfig.W_COST = 0.3
    elif args.scenario == "2-A":
        EnvConfig.NUM_NODES = 30; EnvConfig.NUM_UAVS = 2; EnvConfig.AREA_SIZE = 1000.0; EnvConfig.W_COST = 0.03
    elif args.scenario == "2-B":
        EnvConfig.NUM_NODES = 30; EnvConfig.NUM_UAVS = 2; EnvConfig.AREA_SIZE = 1000.0; EnvConfig.W_COST = 0.3

    env_cfg_dict = {
        "W_SUCCESS": EnvConfig.W_SUCCESS,
        "W_TRACKING": EnvConfig.W_TRACKING,
        "W_COST": EnvConfig.W_COST,
        "num_nodes": int(EnvConfig.NUM_NODES),
        "num_uavs": int(EnvConfig.NUM_UAVS),
        "area_size": float(EnvConfig.AREA_SIZE),
        "w_cost": float(EnvConfig.W_COST)
    }

    # Construct exact HPO trial config dict matching tune_models.py
    if algo_upper == "DQN":
        trial_config = {
            "algo": "DQN",
            "lr": 0.0003204387357042751,
            "gamma": 0.8721224348952492,
            "architecture": "128,512,512",
            "target_network_update_freq": 2000,
            "num_workers": args.num_workers,
            "num_gpus": 1,
            "iterations": args.iterations,
            "phase": 1,
            "env_config": env_cfg_dict
        }
    elif algo_upper == "PPO":
        trial_config = {
            "algo": "PPO",
            "lr": PPOConfig.LR,
            "gamma": PPOConfig.GAMMA,
            "architecture": "512,256",
            "num_workers": args.num_workers,
            "num_gpus": 1,
            "iterations": args.iterations,
            "phase": 1,
            "env_config": env_cfg_dict
        }
    elif algo_upper == "PPO-LSTM":
        trial_config = {
            "algo": "PPO-LSTM",
            "lr": PPOLSTMConfig.LR,
            "gamma": PPOLSTMConfig.GAMMA,
            "architecture": "256,512,512",
            "lstm_cell_size": PPOLSTMConfig.LSTM_CELL_SIZE,
            "max_seq_len": PPOLSTMConfig.MAX_SEQ_LEN,
            "num_workers": args.num_workers,
            "num_gpus": 1,
            "iterations": args.iterations,
            "phase": 1,
            "env_config": env_cfg_dict
        }

    # Initialize Ray with runtime_env for cluster workers
    if not ray.is_initialized():
        runtime_env = {
            "working_dir": PROJECT_ROOT,
            "excludes": [".venv", "artifacts", "ray_results", "head-node-logs", ".git", "baseline_q_table", "node_modules"],
            "env_vars": {"PYTHONPATH": "."}
        }
        ray.init(ignore_reinit_error=True, runtime_env=runtime_env)

    # Call HPO's EXACT train_rllib_trial function on Worker GPU
    print("Dispatching exact HPO train_rllib_trial function to Worker GPU...")
    start_t = time.time()
    future = run_hpo_trainable_remote.remote(trial_config)
    res = ray.get(future)
    elapsed = round(time.time() - start_t, 2)

    print(f"\n==================================================")
    print(f" DIRECT HPO EXECUTION COMPLETED ({elapsed}s)")
    print(f" Algorithm: {algo_upper} | Scenario: {args.scenario}")
    print(f" Result: {json.dumps(res, indent=2)}")
    print(f"==================================================\n")

if __name__ == "__main__":
    main()
