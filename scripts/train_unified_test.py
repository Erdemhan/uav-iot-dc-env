import os
import sys
import argparse
import json
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

os.environ["RAY_DISABLE_METRICS_EXPORT"] = "1"
os.environ["RAY_OVERRIDE_JOB_RUNTIME_ENV"] = "1"
os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["PYTHONUNBUFFERED"] = "1"

import ray
from ray.tune import PlacementGroupFactory
from confs.model_config import GlobalConfig, PPOConfig, DQNConfig, PPOLSTMConfig
from confs.env_config import EnvConfig

# Direct import of HPO's exact trainable function from tune_models.py!
from scripts.tune_models import train_rllib_trial

@ray.remote
class HPOTrainableActor:
    """Remote actor wrapping HPO's exact execution by directly invoking train_rllib_trial with live console log streaming."""
    def __init__(self, config):
        self.config = config

    def run_training(self):
        from ray import tune
        old_report = getattr(tune, "report", None)

        def live_report(metrics):
            i = metrics.get("training_iteration", 0)
            jsr = metrics.get("jsr", 0.0)
            track = metrics.get("tracking_acc", 0.0)
            power = metrics.get("power", 0.0)
            obj = metrics.get("objective", 0.0)
            if i % 5 == 0 or i == self.config.get("iterations", 1000):
                print(f"[WORKER GPU] Iter {i:4d}/{self.config.get('iterations', 1000)} | 30-Seed JSR: {jsr:5.2f}% | Track: {track:5.2f}% | Power: {power:.3f}W | Obj: {obj:.2f}", flush=True)
                sys.stdout.flush()

        tune.report = live_report
        try:
            return train_rllib_trial(self.config)
        finally:
            if old_report:
                tune.report = old_report

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

    if not ray.is_initialized():
        runtime_env = {
            "working_dir": PROJECT_ROOT,
            "excludes": [".venv", "artifacts", "ray_results", "head-node-logs", ".git", "baseline_q_table", "node_modules"],
            "env_vars": {"PYTHONPATH": ".", "RAY_DEDUP_LOGS": "0", "PYTHONUNBUFFERED": "1"}
        }
        ray.init(ignore_reinit_error=True, runtime_env=runtime_env)

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join(PROJECT_ROOT, "artifacts", "scenario-unified-test", timestamp, f"S{args.scenario}", algo_upper.lower())
    os.makedirs(out_dir, exist_ok=True)

    # Exact placement group bundles matching HPO (STRICT_PACK strategy)
    bundles = [{"CPU": 1, "GPU": 1}] + [{"CPU": 1}] * args.num_workers
    placement_group = PlacementGroupFactory(bundles, strategy="STRICT_PACK")

    print(f"Dispatching HPO remote actor to Worker GPU with STRICT_PACK placement group (Workers: {args.num_workers})...")
    actor = HPOTrainableActor.options(scheduling_strategy=placement_group).remote(trial_config)
    
    start_t = time.time()
    train_future = actor.run_training.remote()
    res = ray.get(train_future)
    elapsed = round(time.time() - start_t, 2)

    results_file = os.path.join(out_dir, "results.json")
    config_file = os.path.join(out_dir, "trial_config.json")

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)

    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(trial_config, f, indent=2)

    print(f"\n==================================================")
    print(f" DIRECT HPO EXECUTION COMPLETED ({elapsed}s)")
    print(f" Algorithm: {algo_upper} | Scenario: {args.scenario}")
    print(f" Result: {json.dumps(res, indent=2)}")
    print(f" Saved Results To: {results_file}")
    print(f" Saved Config To:  {config_file}")
    print(f"==================================================\n")

if __name__ == "__main__":
    main()
