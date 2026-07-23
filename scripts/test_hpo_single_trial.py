import os
import sys
import json
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

os.environ["RAY_DISABLE_METRICS_EXPORT"] = "1"
os.environ["RAY_OVERRIDE_JOB_RUNTIME_ENV"] = "1"
os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np
import core.physics
from confs.env_config import EnvConfig
from confs.model_config import UAVConfig

# Override calculate_path_loss with OLD Friis formula (with 1/eta factor as of July 18)
def old_calculate_path_loss(d: float = 1.0, fc: float = UAVConfig.FC, eta: float = UAVConfig.ETA) -> float:
    c = UAVConfig.C
    beta_0 = (1 / eta) * ( (4 * np.pi * fc) / c )**(-2)
    return beta_0

core.physics.calculate_path_loss = old_calculate_path_loss
print("[OLD PHYSICS EMBEDDED] Overrode core.physics.calculate_path_loss with pre-July 22 formula (1/eta factor).")

import ray
from ray import tune
from confs.model_config import GlobalConfig
from scripts.tune_models import train_rllib_trial

def main():
    print("\n==================================================")
    print(" 30-SECOND OLD PHYSICS EMPIRICAL TUNE.RUN VERIFICATION TEST")
    print(" Testing pre-July 22 physics formula on Worker GPU")
    print("==================================================\n")

    # Set Scenario 1-A EnvConfig
    EnvConfig.NUM_NODES = 15; EnvConfig.NUM_UAVS = 1; EnvConfig.AREA_SIZE = 500.0; EnvConfig.W_COST = 0.03

    if not ray.is_initialized():
        runtime_env = {
            "working_dir": PROJECT_ROOT,
            "excludes": [".venv", "artifacts", "ray_results", "head-node-logs", ".git", "baseline_q_table", "node_modules"],
            "env_vars": {"PYTHONPATH": ".", "RAY_DEDUP_LOGS": "0", "PYTHONUNBUFFERED": "1"}
        }
        ray.init(ignore_reinit_error=True, runtime_env=runtime_env)

    # Exact Trial 64 parameters matching HPO
    trial64_config = {
        "algo": "DQN",
        "lr": 0.0003204387357042751,
        "gamma": 0.8721224348952492,
        "architecture": "128,512,512",
        "target_network_update_freq": 2000,
        "num_workers": 10,
        "num_gpus": 1,
        "iterations": 5,
        "phase": 1,
        "env_config": {
            "W_SUCCESS": 0.8,
            "W_TRACKING": 0.2,
            "W_COST": 0.03
        }
    }

    from ray.tune import PlacementGroupFactory
    bundles = [{"CPU": 1, "GPU": 1}] + [{"CPU": 1}] * 10
    trial_resources = PlacementGroupFactory(bundles, strategy="STRICT_PACK")

    out_dir = os.path.join(PROJECT_ROOT, "artifacts", "hpo_old_physics_test_run")
    os.makedirs(out_dir, exist_ok=True)

    print("Starting tune.run() test with OLD physics formula on Worker GPU...")
    start_t = time.time()
    analysis = tune.run(
        train_rllib_trial,
        config=trial64_config,
        resources_per_trial=trial_resources,
        num_samples=1,
        storage_path=out_dir,
        name="test_trial64_old_physics",
        verbose=1
    )
    elapsed = round(time.time() - start_t, 2)

    trial = analysis.trials[0]
    last_result = trial.last_result

    print(f"\n==================================================")
    print(f" OLD PHYSICS EMPIRICAL TEST COMPLETED IN {elapsed}s")
    print(f" Iteration 5 Results (Expected HPO Target: Power=0.240889W, Obj=-0.722667):")
    print(f"   JSR:      {last_result.get('jsr', 0.0):.2f}%")
    print(f"   Tracking: {last_result.get('tracking_acc', 0.0):.2f}%")
    print(f"   Power:    {last_result.get('power', 0.0):.6f} W")
    print(f"   Objective:{last_result.get('objective', 0.0):.6f}")
    print(f"==================================================\n")

if __name__ == "__main__":
    main()
