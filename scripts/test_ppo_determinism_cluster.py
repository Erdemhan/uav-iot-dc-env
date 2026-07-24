import os
import sys
import json
import time

# Ensure project root is in PYTHONPATH
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import ray
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

# Initialize Ray with runtime_env for cluster environment
if not ray.is_initialized():
    runtime_env = {
        "working_dir": project_root,
        "excludes": [".venv", "artifacts", "ray_results", "head-node-logs", ".git", "baseline_q_table", "node_modules"],
        "env_vars": {"PYTHONPATH": "."}
    }
    try:
        ray.init(address="auto", ignore_reinit_error=True, runtime_env=runtime_env)
    except Exception:
        ray.init(ignore_reinit_error=True, runtime_env=runtime_env)

from simulation.pettingzoo_env import UAV_IoT_PZ_Env
from confs.model_config import GlobalConfig, PPOConfig as PPOHyperparams
from confs.env_config import EnvConfig
from scripts.tune_models import run_30seeds_eval

def env_creator(config):
    if "num_nodes" in config: EnvConfig.NUM_NODES = int(config["num_nodes"])
    if "num_uavs" in config:  EnvConfig.NUM_UAVS = int(config["num_uavs"])
    if "area_size" in config: EnvConfig.AREA_SIZE = float(config["area_size"])
    if "w_cost" in config:    EnvConfig.W_COST = float(config["w_cost"])
    return ParallelPettingZooEnv(UAV_IoT_PZ_Env(auto_uav=True, flatten_actions=GlobalConfig.FLATTEN_ACTIONS))

@ray.remote(num_gpus=1)
class ClusterPPODeterminismTester:
    """Ray Remote Actor running on a Worker PC with 1 GPU to test PPO determinism."""
    def run_single_test(self, test_name, seed=42, num_iterations=10):
        import torch
        import numpy as np
        import random
        from ray.rllib.algorithms.ppo import PPOConfig

        # Apply seeds strictly inside Worker GPU process
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        env_name = f"uav_iot_ppo_det_{test_name}_v1"
        try:
            register_env(env_name, env_creator)
        except Exception:
            pass

        env_cfg = {
            "seed": seed,
            "num_nodes": 15,
            "num_uavs": 1,
            "area_size": 500.0,
            "w_cost": 0.03
        }

        dummy_env = UAV_IoT_PZ_Env(auto_uav=True, flatten_actions=GlobalConfig.FLATTEN_ACTIONS)
        obs_space = dummy_env.observation_space("jammer_0")
        act_space = dummy_env.action_space("jammer_0")
        node_obs = dummy_env.observation_space("node_0")
        node_act = dummy_env.action_space("node_0")

        config = (
            PPOConfig()
            .environment(env_name, env_config=env_cfg)
            .framework("torch")
            .debugging(seed=seed)
            .env_runners(
                num_env_runners=GlobalConfig.NUM_WORKERS,
                rollout_fragment_length=PPOHyperparams.ROLLOUT_FRAGMENT_LENGTH
            )
            .training(
                model={"fcnet_hiddens": PPOHyperparams.FCNET_HIDDENS},
                train_batch_size=PPOHyperparams.TRAIN_BATCH_SIZE,
                lr=PPOHyperparams.LR,
                gamma=PPOHyperparams.GAMMA,
            )
            .multi_agent(
                policies={
                    "jammer_policy": (None, obs_space, act_space, {}),
                    "node_policy": (None, node_obs, node_act, {}),
                },
                policy_mapping_fn=lambda agent_id, *args, **kwargs: 
                    "jammer_policy" if agent_id == "jammer_0" else "node_policy",
                policies_to_train=["jammer_policy"],
            )
            .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
            .resources(num_gpus=1)
            .debugging(log_level="WARN")
        )

        algo = config.build()
        rows = []

        for i in range(1, num_iterations + 1):
            algo.train()
            eval_metrics = run_30seeds_eval(
                algo_agent=algo,
                algo_name="PPO",
                env_config=env_cfg,
                phase=1,
                lstm_cell_size=256
            )
            rows.append({
                "iter": i,
                "objective": eval_metrics["objective"],
                "jsr": eval_metrics["jsr"],
                "tracking_acc": eval_metrics["tracking_acc"],
                "power": eval_metrics["power"]
            })

        algo.stop()
        return rows

def main():
    print("=========================================================================")
    print("  CLUSTER PPO DETERMINISM TEST (Dispatched to Worker GPU)")
    print("=========================================================================\n")

    tester = ClusterPPODeterminismTester.remote()

    print("[1/2] Dispatched RUN 1 (10 Iterations, Seed=42) to Worker GPU...")
    fut1 = tester.run_single_test.remote(test_name="run1", seed=42, num_iterations=10)
    run1_results = ray.get(fut1)
    print("      [OK] RUN 1 Completed successfully.\n")

    print("[2/2] Dispatched RUN 2 (10 Iterations, Seed=42) to Worker GPU...")
    fut2 = tester.run_single_test.remote(test_name="run2", seed=42, num_iterations=10)
    run2_results = ray.get(fut2)
    print("      [OK] RUN 2 Completed successfully.\n")

    print("=========================================================================")
    print("  KARŞILAŞTIRMA TABLOSU: RUN 1 vs RUN 2 (Tam Float Değerleri)")
    print("=========================================================================")
    print(f"{'Iter':<6} | {'Run 1 Objective':<22} | {'Run 2 Objective':<22} | {'Mutlak Fark':<18}")
    print("-" * 75)

    comparison_details = []
    max_diff = 0.0
    for r1, r2 in zip(run1_results, run2_results):
        obj1 = r1["objective"]
        obj2 = r2["objective"]
        diff = abs(obj1 - obj2)
        if diff > max_diff:
            max_diff = diff
        
        print(f"{r1['iter']:<6d} | {obj1:<22.15f} | {obj2:<22.15f} | {diff:<18.15f}")
        comparison_details.append({
            "iter": r1["iter"],
            "run1_obj": obj1,
            "run2_obj": obj2,
            "abs_diff": diff,
            "is_identical": diff == 0.0
        })

    print("-" * 75)
    print(f"10 İterasyon Boyunca Maksimum Objective Farkı: {max_diff:.15f}")

    if max_diff == 0.0:
        verdict = "VERIFICATION_SUCCESS_100_PERCENT_IDENTICAL"
        print("\n[SONUÇ: BİREBİR AYNI] PPO koşumları aynı Worker GPU üzerinde %100 DETERMINISTIK ve BİREBİR AYNI çıktı!")
    elif max_diff < 1e-6:
        verdict = "VERIFICATION_NOTICE_MINOR_FLOAT_DIFF"
        print(f"\n[SONUÇ: ÇOK KÜÇÜK FARK] Mikro GPU hassasiyet farkı tespit edildi (Maks Fark: {max_diff:.15f}).")
    else:
        verdict = "VERIFICATION_RESULT_DIFFERENT"
        print(f"\n[SONUÇ: FARKLILIK VAR] İki koşum arasında fark tespit edildi (Maks Fark: {max_diff:.6f}).")

    # Save log to disk
    artifacts_dir = os.path.join(project_root, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    log_json_path = os.path.join(artifacts_dir, "ppo_determinism_test_results.json")
    
    save_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "verdict": verdict,
        "max_diff": max_diff,
        "run1_results": run1_results,
        "run2_results": run2_results,
        "comparison_details": comparison_details
    }
    
    with open(log_json_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=4)
        
    print(f"\n[LOG DOKÜMANI SAKLANDI] Tüm sonuçlar ve log dosyası kaydedildi:")
    print(f"👉 {log_json_path}")

if __name__ == "__main__":
    main()
