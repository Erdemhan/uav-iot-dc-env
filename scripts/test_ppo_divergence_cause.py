import os
import sys
import copy
import json
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

import ray
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.algorithms.ppo import PPOConfig

from simulation.pettingzoo_env import UAV_IoT_PZ_Env
from confs.model_config import GlobalConfig
from confs.env_config import EnvConfig
from scripts.tune_models import run_30seeds_eval

def env_creator_tune(config):
    return ParallelPettingZooEnv(UAV_IoT_PZ_Env(auto_uav=True, flatten_actions=GlobalConfig.FLATTEN_ACTIONS))

def env_creator_train(config):
    if "num_nodes" in config: EnvConfig.NUM_NODES = int(config["num_nodes"])
    if "num_uavs" in config: EnvConfig.NUM_UAVS = int(config["num_uavs"])
    if "area_size" in config: EnvConfig.AREA_SIZE = float(config["area_size"])
    if "w_cost" in config: EnvConfig.W_COST = float(config["w_cost"])
    return ParallelPettingZooEnv(UAV_IoT_PZ_Env(auto_uav=True, flatten_actions=GlobalConfig.FLATTEN_ACTIONS))


@ray.remote(num_gpus=1)
class PPODivergenceTester:
    def test_divergence(self, seed=42):
        import torch
        import random

        lr = 0.0006975966228982835
        gamma = 0.9062519087603899

        # ---------------------------------------------------------
        # RUN A: Exact tune_models.py (HPO) Execution
        # ---------------------------------------------------------
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        env_name_a = "uav_iot_ppo_hpo_exact_v1"
        try: register_env(env_name_a, env_creator_tune)
        except: pass

        dummy_env = UAV_IoT_PZ_Env(auto_uav=True, flatten_actions=GlobalConfig.FLATTEN_ACTIONS)
        obs_space = dummy_env.observation_space("jammer_0")
        act_space = dummy_env.action_space("jammer_0")
        node_obs = dummy_env.observation_space("node_0")
        node_act = dummy_env.action_space("node_0")

        cfg_a = (
            PPOConfig()
            .environment(env_name_a, env_config={"seed": seed})
            .framework("torch")
            .debugging(seed=seed)
            .env_runners(num_env_runners=10, rollout_fragment_length=100)
            .training(
                model={"fcnet_hiddens": [512, 256]},
                train_batch_size=1000,
                lr=lr,
                gamma=gamma,
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
        algo_a = cfg_a.build()
        init_weights = algo_a.get_policy("jammer_policy").get_weights()

        eval_history_a = {}
        for i in range(1, 11):
            algo_a.train()
            if i % 5 == 0:  # HPO only evaluates at i%5==0
                ev = run_30seeds_eval(algo_a, "PPO", {"W_SUCCESS": 0.8, "W_TRACKING": 0.2, "W_COST": 0.03}, phase=1)
                eval_history_a[i] = ev["objective"]

        algo_a.stop()

        # ---------------------------------------------------------
        # RUN B: Exact train.py (Scenario) Execution (has i==1 eval call)
        # ---------------------------------------------------------
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        env_name_b = "uav_iot_ppo_scen_exact_v1"
        try: register_env(env_name_b, env_creator_train)
        except: pass

        cfg_b = (
            PPOConfig()
            .environment(env_name_b, env_config={"seed": seed, "num_nodes": 15, "num_uavs": 1, "area_size": 500.0, "w_cost": 0.03})
            .framework("torch")
            .debugging(seed=seed)
            .env_runners(num_env_runners=10, rollout_fragment_length=100)
            .training(
                model={"fcnet_hiddens": [512, 256]},
                train_batch_size=1000,
                lr=lr,
                gamma=gamma,
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
        algo_b = cfg_b.build()
        algo_b.get_policy("jammer_policy").set_weights(init_weights)

        eval_history_b = {}
        for i in range(1, 11):
            algo_b.train()
            if i % 5 == 0 or i == 1:  # Scenario evaluated at i==1 AND i%5==0
                ev = run_30seeds_eval(algo_b, "PPO", {"seed": seed, "num_nodes": 15, "num_uavs": 1, "area_size": 500.0, "w_cost": 0.03}, phase=1)
                eval_history_b[i] = ev["objective"]

        algo_b.stop()

        # ---------------------------------------------------------
        # RUN C: Scenario Execution WITHOUT i==1 eval call (Fixed train.py)
        # ---------------------------------------------------------
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        algo_c = cfg_b.build()
        algo_c.get_policy("jammer_policy").set_weights(init_weights)

        eval_history_c = {}
        for i in range(1, 11):
            algo_c.train()
            if i % 5 == 0:  # Fixed scenario evaluated ONLY at i%5==0
                ev = run_30seeds_eval(algo_c, "PPO", {"seed": seed, "num_nodes": 15, "num_uavs": 1, "area_size": 500.0, "w_cost": 0.03}, phase=1)
                eval_history_c[i] = ev["objective"]

        algo_c.stop()

        return {
            "eval_a": eval_history_a,  # HPO
            "eval_b": eval_history_b,  # Scenario with i==1
            "eval_c": eval_history_c,  # Scenario without i==1 (Fixed)
        }


def main():
    print("=========================================================================")
    print("  EXACT PPO DIVERGENCE DIAGNOSTIC TEST (Ray GPU Cluster)")
    print("=========================================================================\n")

    runtime_env = {
        "working_dir": PROJECT_ROOT,
        "excludes": [".venv", "artifacts", "ray_results", "head-node-logs", ".git", "node_modules"],
        "env_vars": {"PYTHONPATH": "."}
    }

    if not ray.is_initialized():
        try: ray.init(address="auto", runtime_env=runtime_env)
        except: ray.init(runtime_env=runtime_env)

    tester = PPODivergenceTester.remote()
    print("[1/2] Executing 3 Diagnostic Runs on Worker GPU...")
    res = ray.get(tester.test_divergence.remote(seed=42))

    print("\n=========================================================================")
    print("  RESULTS COMPARISON")
    print("=========================================================================")
    print(f"HPO (Run A) Iter 5 Obj:           {res['eval_a'].get(5):.15f}")
    print(f"Scenario with i==1 (Run B) Iter 5: {res['eval_b'].get(5):.15f}")
    print(f"Scenario FIXED (Run C) Iter 5:    {res['eval_c'].get(5):.15f}")
    print(f"HPO (Run A) Iter 10 Obj:          {res['eval_a'].get(10):.15f}")
    print(f"Scenario with i==1 (Run B) Iter 10:{res['eval_b'].get(10):.15f}")
    print(f"Scenario FIXED (Run C) Iter 10:   {res['eval_c'].get(10):.15f}")

    diff_b5 = abs(res['eval_a'].get(5) - res['eval_b'].get(5))
    diff_c5 = abs(res['eval_a'].get(5) - res['eval_c'].get(5))
    diff_b10 = abs(res['eval_a'].get(10) - res['eval_b'].get(10))
    diff_c10 = abs(res['eval_a'].get(10) - res['eval_c'].get(10))

    print("-" * 65)
    print(f"Diff Iter 5 (Run A vs Run B with i==1): {diff_b5:.15f}")
    print(f"Diff Iter 5 (Run A vs Run C FIXED):    {diff_c5:.15f}")
    print(f"Diff Iter 10 (Run A vs Run B with i==1):{diff_b10:.15f}")
    print(f"Diff Iter 10 (Run A vs Run C FIXED):   {diff_c10:.15f}")

    # Save to JSON artifact on disk as well
    out_dir = os.path.join(PROJECT_ROOT, "artifacts")
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "ppo_divergence_test_results.json")
    with open(json_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n[OK] Full JSON test results saved to disk: {json_path}")


if __name__ == "__main__":
    main()
