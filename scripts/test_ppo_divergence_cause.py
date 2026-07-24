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
class SinglePPODivergenceRunner:
    """Ray Remote Actor running on a dedicated Worker GPU for 3-way parallel testing."""
    def run_test(self, mode="hpo", seed=42, iterations=10):
        import torch
        import random

        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        dummy_env = UAV_IoT_PZ_Env(auto_uav=True, flatten_actions=GlobalConfig.FLATTEN_ACTIONS)
        obs_space = dummy_env.observation_space("jammer_0")
        act_space = dummy_env.action_space("jammer_0")
        node_obs = dummy_env.observation_space("node_0")
        node_act = dummy_env.action_space("node_0")

        lr = 0.0006975966228982835
        gamma = 0.9062519087603899

        if mode == "hpo":
            env_name = "uav_iot_ppo_hpo_parallel_v1"
            try: register_env(env_name, env_creator_tune)
            except: pass
            env_c = {"seed": seed}
            eval_env_c = {"W_SUCCESS": 0.8, "W_TRACKING": 0.2, "W_COST": 0.03}
        else:
            env_name = f"uav_iot_ppo_{mode}_parallel_v1"
            try: register_env(env_name, env_creator_train)
            except: pass
            env_c = {"seed": seed, "num_nodes": 15, "num_uavs": 1, "area_size": 500.0, "w_cost": 0.03}
            eval_env_c = env_c

        cfg = (
            PPOConfig()
            .environment(env_name, env_config=env_c)
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
        algo = cfg.build()

        eval_history = {}
        for i in range(1, iterations + 1):
            algo.train()
            # Mode condition evaluation
            should_eval = False
            if mode == "hpo" and (i % 5 == 0 or i == iterations):
                should_eval = True
            elif mode == "scen_i1" and (i % 5 == 0 or i == iterations or i == 1):
                should_eval = True
            elif mode == "scen_fixed" and (i % 5 == 0 or i == iterations):
                should_eval = True

            if should_eval:
                ev = run_30seeds_eval(algo, "PPO", eval_env_c, phase=1)
                eval_history[i] = ev["objective"]

        algo.stop()
        return eval_history


def main():
    print("=========================================================================")
    print("  3-WAY PARALLEL PPO DIVERGENCE DIAGNOSTIC TEST (Ray GPU Cluster)")
    print("=========================================================================\n")

    runtime_env = {
        "working_dir": PROJECT_ROOT,
        "excludes": [".venv", "artifacts", "ray_results", "head-node-logs", ".git", "node_modules"],
        "env_vars": {"PYTHONPATH": "."}
    }

    if not ray.is_initialized():
        try: ray.init(address="auto", runtime_env=runtime_env)
        except: ray.init(runtime_env=runtime_env)

    print("[1/2] Instantiating 3 Parallel Dedicated Worker GPU Actors...")
    actor_a = SinglePPODivergenceRunner.remote()
    actor_b = SinglePPODivergenceRunner.remote()
    actor_c = SinglePPODivergenceRunner.remote()

    print("[2/2] Launching 3 Diagnostic Runs IN PARALLEL on 3 Separate GPUs...")
    fut_a = actor_a.run_test.remote(mode="hpo", seed=42, iterations=10)
    fut_b = actor_b.run_test.remote(mode="scen_i1", seed=42, iterations=10)
    fut_c = actor_c.run_test.remote(mode="scen_fixed", seed=42, iterations=10)

    eval_a, eval_b, eval_c = ray.get([fut_a, fut_b, fut_c])

    print("\n=========================================================================")
    print("  PARALLEL RESULTS COMPARISON")
    print("=========================================================================")
    print(f"HPO (Run A) Iter 5 Obj:            {eval_a.get(5):.15f}")
    print(f"Scenario with i==1 (Run B) Iter 5: {eval_b.get(5):.15f}")
    print(f"Scenario FIXED (Run C) Iter 5:     {eval_c.get(5):.15f}")
    print(f"HPO (Run A) Iter 10 Obj:           {eval_a.get(10):.15f}")
    print(f"Scenario with i==1 (Run B) Iter 10:{eval_b.get(10):.15f}")
    print(f"Scenario FIXED (Run C) Iter 10:    {eval_c.get(10):.15f}")

    diff_b5 = abs(eval_a.get(5) - eval_b.get(5))
    diff_c5 = abs(eval_a.get(5) - eval_c.get(5))
    diff_b10 = abs(eval_a.get(10) - eval_b.get(10))
    diff_c10 = abs(eval_a.get(10) - eval_c.get(10))

    print("-" * 65)
    print(f"Diff Iter 5 (Run A vs Run B with i==1): {diff_b5:.15f}")
    print(f"Diff Iter 5 (Run A vs Run C FIXED):    {diff_c5:.15f}")
    print(f"Diff Iter 10 (Run A vs Run B with i==1):{diff_b10:.15f}")
    print(f"Diff Iter 10 (Run A vs Run C FIXED):   {diff_c10:.15f}")

    res_dict = {
        "HPO_RunA": eval_a,
        "Scenario_with_i1_RunB": eval_b,
        "Scenario_FIXED_RunC": eval_c,
        "diff_iter5_with_i1": diff_b5,
        "diff_iter5_FIXED": diff_c5,
        "diff_iter10_with_i1": diff_b10,
        "diff_iter10_FIXED": diff_c10
    }

    out_dir = os.path.join(PROJECT_ROOT, "artifacts")
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "ppo_divergence_test_results.json")
    with open(json_path, "w") as f:
        json.dump(res_dict, f, indent=2)

    print(f"\n[OK] Parallel test completed. Results saved to: {json_path}")


if __name__ == "__main__":
    main()
