import os
import sys
import copy
import json
import numpy as np

# Ensure project root in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

import ray
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.algorithms.ppo import PPOConfig

from simulation.pettingzoo_env import UAV_IoT_PZ_Env
from confs.model_config import GlobalConfig, PPOConfig as PPOHyperparams
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
class HPOVsTrainRunner:
    """Ray Remote Actor to compare HPO setup vs train_ppo.py setup step-by-step."""
    def init_algos(self, seed=42):
        import torch
        import random

        self.seed = seed
        self.env_cfg_b = {
            "seed": seed,
            "num_nodes": 15,
            "num_uavs": 1,
            "area_size": 500.0,
            "w_cost": 0.03
        }

        # 1. BUILD ALGORITHM A (HPO tune_models.py exact setup)
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        env_name_a = f"uav_iot_ppo_tune_dbg_v1"
        try: register_env(env_name_a, env_creator_tune)
        except: pass

        dummy_env = UAV_IoT_PZ_Env(auto_uav=True, flatten_actions=GlobalConfig.FLATTEN_ACTIONS)
        obs_space = dummy_env.observation_space("jammer_0")
        act_space = dummy_env.action_space("jammer_0")
        node_obs = dummy_env.observation_space("node_0")
        node_act = dummy_env.action_space("node_0")

        lr_exact = 0.0006975966228982835
        gamma_exact = 0.9062519087603899

        cfg_a = (
            PPOConfig()
            .environment(env_name_a, env_config={"seed": seed})
            .framework("torch")
            .debugging(seed=seed)
            .env_runners(num_env_runners=10, rollout_fragment_length=100)
            .training(
                model={"fcnet_hiddens": [512, 256]},
                train_batch_size=1000,
                lr=lr_exact,
                gamma=gamma_exact,
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
        self.algo_a = cfg_a.build()

        # 2. BUILD ALGORITHM B (train_ppo.py exact setup)
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        env_name_b = f"uav_iot_ppo_train_dbg_v1"
        try: register_env(env_name_b, env_creator_train)
        except: pass

        cfg_b = (
            PPOConfig()
            .environment(env_name_b, env_config=self.env_cfg_b)
            .framework("torch")
            .debugging(seed=seed)
            .env_runners(num_env_runners=10, rollout_fragment_length=100)
            .training(
                model={"fcnet_hiddens": [512, 256]},
                train_batch_size=1000,
                lr=lr_exact,
                gamma=gamma_exact,
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
        self.algo_b = cfg_b.build()

        # Equalize initial weights so both start from the exact same initial state theta_0
        init_weights_a = self.algo_a.get_policy("jammer_policy").get_weights()
        self.algo_b.get_policy("jammer_policy").set_weights(init_weights_a)

        return "[OK] Both remote algorithms initialized with EQUAL INITIAL WEIGHTS (theta_0) successfully."

    def step_and_compare(self, iteration):
        self.algo_a.train()
        self.algo_b.train()

        # Compare weights
        w_a = self.algo_a.get_policy("jammer_policy").get_weights()
        w_b = self.algo_b.get_policy("jammer_policy").get_weights()

        weight_diff = 0.0
        for key in w_a:
            if key in w_b:
                diff_k = np.linalg.norm(np.array(w_a[key]) - np.array(w_b[key]))
                weight_diff += float(diff_k)

        # Evaluate 30 seeds
        eval_a = run_30seeds_eval(self.algo_a, "PPO", {"seed": self.seed}, phase=1)
        eval_b = run_30seeds_eval(self.algo_b, "PPO", self.env_cfg_b, phase=1)

        obj_a = eval_a["objective"]
        obj_b = eval_b["objective"]
        obj_diff = abs(obj_a - obj_b)

        return {
            "iter": iteration,
            "weight_diff": weight_diff,
            "obj_a": obj_a,
            "obj_b": obj_b,
            "obj_diff": obj_diff
        }

    def stop_algos(self):
        self.algo_a.stop()
        self.algo_b.stop()
        return "[OK] Algorithms stopped."


def main():
    print("=========================================================================")
    print("  LIVE STEP-BY-STEP HPO vs TRAIN COMPARISON TRACER (Ray GPU Cluster)")
    print("=========================================================================\n")

    runtime_env = {
        "working_dir": PROJECT_ROOT,
        "excludes": [".venv", "artifacts", "ray_results", "head-node-logs", ".git", "node_modules"],
        "env_vars": {"PYTHONPATH": "."}
    }

    if not ray.is_initialized():
        try:
            ray.init(address="auto", runtime_env=runtime_env)
        except Exception:
            ray.init(runtime_env=runtime_env)

    runner = HPOVsTrainRunner.remote()
    print("[1/2] Initializing Algorithms on Remote Worker GPU...")
    status = ray.get(runner.init_algos.remote(seed=42))
    print(f"      {status}\n")

    print("[2/2] Running Iterations LIVE (Streaming output line by line)...")
    print("=========================================================================")
    print(f"{'Iter':<6} | {'HPO Obj (A)':<20} | {'Train Obj (B)':<20} | {'Obj Farkı':<16} | {'Ağırlık Farkı (||wA-wB||)':<25}")
    print("-" * 95)

    first_diff_iter = None
    total_iterations = 50

    for it in range(1, total_iterations + 1):
        r = ray.get(runner.step_and_compare.remote(it))
        print(f"{r['iter']:<6d} | {r['obj_a']:<20.12f} | {r['obj_b']:<20.12f} | {r['obj_diff']:<16.12f} | {r['weight_diff']:<25.12e}", flush=True)

        if (r['weight_diff'] > 1e-7 or r['obj_diff'] > 1e-7) and first_diff_iter is None:
            first_diff_iter = r['iter']

    print("-" * 95)
    if first_diff_iter:
        print(f"\n[SAPMA TESPİT EDİLDİ] İlk Farklılaşma İterasyonu: İTERASYON {first_diff_iter}")
    else:
        print(f"\n[TAM BİREBİR AYNI] {total_iterations} İterasyon boyunca HPO ve Train kurulumları %100 BİREBİR AYNI çıktı!")

    ray.get(runner.stop_algos.remote())


if __name__ == "__main__":
    main()
