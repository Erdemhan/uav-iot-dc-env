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
class SinglePPORunner:
    """Ray Remote Actor running on a dedicated Worker GPU with STRICT 1-GPU allocation."""
    def init_algo(self, setup_mode="hpo", seed=42):
        import torch
        import random

        self.setup_mode = setup_mode
        self.seed = seed
        self.env_cfg = {
            "seed": seed,
            "num_nodes": 15,
            "num_uavs": 1,
            "area_size": 500.0,
            "w_cost": 0.03
        }

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

        lr_exact = 0.0006975966228982835
        gamma_exact = 0.9062519087603899

        if setup_mode == "hpo":
            env_name = "uav_iot_ppo_tune_dbg_v1"
            try: register_env(env_name, env_creator_tune)
            except: pass
            env_c = {"seed": seed}
        else:
            env_name = "uav_iot_ppo_train_dbg_v1"
            try: register_env(env_name, env_creator_train)
            except: pass
            env_c = self.env_cfg

        cfg = (
            PPOConfig()
            .environment(env_name, env_config=env_c)
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
        self.algo = cfg.build()
        weights = self.algo.get_policy("jammer_policy").get_weights()
        return {"status": "OK", "weights": weights}

    def set_weights(self, weights):
        self.algo.get_policy("jammer_policy").set_weights(weights)
        return "[OK] Weights updated."

    def train_step(self):
        self.algo.train()
        return True

    def get_weights(self):
        return self.algo.get_policy("jammer_policy").get_weights()

    def evaluate_30seeds(self):
        eval_metrics = run_30seeds_eval(
            self.algo, 
            "PPO", 
            {"seed": self.seed} if self.setup_mode == "hpo" else self.env_cfg, 
            phase=1
        )
        return eval_metrics["objective"]

    def stop_algo(self):
        self.algo.stop()
        return True


def main():
    print("=========================================================================")
    print("  LIVE STEP-BY-STEP HPO vs TRAIN (2 Dedicated Worker GPUs Strict Pack)")
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

    print("[1/2] Instantiating 2 Dedicated Worker GPU Actors (STRICT 1 GPU per Actor)...")
    runner_a = SinglePPORunner.remote()
    runner_b = SinglePPORunner.remote()

    fut_a = runner_a.init_algo.remote(setup_mode="hpo", seed=42)
    fut_b = runner_b.init_algo.remote(setup_mode="train", seed=42)

    init_a, init_b = ray.get([fut_a, fut_b])
    print("      [OK] Both Dedicated Worker GPU Actors created.")

    print("      Equalizing initial neural network weights theta_0...")
    ray.get(runner_b.set_weights.remote(init_a["weights"]))
    print("      [OK] Initial weights theta_0 equalized 100% across both GPUs.\n")

    print("[2/2] Running Live Iteration Comparison (Streaming step by step)...")
    print("=========================================================================")
    print(f"{'Iter':<6} | {'HPO Obj (GPU A)':<20} | {'Train Obj (GPU B)':<20} | {'Obj Farkı':<16} | {'Ağırlık Farkı (||wA-wB||)':<25}")
    print("-" * 95)

    first_diff_iter = None
    total_iterations = 50

    for it in range(1, total_iterations + 1):
        # 1. Step both actors in parallel on separate GPUs
        ray.get([runner_a.train_step.remote(), runner_b.train_step.remote()])

        # 2. Get weights and 30-seed evaluations in parallel
        fut_wa = runner_a.get_weights.remote()
        fut_wb = runner_b.get_weights.remote()
        fut_ea = runner_a.evaluate_30seeds.remote()
        fut_eb = runner_b.evaluate_30seeds.remote()

        w_a, w_b, obj_a, obj_b = ray.get([fut_wa, fut_wb, fut_ea, fut_eb])

        # Compute weight norm diff
        weight_diff = 0.0
        for key in w_a:
            if key in w_b:
                diff_k = np.linalg.norm(np.array(w_a[key]) - np.array(w_b[key]))
                weight_diff += float(diff_k)

        obj_diff = abs(obj_a - obj_b)

        print(f"{it:<6d} | {obj_a:<20.12f} | {obj_b:<20.12f} | {obj_diff:<16.12f} | {weight_diff:<25.12e}", flush=True)

        if (weight_diff > 1e-7 or obj_diff > 1e-7) and first_diff_iter is None:
            first_diff_iter = it

    print("-" * 95)
    if first_diff_iter:
        print(f"\n[SAPMA TESPİT EDİLDİ] İlk Farklılaşma İterasyonu: İTERASYON {first_diff_iter}")
    else:
        print(f"\n[TAM BİREBİR AYNI] {total_iterations} İterasyon boyunca HPO ve Train kurulumları %100 BİREBİR AYNI çıktı!")

    ray.get([runner_a.stop_algo.remote(), runner_b.stop_algo.remote()])


if __name__ == "__main__":
    main()
