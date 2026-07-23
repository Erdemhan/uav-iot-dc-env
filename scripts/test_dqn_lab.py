import os
import sys
import argparse
import time
import json
import csv
import numpy as np

# Force Matplotlib headless backend
os.environ["MPLBACKEND"] = "Agg"
os.environ["RAY_DISABLE_METRICS_EXPORT"] = "1"
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")
warnings.filterwarnings("ignore", category=UserWarning, module="pettingzoo")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

import ray
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.algorithms.dqn import DQNConfig as RLlibDQNConfig

from simulation.pettingzoo_env import UAV_IoT_PZ_Env
from confs.env_config import EnvConfig
from confs.model_config import GlobalConfig
from confs.opt_config import OptConfig

def env_creator(config):
    if "num_nodes" in config: EnvConfig.NUM_NODES = int(config["num_nodes"])
    if "num_uavs" in config: EnvConfig.NUM_UAVS = int(config["num_uavs"])
    if "area_size" in config: EnvConfig.AREA_SIZE = float(config["area_size"])
    if "w_cost" in config: EnvConfig.W_COST = float(config["w_cost"])
    return ParallelPettingZooEnv(UAV_IoT_PZ_Env(auto_uav=True, flatten_actions=GlobalConfig.FLATTEN_ACTIONS))

def run_30seeds_eval(algo_agent, env_cfg):
    """Evaluates agent on 30 random seeds with explore=False (Greedy)"""
    ep_jsrs = []
    ep_trackings = []
    ep_powers = []
    ep_sinrs = []
    
    seeds = OptConfig.EVAL_SEEDS
    for seed in seeds:
        eval_env = UAV_IoT_PZ_Env(logger=None, auto_uav=True, flatten_actions=GlobalConfig.FLATTEN_ACTIONS)
        eval_env.w_success = env_cfg.get("W_SUCCESS", EnvConfig.W_SUCCESS)
        eval_env.w_tracking = 1.0 - eval_env.w_success
        eval_env.w_cost = env_cfg.get("W_COST", EnvConfig.W_COST)
        
        obs, infos = eval_env.reset(seed=seed)
        terminated = False
        steps = 0
        ep_jammed = 0
        ep_reachable = 0
        ep_tracking_reachable = 0
        ep_power_sum = 0
        ep_sinr_sum = 0
        ep_sinr_count = 0
        
        while not terminated and steps < EnvConfig.MAX_STEPS:
            steps += 1
            if "jammer_0" in obs:
                jam_obs = obs["jammer_0"]
                try:
                    jam_action = algo_agent.compute_single_action(jam_obs, policy_id="jammer_policy", explore=False)
                except Exception:
                    jam_action = 0
            else:
                jam_action = 0
                
            actions = {"jammer_0": jam_action}
            for ag in eval_env.agents:
                if "node" in ag: actions[ag] = 0
                
            obs, rewards, term, trunc, infos = eval_env.step(actions)
            terminated = any(term.values()) or any(trunc.values())
            
            reachable_count = sum(1 for n in eval_env.nodes if n.connection_status != 1)
            jammed_c = infos.get("jammer_0", {}).get("jammed_count", 0)
            ep_jammed += jammed_c
            if reachable_count > 0:
                ep_reachable += reachable_count
                
            closest_uav = min(eval_env.uavs, key=lambda uav: np.linalg.norm(uav.position - eval_env.attacker.position))
            if eval_env.attacker.current_channel == closest_uav.current_channel:
                if reachable_count > 0:
                    ep_tracking_reachable += 1
                    
            pwr = infos.get("jammer_0", {}).get("jammer_cost", 0)
            ep_power_sum += pwr
            
            s_sum = sum(infos.get("jammer_0", {}).get(f"node_{k}_sinr", 0) for k in range(EnvConfig.NUM_NODES))
            s_avg = s_sum / EnvConfig.NUM_NODES
            s_db = 10 * np.log10(max(s_avg, 1e-12))
            ep_sinr_sum += s_db
            ep_sinr_count += 1

        jsr = (ep_jammed / ep_reachable * 100.0) if ep_reachable > 0 else 0.0
        track = (ep_tracking_reachable / ep_reachable * 100.0) if ep_reachable > 0 else 0.0
        power = ep_power_sum / steps if steps > 0 else 0.0
        sinr = ep_sinr_sum / ep_sinr_count if ep_sinr_count > 0 else 0.0
        
        ep_jsrs.append(jsr)
        ep_trackings.append(track)
        ep_powers.append(power)
        ep_sinrs.append(sinr)
        
    return {
        "jsr_mean": float(np.mean(ep_jsrs)),
        "jsr_std": float(np.std(ep_jsrs)),
        "track_mean": float(np.mean(ep_trackings)),
        "track_std": float(np.std(ep_trackings)),
        "power_mean": float(np.mean(ep_powers)),
        "sinr_mean": float(np.mean(ep_sinrs))
    }

@ray.remote(num_gpus=1, max_concurrency=2)
class LabDQNClusterGPUTrainer:
    """Ray Remote Actor guaranteed to run on a Worker PC with an active GPU."""
    def __init__(self, scenario, iterations, eval_freq, num_workers, env_cfg_dict):
        self.scenario = scenario
        self.iterations = iterations
        self.eval_freq = eval_freq
        self.num_workers = num_workers
        self.env_cfg_dict = env_cfg_dict
        self.progress_rows = []

    def get_progress(self):
        return list(self.progress_rows)

    def train_on_gpu(self):
        import torch
        import random
        torch.set_num_threads(2)
        random.seed(GlobalConfig.RANDOM_SEED)
        torch.manual_seed(GlobalConfig.RANDOM_SEED)
        np.random.seed(GlobalConfig.RANDOM_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(GlobalConfig.RANDOM_SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        if "num_nodes" in self.env_cfg_dict: EnvConfig.NUM_NODES = int(self.env_cfg_dict["num_nodes"])
        if "num_uavs" in self.env_cfg_dict: EnvConfig.NUM_UAVS = int(self.env_cfg_dict["num_uavs"])
        if "area_size" in self.env_cfg_dict: EnvConfig.AREA_SIZE = float(self.env_cfg_dict["area_size"])
        if "w_cost" in self.env_cfg_dict: EnvConfig.W_COST = float(self.env_cfg_dict["w_cost"])

        env_name = f"uav_iot_dqn_lab_{self.scenario.lower()}_gpu_v1"
        try:
            register_env(env_name, env_creator)
        except Exception:
            pass

        dummy_env = UAV_IoT_PZ_Env(auto_uav=True, flatten_actions=GlobalConfig.FLATTEN_ACTIONS)
        obs_space = dummy_env.observation_space("jammer_0")
        act_space = dummy_env.action_space("jammer_0")
        node_obs = dummy_env.observation_space("node_0")
        node_act = dummy_env.action_space("node_0")

        cfg_obj = (
            RLlibDQNConfig()
            .environment(env_name, env_config=self.env_cfg_dict)
            .framework("torch")
            .debugging(seed=GlobalConfig.RANDOM_SEED)
            .env_runners(num_env_runners=self.num_workers, rollout_fragment_length=100)
            .training(
                model={"fcnet_hiddens": [128, 512, 512]},
                gamma=0.8721224348952492,
                lr=0.0003204387357042751,
                train_batch_size=1000,
                target_network_update_freq=2000,
                double_q=True,
                dueling=True,
                replay_buffer_config={"type": "ReplayBuffer", "capacity": 50000},
                num_steps_sampled_before_learning_starts=0,
            )
            .multi_agent(
                policies={
                    "jammer_policy": (None, obs_space, act_space, {}),
                    "node_policy": (None, node_obs, node_act, {}),
                },
                policy_mapping_fn=lambda agent_id, *args, **kwargs: "jammer_policy" if agent_id == "jammer_0" else "node_policy",
                policies_to_train=["jammer_policy"],
            )
            .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
            .resources(num_gpus=1)
            .debugging(log_level="WARN")
        )

        algo = cfg_obj.build()
        print(f"[Worker GPU] DQN initialized on Worker PC with GPU for scenario {self.scenario}...")

        start_time = time.time()
        last_eval = {"jsr_mean": 0.0, "jsr_std": 0.0, "track_mean": 0.0, "power_mean": 0.0, "sinr_mean": 0.0}

        for i in range(1, self.iterations + 1):
            res = algo.train()
            ep_reward = res.get("episode_reward_mean", 0.0)

            if i % self.eval_freq == 0 or i == self.iterations:
                last_eval = run_30seeds_eval(algo, self.env_cfg_dict)
                elapsed = round(time.time() - start_time, 2)
                print(f"[Worker GPU] Iter {i:4d}/{self.iterations} | Reward: {ep_reward:6.2f} | 30-Seed Eval JSR: {last_eval['jsr_mean']:5.2f}% ± {last_eval['jsr_std']:4.2f}% | Track: {last_eval['track_mean']:5.2f}% | Power: {last_eval['power_mean']:.3f}W ({elapsed}s)")

            self.progress_rows.append({
                "iteration": i,
                "episode_reward_mean": ep_reward,
                "eval_jsr_mean": last_eval["jsr_mean"],
                "eval_jsr_std": last_eval["jsr_std"],
                "eval_track_mean": last_eval["track_mean"],
                "eval_power_mean": last_eval["power_mean"],
                "eval_sinr_mean": last_eval["sinr_mean"],
                "time_s": round(time.time() - start_time, 2)
            })

        # Save Checkpoint to Worker PC tmp
        tmp_ckpt = f"/tmp/ckpt_dqn_lab_{self.scenario}"
        if os.path.exists(tmp_ckpt):
            import shutil
            shutil.rmtree(tmp_ckpt)
        os.makedirs(tmp_ckpt, exist_ok=True)
        algo.save(checkpoint_dir=tmp_ckpt)

        checkpoint_bytes = {}
        for root, dirs, files in os.walk(tmp_ckpt):
            for f in files:
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, tmp_ckpt)
                with open(full_path, "rb") as fp:
                    checkpoint_bytes[rel_path] = fp.read()

        algo.stop()
        return {
            "checkpoint_bytes": checkpoint_bytes,
            "progress_rows": self.progress_rows
        }

def main():
    parser = argparse.ArgumentParser(description="Lab DQN Empirical Verification Script (Ray Cluster Remote GPU Actor)")
    parser.add_argument("--scenario", type=str, default="1-A", choices=["1-A", "1-B", "2-A", "2-B"], help="Scenario ID")
    parser.add_argument("--iterations", type=int, default=1000, help="Training iterations")
    parser.add_argument("--eval-freq", type=int, default=5, help="30-seed evaluation frequency (every N iterations)")
    parser.add_argument("--num-workers", type=int, default=10, help="Rollout workers per trial")
    args = parser.parse_args()

    print(f"\n==================================================")
    print(f" LAB DQN CLUSTER GPU VERIFICATION RUN: Scenario {args.scenario}")
    print(f" Iterations: {args.iterations} | Workers: {args.num_workers} | Eval Freq: {args.eval_freq}")
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
        "num_nodes": int(EnvConfig.NUM_NODES),
        "num_uavs": int(EnvConfig.NUM_UAVS),
        "area_size": float(EnvConfig.AREA_SIZE),
        "w_cost": float(EnvConfig.W_COST),
        "W_SUCCESS": EnvConfig.W_SUCCESS,
        "W_TRACKING": EnvConfig.W_TRACKING,
        "W_COST": EnvConfig.W_COST
    }

    if not ray.is_initialized():
        runtime_env = {
            "working_dir": PROJECT_ROOT,
            "excludes": [".venv", "artifacts", "ray_results", "head-node-logs", ".git", "baseline_q_table", "node_modules"],
            "env_vars": {"PYTHONPATH": "."}
        }
        ray.init(ignore_reinit_error=True, runtime_env=runtime_env)

    # Instantiate LabDQNClusterGPUTrainer on Worker PC with 1 GPU!
    trainer_actor = LabDQNClusterGPUTrainer.remote(
        scenario=args.scenario,
        iterations=args.iterations,
        eval_freq=args.eval_freq,
        num_workers=args.num_workers,
        env_cfg_dict=env_cfg_dict
    )

    print(f"Dispatched DQN GPU training actor to a Worker PC in the Ray Cluster...")
    train_future = trainer_actor.train_on_gpu.remote()

    out_dir = os.path.join(PROJECT_ROOT, "artifacts", "test_dqn_lab", f"S{args.scenario}")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "progress_dqn_lab.csv")
    fieldnames = ["iteration", "episode_reward_mean", "eval_jsr_mean", "eval_jsr_std", "eval_track_mean", "eval_power_mean", "eval_sinr_mean", "time_s"]

    # Poll progress every 3 seconds from Head Node while Worker GPU executes training
    while True:
        ready, _ = ray.wait([train_future], timeout=3.0)
        try:
            current_rows = ray.get(trainer_actor.get_progress.remote())
            if current_rows:
                with open(csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(current_rows)
        except Exception:
            pass
        if ready:
            break

    res = ray.get(train_future)
    checkpoint_bytes = res["checkpoint_bytes"]
    progress_rows = res["progress_rows"]

    # Write progress CSV and Checkpoint to Head Node disk
    if progress_rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(progress_rows)

    ckpt_dir = os.path.join(out_dir, "checkpoint_001000")
    os.makedirs(ckpt_dir, exist_ok=True)
    for rel_path, content in checkpoint_bytes.items():
        dest = os.path.join(ckpt_dir, rel_path)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as fp:
            fp.write(content)

    last_eval = progress_rows[-1] if progress_rows else {}
    print("\n==================================================")
    print(" LAB DQN CLUSTER GPU VERIFICATION COMPLETED")
    print(f" Final 1000th Iter 30-Seed JSR:   {last_eval.get('eval_jsr_mean', 0.0):.2f}% ± {last_eval.get('eval_jsr_std', 0.0):.2f}%")
    print(f" Final 1000th Iter Tracking Acc: {last_eval.get('eval_track_mean', 0.0):.2f}%")
    print(f" Final 1000th Iter Avg Power:    {last_eval.get('eval_power_mean', 0.0):.3f} W")
    print(f" Progress CSV Saved: {csv_path}")
    print(f" Checkpoint Saved:   {ckpt_dir}")
    print("==================================================\n")

if __name__ == "__main__":
    main()
