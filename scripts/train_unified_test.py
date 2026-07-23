import os
import sys
import time
import json
import csv
import shutil
import warnings
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# Suppress Ray metrics exporter warnings and allow job environment override
os.environ["RAY_DISABLE_METRICS_EXPORT"] = "1"
os.environ["RAY_OVERRIDE_JOB_RUNTIME_ENV"] = "1"

# Filter specific warnings to clean up output
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")
warnings.filterwarnings("ignore", category=UserWarning, module="pettingzoo")

import ray
import numpy as np
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env

from simulation.pettingzoo_env import UAV_IoT_PZ_Env
from confs.model_config import GlobalConfig, PPOConfig as PPOHyperparams, DQNConfig as DQNHyperparams, PPOLSTMConfig
from confs.env_config import EnvConfig
from confs.opt_config import OptConfig
from scripts.tune_models import run_30seeds_eval

def env_creator(config):
    # Override EnvConfig dynamically for remote workers
    if "num_nodes" in config:
        EnvConfig.NUM_NODES = int(config["num_nodes"])
    if "num_uavs" in config:
        EnvConfig.NUM_UAVS = int(config["num_uavs"])
    if "area_size" in config:
        EnvConfig.AREA_SIZE = float(config["area_size"])
    if "w_cost" in config:
        EnvConfig.W_COST = float(config["w_cost"])
        
    return ParallelPettingZooEnv(UAV_IoT_PZ_Env(auto_uav=True, flatten_actions=GlobalConfig.FLATTEN_ACTIONS))

@ray.remote(num_gpus=1, max_concurrency=2)
class UnifiedClusterGPUTrainer:
    """Ray Remote Actor guaranteed to run on a Worker PC with an active GPU using exact HPO evaluation pipeline."""
    def __init__(self, algo_name, scenario, output_dir, env_cfg, ppo_hp=None, dqn_hp=None, lstm_hp=None, eval_freq=5):
        self.algo_name = algo_name
        self.scenario = scenario
        self.output_dir = output_dir
        self.env_cfg = env_cfg
        self.ppo_hp = ppo_hp
        self.dqn_hp = dqn_hp
        self.lstm_hp = lstm_hp
        self.eval_freq = eval_freq
        self.progress_rows = []

    def get_progress(self):
        """Returns current training progress rows for real-time polling by Head Node."""
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

        env_name = f"uav_iot_unified_{self.algo_name}_gpu_v1"
        for name in [f"uav_iot_unified_{self.algo_name}_v1", env_name]:
            try:
                register_env(name, env_creator)
            except Exception:
                pass

        if "num_nodes" in self.env_cfg:
            EnvConfig.NUM_NODES = int(self.env_cfg["num_nodes"])
        if "num_uavs" in self.env_cfg:
            EnvConfig.NUM_UAVS = int(self.env_cfg["num_uavs"])
        if "area_size" in self.env_cfg:
            EnvConfig.AREA_SIZE = float(self.env_cfg["area_size"])
        if "w_cost" in self.env_cfg:
            EnvConfig.W_COST = float(self.env_cfg["w_cost"])

        dummy_env = UAV_IoT_PZ_Env(auto_uav=True, flatten_actions=GlobalConfig.FLATTEN_ACTIONS)
        obs_space = dummy_env.observation_space("jammer_0")
        act_space = dummy_env.action_space("jammer_0")
        node_obs = dummy_env.observation_space("node_0")
        node_act = dummy_env.action_space("node_0")

        # Load Tuned Hyperparameters
        if self.algo_name == "ppo":
            lr = self.ppo_hp.LR if self.ppo_hp else PPOHyperparams.LR
            gamma = self.ppo_hp.GAMMA if self.ppo_hp else PPOHyperparams.GAMMA
            hidden_layers = self.ppo_hp.MODEL_CONFIG["fcnet_hiddens"] if self.ppo_hp else PPOHyperparams.MODEL_CONFIG["fcnet_hiddens"]
            
            config = (
                PPOConfig()
                .environment(env_name, env_config=self.env_cfg)
                .framework("torch")
                .debugging(seed=GlobalConfig.RANDOM_SEED)
                .env_runners(
                    num_env_runners=GlobalConfig.NUM_WORKERS,
                    rollout_fragment_length=GlobalConfig.ROLLOUT_FRAGMENT_LENGTH
                )
                .training(
                    model={"fcnet_hiddens": hidden_layers},
                    gamma=gamma,
                    lr=lr,
                    train_batch_size=GlobalConfig.TRAIN_BATCH_SIZE,
                    sgd_minibatch_size=PPOHyperparams.SGD_MINIBATCH_SIZE,
                    num_sgd_iter=PPOHyperparams.NUM_SGD_ITER,
                    clip_param=PPOHyperparams.CLIP_PARAM,
                    vf_loss_coeff=PPOHyperparams.VF_LOSS_COEFF,
                    entropy_coeff=PPOHyperparams.ENTROPY_COEFF
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

        elif self.algo_name == "dqn":
            lr = self.dqn_hp.LR if self.dqn_hp else DQNHyperparams.LR
            gamma = self.dqn_hp.GAMMA if self.dqn_hp else DQNHyperparams.GAMMA
            hidden_layers = self.dqn_hp.MODEL_CONFIG["fcnet_hiddens"] if self.dqn_hp else DQNHyperparams.MODEL_CONFIG["fcnet_hiddens"]
            target_freq = getattr(self.dqn_hp, "TARGET_NETWORK_UPDATE_FREQ", 2000) if self.dqn_hp else 2000

            config = (
                DQNConfig()
                .environment(env_name, env_config=self.env_cfg)
                .framework("torch")
                .debugging(seed=GlobalConfig.RANDOM_SEED)
                .env_runners(
                    num_env_runners=GlobalConfig.NUM_WORKERS,
                    rollout_fragment_length=GlobalConfig.ROLLOUT_FRAGMENT_LENGTH
                )
                .training(
                    model={"fcnet_hiddens": hidden_layers},
                    gamma=gamma,
                    lr=lr,
                    train_batch_size=GlobalConfig.TRAIN_BATCH_SIZE,
                    target_network_update_freq=target_freq,
                    double_q=DQNHyperparams.DOUBLE_Q,
                    dueling=DQNHyperparams.DUELING,
                    replay_buffer_config={"type": "ReplayBuffer", "capacity": DQNHyperparams.REPLAY_BUFFER_CAPACITY},
                    num_steps_sampled_before_learning_starts=0
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

        elif self.algo_name in ["ppo_lstm", "ppo-lstm"]:
            lr = self.lstm_hp.LR if self.lstm_hp else PPOLSTMConfig.LR
            gamma = self.lstm_hp.GAMMA if self.lstm_hp else PPOLSTMConfig.GAMMA
            hidden_layers = self.lstm_hp.MODEL_CONFIG["fcnet_hiddens"] if self.lstm_hp else PPOLSTMConfig.MODEL_CONFIG["fcnet_hiddens"]
            lstm_cell_size = getattr(self.lstm_hp, "LSTM_CELL_SIZE", 128) if self.lstm_hp else 128
            max_seq_len = getattr(self.lstm_hp, "MAX_SEQ_LEN", 20) if self.lstm_hp else 20

            model_cfg = {
                "fcnet_hiddens": hidden_layers,
                "use_lstm": True,
                "lstm_cell_size": lstm_cell_size,
                "max_seq_len": max_seq_len
            }

            config = (
                PPOConfig()
                .environment(env_name, env_config=self.env_cfg)
                .framework("torch")
                .debugging(seed=GlobalConfig.RANDOM_SEED)
                .env_runners(
                    num_env_runners=GlobalConfig.NUM_WORKERS,
                    rollout_fragment_length=GlobalConfig.ROLLOUT_FRAGMENT_LENGTH
                )
                .training(
                    model=model_cfg,
                    gamma=gamma,
                    lr=lr,
                    train_batch_size=GlobalConfig.TRAIN_BATCH_SIZE,
                    sgd_minibatch_size=PPOLSTMConfig.SGD_MINIBATCH_SIZE,
                    num_sgd_iter=PPOLSTMConfig.NUM_SGD_ITER,
                    clip_param=PPOLSTMConfig.CLIP_PARAM,
                    vf_loss_coeff=PPOLSTMConfig.VF_LOSS_COEFF,
                    entropy_coeff=PPOLSTMConfig.ENTROPY_COEFF
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

        algo = config.build()
        print(f"[UNIFIED GPU WORKER] {self.algo_name.upper()} Algorithm built successfully for Scenario {self.scenario}...")

        best_objective = -99999.0
        best_ckpt_bytes = None
        start_time = time.time()
        last_eval = {"jsr": 0.0, "tracking_acc": 0.0, "power": 0.0, "objective": -99999.0, "sinr": 0.0}

        for i in range(1, GlobalConfig.MAX_ITERATIONS + 1):
            res = algo.train()
            ep_reward = res.get("episode_reward_mean", 0.0)

            # Run HPO 30-seed evaluation
            if i % self.eval_freq == 0 or i == GlobalConfig.MAX_ITERATIONS:
                last_eval = run_30seeds_eval(algo, self.algo_name, self.env_cfg, phase=1)
                elapsed = round(time.time() - start_time, 2)
                print(f"[UNIFIED GPU WORKER] Iter {i:4d}/{GlobalConfig.MAX_ITERATIONS} | Train Reward: {ep_reward:6.2f} | 30-Seed Eval JSR: {last_eval['jsr']:5.2f}% | Track: {last_eval['tracking_acc']:5.2f}% | Power: {last_eval['power']:.3f}W | Obj: {last_eval['objective']:.2f} ({elapsed}s)")

                # Track best checkpoint in memory
                if last_eval["objective"] > best_objective:
                    best_objective = last_eval["objective"]
                    tmp_best = f"/tmp/best_ckpt_{self.algo_name}_{self.scenario}"
                    if os.path.exists(tmp_best): shutil.rmtree(tmp_best)
                    os.makedirs(tmp_best, exist_ok=True)
                    algo.save(checkpoint_dir=tmp_best)
                    best_ckpt_bytes = {}
                    for root, dirs, files in os.walk(tmp_best):
                        for f in files:
                            fp_path = os.path.join(root, f)
                            rel_p = os.path.relpath(fp_path, tmp_best)
                            with open(fp_path, "rb") as f_obj:
                                best_ckpt_bytes[rel_p] = f_obj.read()

            row = {
                "iteration": i,
                "episode_reward_mean": ep_reward,
                "eval_jsr": last_eval["jsr"],
                "eval_tracking_acc": last_eval["tracking_acc"],
                "eval_power": last_eval["power"],
                "eval_objective": last_eval["objective"],
                "eval_sinr": last_eval["sinr"],
                "time_s": round(time.time() - start_time, 2)
            }
            self.progress_rows.append(row)

        # Save Final Checkpoint (Iteration 1000)
        tmp_final = f"/tmp/final_ckpt_{self.algo_name}_{self.scenario}"
        if os.path.exists(tmp_final): shutil.rmtree(tmp_final)
        os.makedirs(tmp_final, exist_ok=True)
        algo.save(checkpoint_dir=tmp_final)
        final_ckpt_bytes = {}
        for root, dirs, files in os.walk(tmp_final):
            for f in files:
                fp_path = os.path.join(root, f)
                rel_p = os.path.relpath(fp_path, tmp_final)
                with open(fp_path, "rb") as f_obj:
                    final_ckpt_bytes[rel_p] = f_obj.read()

        algo.stop()
        return {
            "final_ckpt_bytes": final_ckpt_bytes,
            "best_ckpt_bytes": best_ckpt_bytes,
            "best_objective": best_objective,
            "progress_rows": self.progress_rows
        }

def train_algorithm_remote(algo_name, scenario, output_dir, ppo_hp=None, dqn_hp=None, lstm_hp=None):
    print(f"\n==================================================")
    print(f" UNIFIED TEST TRAINER: {algo_name.upper()} | Scenario {scenario}")
    print(f" Output Directory: {output_dir}")
    print(f"==================================================\n")

    if scenario == "1-A":
        EnvConfig.NUM_NODES = 15; EnvConfig.NUM_UAVS = 1; EnvConfig.AREA_SIZE = 500.0; EnvConfig.W_COST = 0.03
    elif scenario == "1-B":
        EnvConfig.NUM_NODES = 15; EnvConfig.NUM_UAVS = 1; EnvConfig.AREA_SIZE = 500.0; EnvConfig.W_COST = 0.3
    elif scenario == "2-A":
        EnvConfig.NUM_NODES = 30; EnvConfig.NUM_UAVS = 2; EnvConfig.AREA_SIZE = 1000.0; EnvConfig.W_COST = 0.03
    elif scenario == "2-B":
        EnvConfig.NUM_NODES = 30; EnvConfig.NUM_UAVS = 2; EnvConfig.AREA_SIZE = 1000.0; EnvConfig.W_COST = 0.3

    env_cfg_dict = {
        "seed": GlobalConfig.RANDOM_SEED,
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

    trainer_actor = UnifiedClusterGPUTrainer.remote(
        algo_name=algo_name,
        scenario=scenario,
        output_dir=output_dir,
        env_cfg=env_cfg_dict,
        ppo_hp=ppo_hp,
        dqn_hp=dqn_hp,
        lstm_hp=lstm_hp,
        eval_freq=5
    )

    train_future = trainer_actor.train_on_gpu.remote()

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "progress.csv")
    fieldnames = ["iteration", "episode_reward_mean", "eval_jsr", "eval_tracking_acc", "eval_power", "eval_objective", "eval_sinr", "time_s"]

    # Poll live progress from Worker GPU to Head Node
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
    final_ckpt_bytes = res["final_ckpt_bytes"]
    best_ckpt_bytes = res["best_ckpt_bytes"]
    progress_rows = res["progress_rows"]

    # Write progress CSV
    if progress_rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(progress_rows)

    # Save Checkpoints to Head Node
    final_dir = os.path.join(output_dir, "checkpoint_001000")
    os.makedirs(final_dir, exist_ok=True)
    for rel_p, content in final_ckpt_bytes.items():
        dest = os.path.join(final_dir, rel_p)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as fp:
            fp.write(content)

    if best_ckpt_bytes:
        best_dir = os.path.join(output_dir, "best_checkpoint")
        os.makedirs(best_dir, exist_ok=True)
        for rel_p, content in best_ckpt_bytes.items():
            dest = os.path.join(best_dir, rel_p)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "wb") as fp:
                fp.write(content)

    last = progress_rows[-1] if progress_rows else {}
    print(f"\n[OK] {algo_name.upper()} Training Completed!")
    print(f" Final Iter 1000 JSR:     {last.get('eval_jsr', 0.0):.2f}%")
    print(f" Final Iter 1000 Track:   {last.get('eval_tracking_acc', 0.0):.2f}%")
    print(f" Best Eval Objective:     {res['best_objective']:.2f}")
    print(f" Checkpoints Saved To:    {final_dir}\n")

def main():
    parser = argparse.ArgumentParser(description="Unified Scenario Test Trainer (HPO Pipeline Parity)")
    parser.add_argument("--scenario", type=str, default="1-A", choices=["1-A", "1-B", "2-A", "2-B"], help="Scenario ID")
    parser.add_argument("--algo", type=str, default="all", choices=["ppo", "dqn", "ppo_lstm", "all"], help="Algorithm to train")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    base_out = args.output_dir or os.path.join(PROJECT_ROOT, "artifacts", "scenario-unified-test", timestamp)

    algos = ["ppo", "dqn", "ppo_lstm"] if args.algo == "all" else [args.algo]
    for algo_name in algos:
        algo_dir = os.path.join(base_out, f"S{args.scenario}", algo_name)
        train_algorithm_remote(algo_name, args.scenario, algo_dir)

if __name__ == "__main__":
    main()
