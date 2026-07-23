import os
import sys
import argparse
import json
import time
import csv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

os.environ["RAY_DISABLE_METRICS_EXPORT"] = "1"
os.environ["RAY_OVERRIDE_JOB_RUNTIME_ENV"] = "1"

import ray
from confs.model_config import GlobalConfig, PPOConfig, DQNConfig, PPOLSTMConfig
from confs.env_config import EnvConfig
from scripts.tune_models import train_rllib_trial, run_30seeds_eval, env_creator
from ray.tune.registry import register_env

@ray.remote(num_gpus=1, max_concurrency=2)
class HPOTrainableActor:
    """Remote actor wrapping HPO's exact execution with live progress reporting to Head Node."""
    def __init__(self, config):
        self.config = config
        self.progress_rows = []
        self.final_result = None

    def get_progress(self):
        return list(self.progress_rows)

    def run_training(self):
        import torch
        import random
        import numpy as np

        torch.set_num_threads(2)
        random.seed(GlobalConfig.RANDOM_SEED)
        torch.manual_seed(GlobalConfig.RANDOM_SEED)
        np.random.seed(GlobalConfig.RANDOM_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(GlobalConfig.RANDOM_SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # Execute exact train_rllib_trial logic step-by-step to record live progress
        register_env("uav_iot_ppo_v1", env_creator)
        algo_name = self.config["algo"]
        lr = self.config["lr"]
        gamma = self.config["gamma"]
        architecture = self.config.get("architecture", None)
        if isinstance(architecture, str):
            fcnet_hiddens = [int(x) for x in architecture.split(",")]
        else:
            fcnet_hiddens = list(architecture)

        from simulation.pettingzoo_env import UAV_IoT_PZ_Env
        dummy_env = UAV_IoT_PZ_Env(auto_uav=True, flatten_actions=GlobalConfig.FLATTEN_ACTIONS)
        obs_space = dummy_env.observation_space("jammer_0")
        act_space = dummy_env.action_space("jammer_0")
        node_obs = dummy_env.observation_space("node_0")
        node_act = dummy_env.action_space("node_0")

        if algo_name == "PPO":
            from ray.rllib.algorithms.ppo import PPOConfig as RLlibPPOConfig
            cfg_obj = RLlibPPOConfig()
            model_cfg = {"fcnet_hiddens": fcnet_hiddens}
        elif algo_name == "PPO-LSTM":
            from ray.rllib.algorithms.ppo import PPOConfig as RLlibPPOConfig
            cfg_obj = RLlibPPOConfig()
            model_cfg = {
                "fcnet_hiddens": fcnet_hiddens,
                "use_lstm": True,
                "lstm_cell_size": self.config.get("lstm_cell_size", 256),
                "max_seq_len": self.config.get("max_seq_len", 20)
            }
        elif algo_name == "DQN":
            from ray.rllib.algorithms.dqn import DQNConfig as RLlibDQNConfig
            cfg_obj = RLlibDQNConfig()
            model_cfg = {"fcnet_hiddens": fcnet_hiddens}

        cfg_obj = (
            cfg_obj
            .environment("uav_iot_ppo_v1", env_config={"seed": GlobalConfig.RANDOM_SEED})
            .framework("torch")
            .debugging(seed=GlobalConfig.RANDOM_SEED)
            .env_runners(
                num_env_runners=self.config["num_workers"],
                rollout_fragment_length=100
            )
            .training(
                model=model_cfg,
                train_batch_size=1000,
                lr=lr,
                gamma=gamma,
                **(
                    {
                        "target_network_update_freq": self.config.get("target_network_update_freq", 500),
                        "double_q": True,
                        "dueling": True,
                        "replay_buffer_config": {"type": "ReplayBuffer", "capacity": 50000},
                        "num_steps_sampled_before_learning_starts": 1000
                    } if algo_name == "DQN" else {}
                )
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
            .resources(num_gpus=self.config["num_gpus"])
            .debugging(log_level="WARN")
        )

        algo = cfg_obj.build()
        start_time = time.time()

        iterations = self.config["iterations"]
        phase = self.config["phase"]
        last_eval = {"jsr": 0.0, "tracking_acc": 0.0, "power": 0.0, "objective": -99999.0, "sinr": 0.0}

        for i in range(1, iterations + 1):
            res = algo.train()
            ep_reward = res.get("episode_reward_mean", 0.0)

            if i % 5 == 0 or i == iterations:
                last_eval = run_30seeds_eval(
                    algo_agent=algo,
                    algo_name=algo_name,
                    env_config=self.config["env_config"],
                    phase=phase,
                    lstm_cell_size=self.config.get("lstm_cell_size", 256)
                )
                elapsed = round(time.time() - start_time, 2)
                print(f"[WORKER GPU] Iter {i:4d}/{iterations} | Reward: {ep_reward:6.2f} | 30-Seed JSR: {last_eval['jsr']:5.2f}% | Track: {last_eval['tracking_acc']:5.2f}% | Power: {last_eval['power']:.3f}W | Obj: {last_eval['objective']:.2f} ({elapsed}s)")

            self.progress_rows.append({
                "iteration": i,
                "episode_reward_mean": ep_reward,
                "eval_jsr": last_eval["jsr"],
                "eval_tracking_acc": last_eval["tracking_acc"],
                "eval_power": last_eval["power"],
                "eval_objective": last_eval["objective"],
                "eval_sinr": last_eval["sinr"],
                "time_s": round(time.time() - start_time, 2)
            })

        algo.stop()
        self.final_result = last_eval
        return {
            "result": last_eval,
            "progress_rows": self.progress_rows
        }

def main():
    parser = argparse.ArgumentParser(description="Unified Test Trainer directly executing HPO's exact train_rllib_trial code with live progress polling.")
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
            "env_vars": {"PYTHONPATH": "."}
        }
        ray.init(ignore_reinit_error=True, runtime_env=runtime_env)

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join(PROJECT_ROOT, "artifacts", "scenario-unified-test", timestamp, f"S{args.scenario}", algo_upper.lower())
    os.makedirs(out_dir, exist_ok=True)
    csv_file = os.path.join(out_dir, "progress.csv")
    fieldnames = ["iteration", "episode_reward_mean", "eval_jsr", "eval_tracking_acc", "eval_power", "eval_objective", "eval_sinr", "time_s"]

    # Instantiate Actor on Worker GPU
    print("Dispatching HPO remote actor to Worker GPU...")
    actor = HPOTrainableActor.remote(trial_config)
    train_future = actor.run_training.remote()

    print(f"Live Progress Polling Active! Saving progress CSV to: {csv_file}")
    last_printed_iter = 0

    while True:
        ready, _ = ray.wait([train_future], timeout=3.0)
        try:
            rows = ray.get(actor.get_progress.remote())
            if rows:
                with open(csv_file, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)

                latest = rows[-1]
                cur_iter = latest["iteration"]
                if cur_iter > last_printed_iter and (cur_iter % 5 == 0 or cur_iter == args.iterations):
                    print(f"  Iter {cur_iter:4d}/{args.iterations} | Train Reward: {latest['episode_reward_mean']:6.2f} | 30-Seed Eval JSR: {latest['eval_jsr']:5.2f}% | Track: {latest['eval_tracking_acc']:5.2f}% | Power: {latest['eval_power']:.3f}W ({latest['time_s']}s)")
                    last_printed_iter = cur_iter
        except Exception:
            pass

        if ready:
            break

    res_data = ray.get(train_future)
    res = res_data["result"]
    progress_rows = res_data["progress_rows"]

    results_file = os.path.join(out_dir, "results.json")
    config_file = os.path.join(out_dir, "trial_config.json")

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)

    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(trial_config, f, indent=2)

    if progress_rows:
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(progress_rows)

    print(f"\n==================================================")
    print(f" DIRECT HPO EXECUTION COMPLETED")
    print(f" Algorithm: {algo_upper} | Scenario: {args.scenario}")
    print(f" Result: {json.dumps(res, indent=2)}")
    print(f" Saved Results To: {results_file}")
    print(f" Saved Config To:  {config_file}")
    print(f" Saved Progress To:{csv_file}")
    print(f"==================================================\n")

if __name__ == "__main__":
    main()
