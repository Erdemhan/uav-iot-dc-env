import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings
import argparse

# Suppress Ray metrics exporter warnings and allow job environment override
os.environ["RAY_DISABLE_METRICS_EXPORT"] = "1"
os.environ["RAY_OVERRIDE_JOB_RUNTIME_ENV"] = "1"

# Filter specific warnings to clean up output
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")
warnings.filterwarnings("ignore", category=UserWarning, module="pettingzoo")

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env

from simulation.pettingzoo_env import UAV_IoT_PZ_Env
from confs.model_config import GlobalConfig, PPOConfig as PPOHyperparams, DQNConfig as DQNHyperparams, PPOLSTMConfig
from confs.env_config import EnvConfig

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

@ray.remote(num_gpus=1)
class ClusterGPUTrainer:
    """Ray Remote Actor guaranteed to run on a Worker PC with an active GPU."""
    def __init__(self, algo_name, scenario, output_dir, env_cfg, ppo_hp=None, dqn_hp=None, lstm_hp=None):
        self.algo_name = algo_name
        self.scenario = scenario
        self.output_dir = output_dir
        self.env_cfg = env_cfg
        self.ppo_hp = ppo_hp
        self.dqn_hp = dqn_hp
        self.lstm_hp = lstm_hp

    def train_on_gpu(self):
        import torch
        import numpy as np
        import random

        torch.set_num_threads(2)
        random.seed(GlobalConfig.RANDOM_SEED)
        torch.manual_seed(GlobalConfig.RANDOM_SEED)
        np.random.seed(GlobalConfig.RANDOM_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(GlobalConfig.RANDOM_SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        env_name = f"uav_iot_{self.algo_name}_gpu_v1"
        try:
            register_env(env_name, env_creator)
        except Exception:
            pass

        dummy_env = UAV_IoT_PZ_Env(auto_uav=True, flatten_actions=GlobalConfig.FLATTEN_ACTIONS)
        obs_space = dummy_env.observation_space("jammer_0")
        act_space = dummy_env.action_space("jammer_0")
        node_obs = dummy_env.observation_space("node_0")
        node_act = dummy_env.action_space("node_0")

        if self.algo_name == "ppo":
            config = (
                PPOConfig()
                .environment(env_name, env_config=self.env_cfg)
                .framework("torch")
                .debugging(seed=GlobalConfig.RANDOM_SEED)
                .env_runners(
                    num_env_runners=GlobalConfig.NUM_WORKERS, 
                    rollout_fragment_length=self.ppo_hp["ROLLOUT_FRAGMENT_LENGTH"]
                )
                .training(
                    model={"fcnet_hiddens": self.ppo_hp["FCNET_HIDDENS"]},
                    train_batch_size=self.ppo_hp["TRAIN_BATCH_SIZE"],
                    lr=self.ppo_hp["LR"],
                    gamma=self.ppo_hp["GAMMA"],
                )
            )
        elif self.algo_name == "dqn":
            config = (
                DQNConfig()
                .environment(env_name, env_config=self.env_cfg)
                .framework("torch")
                .debugging(seed=GlobalConfig.RANDOM_SEED)
                .env_runners(
                    num_env_runners=GlobalConfig.NUM_WORKERS,
                    rollout_fragment_length=self.dqn_hp["ROLLOUT_FRAGMENT_LENGTH"]
                ) 
                .training(
                    model={"fcnet_hiddens": self.dqn_hp["FCNET_HIDDENS"]},
                    gamma=self.dqn_hp["GAMMA"],
                    lr=self.dqn_hp["LR"],
                    train_batch_size=self.dqn_hp["TRAIN_BATCH_SIZE"],
                    target_network_update_freq=self.dqn_hp["TARGET_NETWORK_UPDATE_FREQ"],
                    double_q=self.dqn_hp["DOUBLE_Q"],
                    dueling=self.dqn_hp["DUELING"],
                    replay_buffer_config={
                        "type": "ReplayBuffer",
                        "capacity": self.dqn_hp["REPLAY_BUFFER_CAPACITY"],
                    },
                    num_steps_sampled_before_learning_starts=self.dqn_hp["NUM_STEPS_SAMPLED_BEFORE_LEARNING_STARTS"],
                    training_intensity=self.dqn_hp["TRAINING_INTENSITY"],
                )
            )
        elif self.algo_name == "ppo_lstm":
            config = (
                PPOConfig()
                .environment(env_name, env_config=self.env_cfg)
                .framework("torch")
                .debugging(seed=GlobalConfig.RANDOM_SEED)
                .env_runners(
                    num_env_runners=GlobalConfig.NUM_WORKERS, 
                    rollout_fragment_length=self.lstm_hp["ROLLOUT_FRAGMENT_LENGTH"]
                )
                .training(
                    model={
                        "fcnet_hiddens": self.lstm_hp["FCNET_HIDDENS"],
                        "use_lstm": self.lstm_hp["USE_LSTM"],
                        "lstm_cell_size": self.lstm_hp["LSTM_CELL_SIZE"],
                        "max_seq_len": self.lstm_hp["MAX_SEQ_LEN"],
                    },
                    train_batch_size=self.lstm_hp["TRAIN_BATCH_SIZE"],
                    lr=self.lstm_hp["LR"],
                    gamma=self.lstm_hp["GAMMA"],
                )
            )

        config = (
            config
            .multi_agent(
                policies={
                    "jammer_policy": (None, obs_space, act_space, {}),
                    "node_policy": (None, node_obs, node_act, {}),
                },
                policy_mapping_fn=lambda agent_id, *args, **kwargs: 
                    "jammer_policy" if agent_id == "jammer_0" else "node_policy",
                policies_to_train=["jammer_policy"],
            )
            .api_stack(enable_rl_module_and_learner=False,
                       enable_env_runner_and_connector_v2=False)
            .resources(num_gpus=1)
            .debugging(log_level="WARN")
        )
        if self.algo_name == "dqn":
            config = config.experimental(_validate_config=False)

        # Build algorithm directly on Worker GPU!
        algo = config.build()
        print(f"[Worker GPU] Successfully initialized {self.algo_name.upper()} on Worker GPU for scenario {self.scenario} ({GlobalConfig.TRAIN_ITERATIONS} iterations)...")

        best_reward = -float('inf')
        no_improvement_count = 0

        for i in range(1, GlobalConfig.TRAIN_ITERATIONS + 1):
            result = algo.train()
            reward = result.get("episode_reward_mean") or result.get("env_runners/episode_reward_mean")
            reward_str = f"{reward:.2f}" if reward is not None else "N/A"
            print(f"Iteration {i}/{GlobalConfig.TRAIN_ITERATIONS} - Episode Reward Mean: {reward_str}")

            if reward is not None:
                if reward > best_reward:
                    best_reward = reward
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

                if no_improvement_count >= GlobalConfig.EARLY_STOPPING_PATIENCE and reward >= GlobalConfig.EARLY_STOPPING_MIN_REWARD:
                    print(f"\n[Early Stopping] No improvement in reward for {GlobalConfig.EARLY_STOPPING_PATIENCE} iterations. Stopping training.")
                    break

        ckpt_dir = os.path.abspath(os.path.join(self.output_dir, "checkpoint_001000"))
        os.makedirs(ckpt_dir, exist_ok=True)
        print(f"Saving final {self.algo_name.upper()} GPU model checkpoint to {ckpt_dir}...")
        
        # Use formal RLlib Old API Stack checkpoint method directly on Worker PC
        if hasattr(algo, "save_checkpoint"):
            algo.save_checkpoint(ckpt_dir)
        else:
            algo.save(checkpoint_dir=ckpt_dir)
            
        print(f"Successfully saved GPU checkpoint to {ckpt_dir}")
        algo.stop()
        return True

if __name__ == "__main__":
    from core.logger import setup_console_logging
    
    parser = argparse.ArgumentParser(description="Unified GPU Training Script for UAV-IoT Environment")
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "dqn", "ppo_lstm"], help="Algorithm to train (ppo, dqn, ppo_lstm)")
    parser.add_argument("--scenario", type=str, default="1-A", choices=["1-A", "1-B", "2-A", "2-B"], help="Scenario combination to run")
    parser.add_argument("--output-dir", type=str, default="ray_results", help="Output directory for checkpoints and logs")
    parser.add_argument("--ray-address", type=str, default=None, help="Ray cluster address (e.g. 'auto' or IP address)")
    args = parser.parse_args()

    setup_console_logging(f"train_{args.algo}")

    # Ensure output_dir is an absolute path for remote workers
    abs_output_dir = os.path.abspath(args.output_dir)

    # Apply scenario config dynamically
    if args.scenario == "1-A":
        EnvConfig.NUM_NODES = 15; EnvConfig.NUM_UAVS = 1; EnvConfig.AREA_SIZE = 500.0; EnvConfig.W_COST = 0.03
    elif args.scenario == "1-B":
        EnvConfig.NUM_NODES = 15; EnvConfig.NUM_UAVS = 1; EnvConfig.AREA_SIZE = 500.0; EnvConfig.W_COST = 0.3
    elif args.scenario == "2-A":
        EnvConfig.NUM_NODES = 30; EnvConfig.NUM_UAVS = 2; EnvConfig.AREA_SIZE = 1000.0; EnvConfig.W_COST = 0.03
    elif args.scenario == "2-B":
        EnvConfig.NUM_NODES = 30; EnvConfig.NUM_UAVS = 2; EnvConfig.AREA_SIZE = 1000.0; EnvConfig.W_COST = 0.3

    if not ray.is_initialized():
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        runtime_env = {
            "working_dir": project_root,
            "excludes": [".venv", "artifacts", "ray_results", "head-node-logs", ".git", "baseline_q_table", "node_modules"],
            "env_vars": {"PYTHONPATH": "."}
        }
        
        init_kwargs = {
            "ignore_reinit_error": True,
            "runtime_env": runtime_env
        }
        if args.ray_address:
            init_kwargs["address"] = args.ray_address
            
        try:
            ray.init(**init_kwargs)
        except Exception:
            pass

    register_env(f"uav_iot_{args.algo}_v1", env_creator)

    env_cfg = {
        "seed": GlobalConfig.RANDOM_SEED,
        "num_nodes": int(EnvConfig.NUM_NODES),
        "num_uavs": int(EnvConfig.NUM_UAVS),
        "area_size": float(EnvConfig.AREA_SIZE),
        "w_cost": float(EnvConfig.W_COST)
    }

    ppo_hp = {
        "ROLLOUT_FRAGMENT_LENGTH": PPOHyperparams.ROLLOUT_FRAGMENT_LENGTH,
        "FCNET_HIDDENS": PPOHyperparams.FCNET_HIDDENS,
        "TRAIN_BATCH_SIZE": PPOHyperparams.TRAIN_BATCH_SIZE,
        "LR": PPOHyperparams.LR,
        "GAMMA": PPOHyperparams.GAMMA,
    }

    dqn_hp = {
        "ROLLOUT_FRAGMENT_LENGTH": DQNHyperparams.ROLLOUT_FRAGMENT_LENGTH,
        "FCNET_HIDDENS": DQNHyperparams.FCNET_HIDDENS,
        "GAMMA": DQNHyperparams.GAMMA,
        "LR": DQNHyperparams.LR,
        "TRAIN_BATCH_SIZE": DQNHyperparams.TRAIN_BATCH_SIZE,
        "TARGET_NETWORK_UPDATE_FREQ": DQNHyperparams.TARGET_NETWORK_UPDATE_FREQ,
        "DOUBLE_Q": DQNHyperparams.DOUBLE_Q,
        "DUELING": DQNHyperparams.DUELING,
        "REPLAY_BUFFER_CAPACITY": DQNHyperparams.REPLAY_BUFFER_CAPACITY,
        "NUM_STEPS_SAMPLED_BEFORE_LEARNING_STARTS": DQNHyperparams.NUM_STEPS_SAMPLED_BEFORE_LEARNING_STARTS,
        "TRAINING_INTENSITY": DQNHyperparams.TRAINING_INTENSITY,
    }

    lstm_hp = {
        "ROLLOUT_FRAGMENT_LENGTH": PPOLSTMConfig.ROLLOUT_FRAGMENT_LENGTH,
        "FCNET_HIDDENS": PPOLSTMConfig.FCNET_HIDDENS,
        "USE_LSTM": PPOLSTMConfig.USE_LSTM,
        "LSTM_CELL_SIZE": PPOLSTMConfig.LSTM_CELL_SIZE,
        "MAX_SEQ_LEN": PPOLSTMConfig.MAX_SEQ_LEN,
        "TRAIN_BATCH_SIZE": PPOLSTMConfig.TRAIN_BATCH_SIZE,
        "LR": PPOLSTMConfig.LR,
        "GAMMA": PPOLSTMConfig.GAMMA,
    }

    # Instantiate ClusterGPUTrainer on a Worker PC with 1 GPU!
    gpu_trainer = ClusterGPUTrainer.remote(
        algo_name=args.algo,
        scenario=args.scenario,
        output_dir=abs_output_dir,
        env_cfg=env_cfg,
        ppo_hp=ppo_hp,
        dqn_hp=dqn_hp,
        lstm_hp=lstm_hp
    )

    print(f"Dispatched {args.algo.upper()} GPU training task to a Worker PC in the Ray Cluster...")
    success = ray.get(gpu_trainer.train_on_gpu.remote())
    print(f"{args.algo.upper()} GPU Training Completed Successfully: {success}")

    if ray.is_initialized():
        ray.shutdown()
