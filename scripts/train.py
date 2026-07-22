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
from confs.model_config import GlobalConfig, PPOConfig as PPOHyperparams, DQNHyperparams, PPOLSTMConfig
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

if __name__ == "__main__":
    from core.logger import setup_console_logging
    
    parser = argparse.ArgumentParser(description="Unified Parametric RL Training Script for UAV-IoT Environment")
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "dqn", "ppo_lstm"], help="Algorithm to train (ppo, dqn, ppo_lstm)")
    parser.add_argument("--scenario", type=str, default="1-A", choices=["1-A", "1-B", "2-A", "2-B"], help="Scenario combination to run")
    parser.add_argument("--output-dir", type=str, default="ray_results", help="Output directory for checkpoints and logs")
    parser.add_argument("--ray-address", type=str, default=None, help="Ray cluster address (e.g. 'auto' or IP address)")
    args = parser.parse_args()

    setup_console_logging(f"train_{args.algo}")

    # Apply scenario config dynamically on driver node before dummy env is created
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

    # Reproducibility
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

    env_name = f"uav_iot_{args.algo}_v1"
    register_env(env_name, env_creator)

    # Define Policies
    dummy_env = UAV_IoT_PZ_Env(auto_uav=True, flatten_actions=GlobalConfig.FLATTEN_ACTIONS)
    obs_space = dummy_env.observation_space("jammer_0")
    act_space = dummy_env.action_space("jammer_0")
    node_obs = dummy_env.observation_space("node_0")
    node_act = dummy_env.action_space("node_0")

    env_cfg = {
        "seed": GlobalConfig.RANDOM_SEED,
        "num_nodes": int(EnvConfig.NUM_NODES),
        "num_uavs": int(EnvConfig.NUM_UAVS),
        "area_size": float(EnvConfig.AREA_SIZE),
        "w_cost": float(EnvConfig.W_COST)
    }

    # Check CUDA availability safely to prevent Error 302 on machines without working CUDA libraries
    use_gpu = GlobalConfig.USE_GPU and torch.cuda.is_available()
    try:
        if use_gpu:
            _ = torch.cuda.device_count()
    except Exception:
        use_gpu = False

    # Configure algorithm dynamically based on --algo parameter
    if args.algo == "ppo":
        config = (
            PPOConfig()
            .environment(env_name, env_config=env_cfg)
            .framework("torch")
            .debugging(seed=GlobalConfig.RANDOM_SEED)
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
        )
    elif args.algo == "dqn":
        config = (
            DQNConfig()
            .environment(env_name, env_config=env_cfg)
            .framework("torch")
            .debugging(seed=GlobalConfig.RANDOM_SEED)
            .env_runners(
                num_env_runners=GlobalConfig.NUM_WORKERS,
                rollout_fragment_length=DQNHyperparams.ROLLOUT_FRAGMENT_LENGTH
            ) 
            .training(
                model={"fcnet_hiddens": DQNHyperparams.FCNET_HIDDENS},
                gamma=DQNHyperparams.GAMMA,
                lr=DQNHyperparams.LR,
                train_batch_size=DQNHyperparams.TRAIN_BATCH_SIZE,
                target_network_update_freq=DQNHyperparams.TARGET_NETWORK_UPDATE_FREQ,
                double_q=DQNHyperparams.DOUBLE_Q,
                dueling=DQNHyperparams.DUELING,
                replay_buffer_config={
                    "type": "ReplayBuffer",
                    "capacity": DQNHyperparams.REPLAY_BUFFER_CAPACITY,
                },
                num_steps_sampled_before_learning_starts=DQNHyperparams.NUM_STEPS_SAMPLED_BEFORE_LEARNING_STARTS,
                training_intensity=DQNHyperparams.TRAINING_INTENSITY,
            )
        )
    elif args.algo == "ppo_lstm":
        config = (
            PPOConfig()
            .environment(env_name, env_config=env_cfg)
            .framework("torch")
            .debugging(seed=GlobalConfig.RANDOM_SEED)
            .env_runners(
                num_env_runners=GlobalConfig.NUM_WORKERS, 
                rollout_fragment_length=PPOLSTMConfig.ROLLOUT_FRAGMENT_LENGTH
            )
            .training(
                model={
                    "fcnet_hiddens": PPOLSTMConfig.FCNET_HIDDENS,
                    "use_lstm": PPOLSTMConfig.USE_LSTM,
                    "lstm_cell_size": PPOLSTMConfig.LSTM_CELL_SIZE,
                    "max_seq_len": PPOLSTMConfig.MAX_SEQ_LEN,
                },
                train_batch_size=PPOLSTMConfig.TRAIN_BATCH_SIZE,
                lr=PPOLSTMConfig.LR,
                gamma=PPOLSTMConfig.GAMMA,
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
        .resources(num_gpus=1 if use_gpu else 0)
        .debugging(log_level="WARN")
    )
    if args.algo == "dqn":
        config = config.experimental(_validate_config=False)

    # Unified Direct RLlib Training Loop
    algo = config.build()
    print(f"Starting {args.algo.upper()} training for scenario {args.scenario} ({GlobalConfig.TRAIN_ITERATIONS} iterations)...")

    best_reward = -float('inf')
    no_improvement_count = 0

    for i in range(1, GlobalConfig.TRAIN_ITERATIONS + 1):
        result = algo.train()
        reward = result.get("env_runners/episode_reward_mean") or result.get("episode_reward_mean")
        reward_str = f"{reward:.2f}" if reward is not None else "N/A"
        print(f"Iteration {i}/{GlobalConfig.TRAIN_ITERATIONS} - Episode Reward Mean: {reward_str}")

        # Early Stopping
        if reward is not None:
            if reward > best_reward:
                best_reward = reward
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if no_improvement_count >= GlobalConfig.EARLY_STOPPING_PATIENCE and reward >= GlobalConfig.EARLY_STOPPING_MIN_REWARD:
                print(f"\n[Early Stopping] No improvement in reward for {GlobalConfig.EARLY_STOPPING_PATIENCE} iterations. Stopping training.")
                break

    ckpt_dir = os.path.abspath(os.path.join(args.output_dir, "checkpoint_001000"))
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"Saving final {args.algo.upper()} model checkpoint to {ckpt_dir}...")
    if hasattr(algo, "save_to_path"):
        algo.save_to_path(ckpt_dir)
    else:
        algo.save(checkpoint_dir=ckpt_dir)
    print(f"Successfully saved checkpoint to {ckpt_dir}")

    algo.stop()
    print(f"{args.algo.upper()} Training Completed.")
    if ray.is_initialized():
        ray.shutdown()
