import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings

# Suppress Ray metrics exporter warnings
os.environ["RAY_DISABLE_METRICS_EXPORT"] = "1"

# Filter specific warnings to clean up output
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")
warnings.filterwarnings("ignore", category=UserWarning, module="pettingzoo")
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env
from ray.rllib.policy.policy import Policy

from simulation.pettingzoo_env import UAV_IoT_PZ_Env

def env_creator(config):
    from confs.model_config import GlobalConfig
    # This ensures identical gradient flow and exploration structure
    return ParallelPettingZooEnv(UAV_IoT_PZ_Env(auto_uav=True, flatten_actions=GlobalConfig.FLATTEN_ACTIONS))

from ray.tune import Callback
from ray.tune.stopper import Stopper

class EarlyStoppingStopper(Stopper):
    """Custom stopper to stop training when average reward stops improving"""
    def __init__(self, patience=None, min_reward=None):
        from confs.model_config import GlobalConfig
        self.patience = patience if patience is not None else GlobalConfig.EARLY_STOPPING_PATIENCE
        self.min_reward = min_reward if min_reward is not None else GlobalConfig.EARLY_STOPPING_MIN_REWARD
        self.best_reward = -float('inf')
        self.no_improvement_count = 0

    def __call__(self, trial_id, result):
        # Stop if reached max iterations
        from confs.model_config import GlobalConfig
        iteration = result.get("training_iteration", 0)
        if iteration >= GlobalConfig.TRAIN_ITERATIONS:
            return True
            
        reward = result.get("env_runners/episode_reward_mean") or result.get("episode_reward_mean")
        if reward is None:
            return False
            
        if reward > self.best_reward:
            self.best_reward = reward
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
            
        if self.no_improvement_count >= self.patience and reward >= self.min_reward:
            print(f"\n[Early Stopping] No improvement in reward for {self.patience} iterations. Stopping trial.")
            return True
        return False

    def stop_all(self):
        return False

class ProgressCallback(Callback):
    """Callback to print progress in a format run_experiments.py can parse"""
    def on_trial_result(self, iteration, trials, trial, result, **info):
        print(f"Iteration {result['training_iteration']}")

if __name__ == "__main__":
    from core.logger import setup_console_logging
    setup_console_logging("train_ppo")
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="ray_results", help="Output directory for PPO")
    args = parser.parse_args()
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    runtime_env = {"env_vars": {"PYTHONPATH": project_root}}
    try:
        ray.init(num_gpus=1, ignore_reinit_error=True, runtime_env=runtime_env)
    except Exception:
        ray.init(num_gpus=1, ignore_reinit_error=True, runtime_env=runtime_env)
    
    # Reproducibility
    import torch
    import numpy as np
    import random
    from confs.model_config import GlobalConfig
    
    torch.set_num_threads(2)
    
    random.seed(GlobalConfig.RANDOM_SEED)
    torch.manual_seed(GlobalConfig.RANDOM_SEED)
    np.random.seed(GlobalConfig.RANDOM_SEED)
    # CUDA determinism
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(GlobalConfig.RANDOM_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    register_env("uav_iot_ppo_v1", env_creator)
    
    # Define Policies
    # CRITICAL: Must match flatten_actions setting used in env_creator
    from confs.model_config import GlobalConfig, PPOConfig as PPOHyperparams
    dummy_env = UAV_IoT_PZ_Env(auto_uav=True, flatten_actions=GlobalConfig.FLATTEN_ACTIONS)
    obs_space = dummy_env.observation_space("jammer_0")
    act_space = dummy_env.action_space("jammer_0")
    node_obs = dummy_env.observation_space("node_0")
    node_act = dummy_env.action_space("node_0")
    
    config = (
        PPOConfig()
        .environment("uav_iot_ppo_v1", env_config={"seed": GlobalConfig.RANDOM_SEED})
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
        .multi_agent(
            policies={
                "jammer_policy": (None, obs_space, act_space, {}),
                "node_policy": (None, node_obs, node_act, {}),
            },
            policy_mapping_fn=lambda agent_id, *args, **kwargs: 
                "jammer_policy" if agent_id == "jammer_0" else "node_policy",
            policies_to_train=["jammer_policy"],
        )
        # FAIRNESS: Use Old API Stack to match DQN's API for fair comparison
        .api_stack(enable_rl_module_and_learner=False,
                   enable_env_runner_and_connector_v2=False)
        .resources(num_gpus=1 if GlobalConfig.USE_GPU else 0)
        .debugging(log_level="WARN")
    )
    
    stopper = EarlyStoppingStopper()
    
    analysis = tune.run(
        "PPO",
        config=config.to_dict(),
        stop=stopper, 
        checkpoint_at_end=True,
        checkpoint_freq=GlobalConfig.CHECKPOINT_FREQ,
        keep_checkpoints_num=GlobalConfig.KEEP_CHECKPOINTS_NUM,
        checkpoint_score_attr=GlobalConfig.CHECKPOINT_SCORE_ATTR,
        storage_path=os.path.abspath(args.output_dir),
        trial_dirname_creator=lambda trial: f"t_{trial.trial_id}",
        callbacks=[ProgressCallback()]
    )
    
    print("Training Completed.")
    ray.shutdown()
