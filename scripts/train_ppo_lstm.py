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

from simulation.pettingzoo_env import UAV_IoT_PZ_Env

def env_creator(config):
    from confs.model_config import GlobalConfig
    return ParallelPettingZooEnv(UAV_IoT_PZ_Env(auto_uav=True, flatten_actions=GlobalConfig.FLATTEN_ACTIONS))

class ProgressCallback(tune.Callback):
    """Callback to print progress in a format run_experiments.py can parse"""
    def on_trial_result(self, iteration, trials, trial, result, **info):
        print(f"Iteration {result['training_iteration']}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="ray_results", help="Output directory for PPO-LSTM")
    args = parser.parse_args()
    
    ray.init()
    
    # Reproducibility
    import torch
    import numpy as np
    from confs.model_config import GlobalConfig, PPOLSTMConfig
    
    torch.manual_seed(GlobalConfig.RANDOM_SEED)
    np.random.seed(GlobalConfig.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(GlobalConfig.RANDOM_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    register_env("uav_iot_ppo_lstm_v1", env_creator)
    
    # Define Policies
    dummy_env = UAV_IoT_PZ_Env(auto_uav=True, flatten_actions=GlobalConfig.FLATTEN_ACTIONS)
    obs_space = dummy_env.observation_space("jammer_0")
    act_space = dummy_env.action_space("jammer_0")
    node_obs = dummy_env.observation_space("node_0")
    node_act = dummy_env.action_space("node_0")
    
    config = (
        PPOConfig()
        .environment("uav_iot_ppo_lstm_v1", env_config={"seed": GlobalConfig.RANDOM_SEED})
        .framework("torch")
        .debugging(seed=GlobalConfig.RANDOM_SEED)
        .env_runners(
            num_env_runners=PPOLSTMConfig.NUM_WORKERS, 
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
        .multi_agent(
            policies={
                "jammer_policy": (None, obs_space, act_space, {}),
                "node_policy": (None, node_obs, node_act, {}),
            },
            policy_mapping_fn=lambda agent_id, *args, **kwargs: 
                "jammer_policy" if agent_id == "jammer_0" else "node_policy",
            policies_to_train=["jammer_policy"],
        )
        # Use Old API Stack to match others
        .api_stack(enable_rl_module_and_learner=False,
                   enable_env_runner_and_connector_v2=False)
        .resources(num_gpus=1 if PPOLSTMConfig.USE_GPU else 0)
        .debugging(log_level="WARN")
    )
    
    analysis = tune.run(
        "PPO", # Still PPO algo, just configured with LSTM
        name="PPO_LSTM",
        config=config.to_dict(),
        stop={"training_iteration": GlobalConfig.TRAIN_ITERATIONS}, 
        checkpoint_at_end=True,
        storage_path=os.path.abspath(args.output_dir),
        callbacks=[ProgressCallback()]
    )
    
    print("Training Completed.")
    ray.shutdown()
