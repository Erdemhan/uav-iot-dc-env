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
from ray.rllib.algorithms.dqn import DQNConfig
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

from simulation.pettingzoo_env import UAV_IoT_PZ_Env
from confs.model_config import DQNConfig as DQNHyperparams, GlobalConfig

def env_creator(config):
    # Enable internal UAV controller for training
    # DQN requires Discrete action space, so we flatten MultiDiscrete([3, 10]) -> Discrete(30)
    env = UAV_IoT_PZ_Env(auto_uav=True, flatten_actions=GlobalConfig.FLATTEN_ACTIONS)
    return ParallelPettingZooEnv(env)

if __name__ == "__main__":
    if ray.is_initialized():
        ray.shutdown()
    ray.init() 
    
    # Reproducibility
    import torch
    import numpy as np
    torch.manual_seed(GlobalConfig.RANDOM_SEED)
    np.random.seed(GlobalConfig.RANDOM_SEED)
    # CUDA determinism
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(GlobalConfig.RANDOM_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    register_env("uav_iot_dqn_v1", env_creator)
    
    print("Starting DQN Training (Ray RLLib with Official DQN Patch)...")
    
    # Clean up previous results
    storage_path = os.path.abspath("./ray_results_dqn")
    if os.path.exists(storage_path):
        import shutil
        shutil.rmtree(storage_path, ignore_errors=True)
    
    # Get spaces for multi-agent config
    dummy_env = UAV_IoT_PZ_Env(auto_uav=True, flatten_actions=GlobalConfig.FLATTEN_ACTIONS)
    obs_space = dummy_env.observation_space("jammer_0")
    act_space = dummy_env.action_space("jammer_0")
    node_obs = dummy_env.observation_space("node_0")
    node_act = dummy_env.action_space("node_0")

    config = (
        DQNConfig()
        .environment("uav_iot_dqn_v1", env_config={"seed": GlobalConfig.RANDOM_SEED})
        .framework("torch")
        .debugging(seed=GlobalConfig.RANDOM_SEED)
        # FAIRNESS: Match PPO's worker count for equal sampling
        .env_runners(num_env_runners=DQNHyperparams.NUM_WORKERS) 
        .training(
            model={"fcnet_hiddens": DQNHyperparams.FCNET_HIDDENS},
            gamma=DQNHyperparams.GAMMA,
            lr=DQNHyperparams.LR,
            train_batch_size=DQNHyperparams.TRAIN_BATCH_SIZE,
            target_network_update_freq=DQNHyperparams.TARGET_NETWORK_UPDATE_FREQ,
            double_q=DQNHyperparams.DOUBLE_Q,
            dueling=DQNHyperparams.DUELING,
            replay_buffer_config={
                "type": "MultiAgentReplayBuffer",
                "capacity": DQNHyperparams.REPLAY_BUFFER_CAPACITY,
            },
            num_steps_sampled_before_learning_starts=DQNHyperparams.NUM_STEPS_SAMPLED_BEFORE_LEARNING_STARTS,
            training_intensity=DQNHyperparams.TRAINING_INTENSITY,
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
        # CRITICAL: DQN requires Old API Stack even with flattened actions
        # New API Stack has unresolved compatibility issues with DQN
        .api_stack(enable_rl_module_and_learner=False,
                   enable_env_runner_and_connector_v2=False)
        # FAIRNESS: Match PPO's GPU usage
        .resources(num_gpus=1 if DQNHyperparams.USE_GPU else 0)
        .debugging(log_level="WARN")
        .experimental(_validate_config=False)
    )

    # Run Training
    print(f"Iterations: {GlobalConfig.TRAIN_ITERATIONS}")
    
    analysis = tune.run(
        "DQN", 
        name="DQN_EXPERIMENT",
        config=config.to_dict(),
        stop={"training_iteration": GlobalConfig.TRAIN_ITERATIONS}, 
        checkpoint_at_end=True,
        storage_path=storage_path
    )
    
    print("DQN Training Completed.")
    ray.shutdown()
