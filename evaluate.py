
import ray
import os
import warnings
# Filter specific warnings to clean up output
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")
warnings.filterwarnings("ignore", category=UserWarning, module="pettingzoo")
import glob
from ray.rllib.algorithms.ppo import PPOConfig
from ray import tune
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

from simulation.pettingzoo_env import UAV_IoT_PZ_Env
from visualization.visualization import Visualization
from visualization.visualizer import SimulationVisualizer
import matplotlib.pyplot as plt
import time
import numpy as np
import torch
from confs.config import UAVConfig
from confs.env_config import EnvConfig
from core.logger import SimulationLogger

import argparse

def env_creator_ppo(config):
    from confs.model_config import GlobalConfig
    return ParallelPettingZooEnv(UAV_IoT_PZ_Env(auto_uav=True, flatten_actions=GlobalConfig.FLATTEN_ACTIONS))

def env_creator_dqn(config):
    from confs.model_config import GlobalConfig
    return ParallelPettingZooEnv(UAV_IoT_PZ_Env(auto_uav=True, flatten_actions=GlobalConfig.FLATTEN_ACTIONS))

def find_latest_checkpoint(base_dir="./ray_results"):
    # Find all checkpoint folders recursively
    # Structure could be: base_dir / ExpName / TrialName / checkpoint_000000
    search_pattern = os.path.join(base_dir, "**", "checkpoint_*")
    ckpt_dirs = glob.glob(search_pattern, recursive=True)
    
    if not ckpt_dirs:
        return None
        
    # Filter out files (if any match) - glob might match files if they verify pattern
    ckpt_dirs = [d for d in ckpt_dirs if os.path.isdir(d)]
    
    if not ckpt_dirs:
        return None
        
    # Sort by mtime to get the latest
    latest_ckpt = max(ckpt_dirs, key=os.path.getmtime)
    return latest_ckpt

def main():
    parser = argparse.ArgumentParser(description="Evaluate Trained Model")
    parser.add_argument("--algo", type=str, default="PPO", help="Algorithm (PPO or DQN)")
    parser.add_argument("--dir", type=str, default="./ray_results", help="Checkpoint Directory")
    parser.add_argument("--no-viz", action="store_true", help="Disable visualization")
    args = parser.parse_args()

    ray.init() 
    register_env("uav_iot_ppo_v1", env_creator_ppo)
    register_env("uav_iot_dqn_v1", env_creator_dqn)
    
    checkpoint_path = find_latest_checkpoint(base_dir=args.dir)
    
    if not checkpoint_path:
        print(f"No checkpoint found in {args.dir}. Run training first.")
        return

    print(f"Loading Checkpoint: {checkpoint_path}")
    checkpoint_path = os.path.abspath(checkpoint_path)
    
    # Re-create config matches train.py (Simplified)
    try:
        algo = Algorithm.from_checkpoint(checkpoint_path)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    # Setup Simulation for Inference
    full_config = {}
    full_config.update(UAVConfig.__dict__)
    full_config.update(EnvConfig.__dict__)
    logger = SimulationLogger(config_dict=full_config)
    
    # Auto UAV mode for compatibility
    # Use GlobalConfig to ensure consistency with training
    from confs.model_config import GlobalConfig
    env = UAV_IoT_PZ_Env(logger=logger, auto_uav=True, flatten_actions=GlobalConfig.FLATTEN_ACTIONS)
    if not args.no_viz:
        viz = Visualization()
    
    obs_dict, infos = env.reset()
    env.uav_controller = env.uav_controller # Ensure controller exists (set in reset)
    
    print("Starting Evaluation Episode...")
    
    try:
        for _ in range(EnvConfig.MAX_STEPS):
            actions = {}
            
            # Jammer Action from RLLib Policy
            # Env is PettingZoo, but we have raw obs dict
            # We need to compute action for 'jammer_0'
            if 'jammer_0' in obs_dict: # If not done
                jam_obs = obs_dict['jammer_0']
                # Hybrid Inference (Supports both New API Stack and Old API Stack)
                try:
                    # 1. Try New API Stack (RLModule)
                    module = algo.get_module("jammer_policy")
                    # jam_obs is (obs_dim,). We need batch (1, obs_dim) and Tensor.
                    obs_tensor = torch.from_numpy(jam_obs).float().unsqueeze(0)
                    
                    with torch.no_grad():
                        # forward_inference returns dict with "action_dist_inputs"
                        fwd_out = module.forward_inference({"obs": obs_tensor})
                        logits = fwd_out["action_dist_inputs"] # Shape [1, 13] (3+10)
                        
                        # Split logits for MultiDiscrete([3, 10])
                        logits_ch = logits[:, :3]
                        logits_pow = logits[:, 3:]
                        
                        # Deterministic Action (Argmax) for Evaluation
                        ch_idx = torch.argmax(logits_ch, dim=1).item()
                        pow_idx = torch.argmax(logits_pow, dim=1).item()
                        
                        jam_action = np.array([ch_idx, pow_idx])
                        
                except Exception:
                    # 2. Fallback to Old API Stack (compute_single_action)
                    # This works for DQN Old Stack
                    jam_action = algo.compute_single_action(
                        observation=jam_obs,
                        policy_id="jammer_policy",
                        explore=False
                    )
            
            # CRITICAL FIX: Add jammer action to actions dict!
            actions['jammer_0'] = jam_action
            
            # Nodes
            for agent in env.agents:
                if "node" in agent:
                    actions[agent] = 0

            # Step
            obs_dict, rewards, terminations, truncations, infos = env.step(actions)
            
            if not args.no_viz:
                viz.render(env)
                time.sleep(0.05) # Fast replay
            
            if any(terminations.values()) or any(truncations.values()):
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        logger.close()
        # plt.close()
        ray.shutdown()
        
        print("Evaluation Finished. Generating Report...")
        analysis_viz = SimulationVisualizer(exp_dir=logger.log_dir)
        analysis_viz.generate_report()
        analysis_viz.show_dashboard(show=not args.no_viz)

if __name__ == "__main__":
    main()
