
import ray
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
    parser.add_argument("--algo", type=str, default="PPO", help="Algorithm (Baseline, PPO, or DQN)")
    parser.add_argument("--dir", type=str, default="./ray_results", help="Model Directory")
    parser.add_argument("--no-viz", action="store_true", help="Disable visualization")
    parser.add_argument("--output-dir", type=str, default="logs", help="Output directory for evaluation logs")
    args = parser.parse_args()

    # Reproducibility
    from confs.model_config import GlobalConfig
    import torch
    import numpy as np
    torch.manual_seed(GlobalConfig.RANDOM_SEED)
    np.random.seed(GlobalConfig.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(GlobalConfig.RANDOM_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # === BASELINE EVALUATION ===
    if args.algo.upper() == "BASELINE":
        print("="*70)
        print("Evaluating Baseline (QJC)")
        print("="*70)
        
        from simulation.controllers import UAVRuleBasedController
        
        # Setup logger
        full_config = {}
        full_config.update(UAVConfig.__dict__)
        full_config.update(EnvConfig.__dict__)
        logger = SimulationLogger(config_dict=full_config, log_dir=args.output_dir)
        
        # Create environment
        env = UAV_IoT_PZ_Env(logger=logger)
        uav_controller = UAVRuleBasedController(env)
        
        if not args.no_viz:
            viz = Visualization()
        
        # Reset
        observations, infos = env.reset(seed=GlobalConfig.RANDOM_SEED)
        uav_controller.reset()
        
        # Load trained Q-table
        try:
            env.attacker.load_model(args.dir)
            print(f"[OK] Loaded Q-table from: {args.dir}")
            env.attacker.temp_xi = 0  # Greedy policy for evaluation
        except Exception as e:
            print(f"[WARNING] Could not load Q-table from {args.dir}: {e}")
            print("  Running with empty Q-table")
        
        print("Starting evaluation episode...")
        
        try:
            for step in range(EnvConfig.MAX_STEPS):
                actions = {}
                
                # UAV Action
                if 'uav_0' in env.agents:
                    actions['uav_0'] = uav_controller.get_action()
                
                # Jammer Action
                if 'jammer_0' in env.agents:
                    from confs.model_config import QJCConfig
                    jam_channel = env.attacker.select_channel_qjc()
                    jam_power_level = QJCConfig.MAX_POWER_LEVEL  # Use max power (0-9 scale)
                    actions['jammer_0'] = np.array([jam_channel, jam_power_level])
                
                # Step environment
                observations, rewards, terminations, truncations, infos = env.step(actions)
                
                # Visualize
                if not args.no_viz:
                    viz.update(env)
                
                # Check if done
                if all(terminations.values()) or all(truncations.values()):
                    break
        
        except KeyboardInterrupt:
            print("\nEvaluation interrupted by user.")
        finally:
            logger.close()
            if not args.no_viz:
                viz.close()
        
        print(f"\n[OK] Baseline evaluation complete. Logs saved to: {args.output_dir}")
        return

    # === PPO/DQN EVALUATION ===
    ray.init() 
    register_env("uav_iot_ppo_v1", env_creator_ppo)
    register_env("uav_iot_dqn_v1", env_creator_dqn)
    register_env("uav_iot_ppo_lstm_v1", env_creator_ppo) # Same creator works
    
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
    logger = SimulationLogger(config_dict=full_config, log_dir=args.output_dir)
    
    # Auto UAV mode for compatibility
    # Use GlobalConfig to ensure consistency with training
    from confs.model_config import GlobalConfig
    env = UAV_IoT_PZ_Env(logger=logger, auto_uav=True, flatten_actions=GlobalConfig.FLATTEN_ACTIONS)
    if not args.no_viz:
        viz = Visualization()
    
    # Reset with seed
    obs_dict, infos = env.reset(seed=GlobalConfig.RANDOM_SEED)
    env.uav_controller = env.uav_controller # Ensure controller exists (set in reset)
    
    # Initialize LSTM State if needed
    lstm_state = []
    if args.algo == "PPO-LSTM":
        from confs.model_config import PPOLSTMConfig
        cell_size = PPOLSTMConfig.LSTM_CELL_SIZE
        # Initial state: [num_layers * [h, c]] -> simplified for 1 layer
        lstm_state = [
            np.zeros(cell_size, dtype=np.float32), 
            np.zeros(cell_size, dtype=np.float32)
        ]
    
    print("Starting Evaluation Episode...")
    
    try:
        for _ in range(EnvConfig.MAX_STEPS):
            actions = {}
            
            # Jammer Action from RLLib Policy
            if 'jammer_0' in obs_dict:
                jam_obs = obs_dict['jammer_0']
                
                # Use compute_single_action directly (safest for both MLP and LSTM)
                # It handles state management automatically if we pass it correctly
                try:
                    display_state = lstm_state if args.algo == "PPO-LSTM" else []
                    
                    # Returns: action, state_out, info
                    result = algo.compute_single_action(
                        observation=jam_obs,
                        state=display_state,
                        policy_id="jammer_policy",
                        explore=False
                    )
                    
                    if isinstance(result, tuple) and len(result) >= 1:
                         # Unpack based on return signature
                         # For LSTM: action, state_out, info
                         # For MLP: action (sometimes just action?) -> verification needed
                         # RLLib compute_single_action returns (action, state_out, info) IF state is provided or full_fetch=True
                         # If state is [], it might just return action.
                         
                         if args.algo == "PPO-LSTM":
                             jam_action = result[0]
                             lstm_state = result[1]
                         else:
                             # For MLP, it usually returns just action unless full_fetch=True
                             # But let's handle the tuple case if it returns (action, [], {})
                             if isinstance(result, tuple):
                                 jam_action = result[0]
                             else:
                                 jam_action = result
                    else:
                        jam_action = result

                except Exception as e:
                    print(f"Error in inference: {e}")
                    # Fallback
                    jam_action = 0 
            
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
