
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.logger import SimulationLogger
from simulation.pettingzoo_env import UAV_IoT_PZ_Env
from simulation.controllers import UAVRuleBasedController
from visualization.visualization import Visualization
import numpy as np
from confs.config import UAVConfig
from confs.env_config import EnvConfig
from confs.model_config import QJCConfig
import os
import argparse

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="baseline_q_table", 
                       help="Directory to save baseline outputs")
    args = parser.parse_args()
    
    print(f"--- Starting Baseline (QJC) Training ---")
    print(f"Output Directory: {args.output_dir}\n")
    
    # Reproducibility
    from confs.model_config import GlobalConfig
    np.random.seed(GlobalConfig.RANDOM_SEED)
    
    # Configuration
    # Training episodes automatically calculated to match PPO/DQN total steps:
    # Formula: (TRAIN_ITERATIONS × TRAIN_BATCH_SIZE) / MAX_STEPS
    # This is defined in QJCConfig and auto-adjusts with config changes
    TRAIN_EPISODES = QJCConfig.TRAIN_EPISODES
    SAVE_PATH = args.output_dir
    
    # Disable Logger for training speed (Optional, or keep minimal)
    # We use a dummy logger or None
    env = UAV_IoT_PZ_Env(logger=None, auto_uav=True)
    
    # We need to maintain the Q-Table across resets
    learned_q_table = None
    learned_counts = None
    
    # Check if we want to continue training or start fresh
    # For fairness, let's start fresh to ensure exactly N episodes of training.
    if os.path.exists(SAVE_PATH):
        import shutil
        shutil.rmtree(SAVE_PATH)
        print("Cleared previous Q-Table for fair fresh training.")
    
    os.makedirs(SAVE_PATH, exist_ok=True)
    
    # Create CSV for training curve
    import csv
    training_log_path = os.path.join(SAVE_PATH, "training_curve.csv")
    training_log = open(training_log_path, 'w', newline='')
    csv_writer = csv.writer(training_log)
    csv_writer.writerow(['episode', 'total_reward', 'mean_reward'])

    for episode in range(TRAIN_EPISODES):
        obs, infos = env.reset()
        
        # INJECT LEARNED KNOWLEDGE
        # Since reset() creates a new SmartAttacker, we must overwrite its brain
        if learned_q_table is not None:
             env.attacker.q_table = learned_q_table
             env.attacker.channel_counts = learned_counts
        
        terminated = False
        truncations = False
        
        step_count = 0
        total_reward = 0
        
        while not (terminated or truncations):
            actions = {}
            
            # Jammer Action (QJC)
            # 1. Select
            selected_channel = env.attacker.select_channel_qjc()
            selected_power_level = QJCConfig.MAX_POWER_LEVEL  # Max power
            
            actions['jammer_0'] = np.array([selected_channel, selected_power_level])
            
            # Nodes
            for agent in env.agents:
                if "node" in agent:
                    actions[agent] = 0
            
            # Step
            obs, rewards, term, trunc, infos = env.step(actions)
            
            # 2. Update Q-Table
            r_jam = rewards['jammer_0']
            ch_idx = env.attacker.current_channel
            env.attacker.update_qjc(ch_idx, r_jam)
            
            total_reward += r_jam
            
            terminated = any(term.values())
            truncations = any(trunc.values())
            step_count += 1
            
        # End of Episode
        # EXTRACT KNOWLEDGE
        learned_q_table = env.attacker.q_table.copy()
        learned_counts = env.attacker.channel_counts.copy()
        
        # Log to CSV
        mean_reward = total_reward / step_count if step_count > 0 else 0
        csv_writer.writerow([episode + 1, total_reward, mean_reward])
        
        # Dynamic logging interval (at least every episode for short runs, or 10% for long runs)
        log_interval = max(1, TRAIN_EPISODES // 10)
        
        if (episode + 1) % log_interval == 0:
            print(f"Episode {episode+1}/{TRAIN_EPISODES} - Total Reward: {total_reward:.2f} - Q-Table: {np.round(learned_q_table, 2)}")

    # Close training log
    training_log.close()
    
    # Save Final Model
    # Since env.attacker has the latest state
    env.attacker.save_model(SAVE_PATH)
    print(f"Baseline Training Completed. Q-Table saved to {SAVE_PATH}")
    print(f"Training curve saved to {training_log_path}")

if __name__ == "__main__":
    main()
