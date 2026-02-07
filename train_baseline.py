
from core.logger import SimulationLogger
from simulation.pettingzoo_env import UAV_IoT_PZ_Env
from simulation.controllers import UAVRuleBasedController
from visualization.visualization import Visualization
import numpy as np
from confs.config import UAVConfig
from confs.env_config import EnvConfig
from confs.model_config import QJCConfig
import os

def main():
    print("--- Starting Baseline (QJC) Training ---")
    
    # Reproducibility
    from confs.model_config import GlobalConfig
    np.random.seed(GlobalConfig.RANDOM_SEED)
    
    # Configuration
    # PPO trains for TRAIN_ITERATIONS * TRAIN_BATCH_SIZE steps.
    # Ex: 20 * 1000 = 20,000 steps.
    # Baseline Episode = 100 steps.
    # To match samples: 20,000 / 100 = 200 Episodes.
    TRAIN_EPISODES = QJCConfig.TRAIN_EPISODES
    SAVE_PATH = QJCConfig.SAVE_PATH
    
    # Disable Logger for training speed (Optional, or keep minimal)
    # We use a dummy logger or None
    env = UAV_IoT_PZ_Env(logger=None, auto_uav=True)
    
    # We need to maintain the Q-Table across resets
    learned_q_table = None
    learned_counts = None
    
    # Check if we want to continue training or start fresh
    # For fairness, let's start fresh or load if exists? 
    # Let's start fresh to ensure exactly N episodes of training.
    if os.path.exists(SAVE_PATH):
        import shutil
        shutil.rmtree(SAVE_PATH)
        print("Cleared previous Q-Table for fair fresh training.")

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
        
        if (episode + 1) % 50 == 0:
            print(f"Episode {episode+1}/{TRAIN_EPISODES} - Total Reward: {total_reward:.2f} - Q-Table: {np.round(learned_q_table, 2)}")

    # Save Final Model
    # Since env.attacker has the latest state
    env.attacker.save_model(SAVE_PATH)
    print(f"Baseline Training Completed. Q-Table saved to {SAVE_PATH}")

if __name__ == "__main__":
    main()
