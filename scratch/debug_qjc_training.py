import numpy as np
import os
import sys

# Add project root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.pettingzoo_env import UAV_IoT_PZ_Env
from confs.model_config import QJCConfig
from confs.env_config import EnvConfig

def debug_training():
    env = UAV_IoT_PZ_Env(logger=None, auto_uav=True)
    
    # Use baseline hyperparameters from config
    tau_0 = 0.05
    gamma = 0.90
    temp_xi = 0.5
    mu_offset = 1.1
    
    learned_q_table = None
    learned_counts = None
    
    # Run 5 episodes and print stats
    for ep in range(1, 6):
        obs, infos = env.reset()
        if learned_q_table is not None:
            env.attacker.q_table = learned_q_table
            env.attacker.channel_counts = learned_counts
            
        env.attacker.tau_0 = tau_0
        env.attacker.gamma = gamma
        env.attacker.temp_xi = temp_xi
        env.attacker.mu_offset = mu_offset
        
        if ep == 1:
            print("--- Initial Q-Table & Counts ---")
            print("Q-Table:", env.attacker.q_table)
            print("Counts:", env.attacker.channel_counts)
            
        terminated = False
        truncations = False
        
        ep_reward = 0.0
        steps = 0
        channel_selections = []
        
        while not (terminated or truncations):
            selected_channel = env.attacker.select_channel_qjc()
            channel_selections.append(selected_channel)
            selected_power_level = 9
            
            actions = {'jammer_0': np.array([selected_channel, selected_power_level])}
            for agent in env.agents:
                if "node" in agent:
                    actions[agent] = 0
                    
            obs, rewards, term, trunc, infos = env.step(actions)
            r_jam = rewards['jammer_0']
            ep_reward += r_jam
            steps += 1
            
            # Update QJC
            env.attacker.update_qjc(selected_channel, r_jam)
            
            terminated = any(term.values())
            truncations = any(trunc.values())
            
        print(f"\n--- Episode {ep} Complete ---")
        print(f"Steps: {steps}, Total Reward: {ep_reward:.4f}")
        print("Channel Selections (first 20 steps):", channel_selections[:20])
        print("Q-Table:", env.attacker.q_table)
        print("Counts:", env.attacker.channel_counts)

if __name__ == "__main__":
    from core.logger import setup_console_logging
    setup_console_logging("debug_qjc_training")
    debug_training()
