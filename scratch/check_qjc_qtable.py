import numpy as np
import os
import sys

# Add project root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.pettingzoo_env import UAV_IoT_PZ_Env
from confs.model_config import QJCConfig
from confs.env_config import EnvConfig
from scripts.tune_models import run_30seeds_eval

def check_qjc():
    # Let's test with Trial 3 parameters
    tau_0 = 0.2977589608370269
    gamma = 0.8586553265614296
    temp_xi = 0.996973969547569
    mu_offset = 1.4767450920291654
    iterations = 5
    
    total_episodes = iterations * 10
    
    env = UAV_IoT_PZ_Env(logger=None, auto_uav=True)
    learned_q_table = None
    learned_counts = None
    
    print(f"Training QJC for {total_episodes} episodes...")
    for ep in range(1, total_episodes + 1):
        obs, infos = env.reset()
        if learned_q_table is not None:
            env.attacker.q_table = learned_q_table
            env.attacker.channel_counts = learned_counts
            
        env.attacker.tau_0 = tau_0
        env.attacker.gamma = gamma
        env.attacker.temp_xi = temp_xi
        env.attacker.mu_offset = mu_offset
        
        terminated = False
        truncations = False
        
        while not (terminated or truncations):
            actions = {}
            selected_channel = env.attacker.select_channel_qjc()
            selected_power_level = 9
            
            actions['jammer_0'] = np.array([selected_channel, selected_power_level])
            for agent in env.agents:
                if "node" in agent: actions[agent] = 0
                
            obs, rewards, term, trunc, infos = env.step(actions)
            r_jam = rewards['jammer_0']
            ch_idx = env.attacker.current_channel
            env.attacker.update_qjc(ch_idx, r_jam)
            
            terminated = any(term.values())
            truncations = any(trunc.values())
            
        learned_q_table = env.attacker.q_table.copy()
        learned_counts = env.attacker.channel_counts.copy()
        
        if ep % 10 == 0:
            print(f"Episode {ep}:")
            print("  Q-Table:", learned_q_table)
            print("  Counts:", learned_counts)
            
    print("\nFinal trained Q-Table:")
    print("Q-Table:", learned_q_table)
    print("Counts:", learned_counts)
    
    # Run evaluation with original trial parameters
    eval_metrics_trial = run_30seeds_eval(
        algo_agent=None,
        algo_name="Baseline",
        env_config={
            "W_SUCCESS": 0.8,
            "W_TRACKING": 0.2,
            "W_COST": 0.03
        },
        phase=1,
        q_table=learned_q_table,
        q_counts=learned_counts,
        q_params={
            "tau_0": tau_0,
            "gamma": gamma,
            "mu_offset": mu_offset
        }
    )
    print("\nEvaluation metrics with trial parameters:")
    print(eval_metrics_trial)

    # Run evaluation with a different tau_0 to see if values change
    eval_metrics_alt = run_30seeds_eval(
        algo_agent=None,
        algo_name="Baseline",
        env_config={
            "W_SUCCESS": 0.8,
            "W_TRACKING": 0.2,
            "W_COST": 0.03
        },
        phase=1,
        q_table=learned_q_table,
        q_counts=learned_counts,
        q_params={
            "tau_0": 0.5, # Much higher learning rate for fast adaptation during evaluation
            "gamma": gamma,
            "mu_offset": mu_offset
        }
    )
    print("\nEvaluation metrics with alternative parameters (tau_0 = 0.5):")
    print(eval_metrics_alt)

if __name__ == "__main__":
    check_qjc()
