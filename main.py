
from core.logger import SimulationLogger
from simulation.pettingzoo_env import UAV_IoT_PZ_Env
from simulation.controllers import UAVRuleBasedController
from visualization.visualization import Visualization
from visualization.visualizer import SimulationVisualizer
import matplotlib.pyplot as plt
import time
import numpy as np
from confs.config import UAVConfig
from confs.env_config import EnvConfig
from confs.model_config import GlobalConfig
import torch

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-viz", action="store_true", help="Disable visualization")
    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(GlobalConfig.RANDOM_SEED)
    np.random.seed(GlobalConfig.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(GlobalConfig.RANDOM_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # 1. Start Logger
    full_config = {}
    full_config.update(UAVConfig.__dict__)
    full_config.update(EnvConfig.__dict__)
    
    logger = SimulationLogger(config_dict=full_config)
    
    # 2. Start Environment (PettingZoo)
    env = UAV_IoT_PZ_Env(logger=logger)
    
    # 3. Start Controller
    uav_controller = UAVRuleBasedController(env)
    
    # 3. Start Visualization
    if not args.no_viz:
        viz = Visualization()
    
    # 4. Simulation Loop
    observations, infos = env.reset(seed=GlobalConfig.RANDOM_SEED)
    uav_controller.reset() # Reset controller state
    
    # LOAD TRAINED BASELINE MODEL (For Evaluation)
    try:
        env.attacker.load_model("baseline_q_table")
    except:
        print("No trained Baseline model found. Running with empty Q-Table (Online Learning).")
    
    print("Simulation Started (PettingZoo Multi-Agent)...")
    
    try:
        # Loop for MAX_STEPS from config
        for _ in range(EnvConfig.MAX_STEPS):
            actions = {}
            
            # --- UAV Action (Rule Based) ---
            if 'uav_0' in env.agents:
                uav_action = uav_controller.get_action() 
                actions['uav_0'] = uav_action
            
            # --- Jammer Action (QJC - Baseline Algorithm) ---
            if 'jammer_0' in env.agents:
                # 1. Select Channel using QJC (Softmax)
                # We access the entity directly via env.attacker for the baseline algorithm
                # Note: In a pure gym interface, this logic would be inside the Environment or an Agent class wrapper.
                # Here we simulate the "Agent" part in the main loop for the Baseline.
                
                selected_channel = env.attacker.select_channel_qjc()
                
                # 2. Select Power (Assume Max for strong attack or random 0-10)
                # Liao et al. focuses on channel. Let's fix power to max (Level 9) or high random.
                # To be comparable, let's pick max power to stress test the UAV.
                selected_power_level = 9 # Max power
                
                # Action: [Channel, PowerLevel]
                actions['jammer_0'] = np.array([selected_channel, selected_power_level])
                
                # Note: Update Q-Table happens inside the loop after Reward is received?
                # Actually, Q-Update needs Reward. 
                # We need to call update_qjc AFTER step.
            
            # --- Node Actions (Passive) ---
            for agent_id in env.agents:
                if agent_id.startswith('node_'):
                    # No-Op
                     actions[agent_id] = 0

            # Step
            observations, rewards, terminations, truncations, infos = env.step(actions)
            
            # --- QJC Update (Baseline) ---
            if 'jammer_0' in env.agents:
                # Get Reward
                r_jam = rewards['jammer_0']
                # Get Action taken (we know it from above)
                # ch_idx from env.attacker.current_channel is reliable
                ch_idx = env.attacker.current_channel
                
                # Update Q-Table
                env.attacker.update_qjc(ch_idx, r_jam)
            
            # Render
            if not args.no_viz:
                viz.render(env)
                # Wait a bit
                time.sleep(UAVConfig.SIMULATION_DELAY) 
            
            # Check global termination
            if any(terminations.values()) or any(truncations.values()):
                break
                
    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        # 5. Finish
        logger.close()
        # plt.close() # Only if viz exists, handled by main close usually
        print("Simulation Completed. Analysis starting...")
        
        # 6. Automated Analysis and Visualization
        try:
            analysis_viz = SimulationVisualizer(exp_dir=logger.log_dir)
            analysis_viz.generate_report()
            # Show Dashboard
            analysis_viz.show_dashboard(show=not args.no_viz)
        except Exception as e:
            print(f"Automated Analysis Error: {e}")

if __name__ == "__main__":
    main()
