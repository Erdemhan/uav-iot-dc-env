
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

def main():
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
    viz = Visualization()
    
    # 4. Simulation Loop
    observations, infos = env.reset()
    uav_controller.reset() # Reset controller state
    
    print("Simulation Started (PettingZoo Multi-Agent)...")
    
    try:
        # Loop for MAX_STEPS from config
        for _ in range(EnvConfig.MAX_STEPS):
            actions = {}
            
            # --- UAV Action (Rule Based) ---
            if 'uav_0' in env.agents:
                uav_action = uav_controller.get_action() 
                actions['uav_0'] = uav_action
            
            # --- Jammer Action (Learning/Random) ---
            if 'jammer_0' in env.agents:
                # Sample from action space for now (Random Agent)
                actions['jammer_0'] = env.action_spaces['jammer_0'].sample()
            
            # --- Node Actions (Passive) ---
            for agent_id in env.agents:
                if agent_id.startswith('node_'):
                    # No-Op
                     actions[agent_id] = 0

            # Step
            observations, rewards, terminations, truncations, infos = env.step(actions)
            
            # Render
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
        plt.close()
        print("Simulation Completed. Analysis starting...")
        
        # 6. Automated Analysis and Visualization
        try:
            analysis_viz = SimulationVisualizer(exp_dir=logger.log_dir)
            analysis_viz.generate_report()
            # Show Dashboard
            analysis_viz.show_dashboard()
        except Exception as e:
            print(f"Automated Analysis Error: {e}")

if __name__ == "__main__":
    main()
