
from core.logger import SimulationLogger
from simulation.environment import UAV_IoT_Env
from visualization.visualization import Visualization
from visualization.visualizer import SimulationVisualizer
import matplotlib.pyplot as plt
import time



def main():
    # 1. Start Logger
    # Can get Config from environment or config.py. 
    # Can pass extra parameters to Logger.
    logger = SimulationLogger(config_dict={"Simulation": "Test Run v1"})
    
    # 2. Start Environment
    env = UAV_IoT_Env(logger=logger)
    
    # 3. Start Visualization
    viz = Visualization()
    
    # 4. Simulation Loop
    obs, info = env.reset()
    
    print("Simulation Started...")
    
    try:
        for _ in range(100):
            # Select Random Action (Attacker Power)
            action = env.action_space.sample()
            
            # Step
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Render
            viz.render(env)
            
            if terminated or truncated:
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
        except Exception as e:
            print(f"Automated Analysis Error: {e}")

if __name__ == "__main__":
    main()
