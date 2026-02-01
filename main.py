from tez_reporter import TezReporter
from logger import SimulationLogger
from environment import UAV_IoT_Env
from visualization import Visualization
from visualizer import SimulationVisualizer
import matplotlib.pyplot as plt
import time

TezReporter("main.py", "Simülasyon Başlatıldı")

def main():
    # 1. Logger Başlat
    # Config'i environment içinden veya config.py'den alabiliriz. 
    # Logger'a kaydetmek istediğimiz ekstra parametreleri verebiliriz.
    logger = SimulationLogger(config_dict={"Simulation": "Test Run v1"})
    
    # 2. Ortamı Başlat
    env = UAV_IoT_Env(logger=logger)
    
    # 3. Görselleştirmeyi Başlat
    viz = Visualization()
    
    # 4. Simülasyon Döngüsü
    obs, info = env.reset()
    
    try:
        for _ in range(100):
            # Rastgele bir aksiyon seç (Saldırgan Gücü)
            action = env.action_space.sample()
            
            # Adım İlerle
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Ekrana Çiz
            viz.render(env)
            
            if terminated or truncated:
                break
                
    except KeyboardInterrupt:
        print("Kullanıcı tarafından durduruldu.")
    finally:
        # 5. Kapanış
        logger.close()
        plt.close()
        print("Simülasyon tamamlandı. Analiz başlatılıyor...")
        
        # 6. Otomatik Analiz ve Görselleştirme
        try:
            analysis_viz = SimulationVisualizer(exp_dir=logger.log_dir)
            analysis_viz.generate_report()
        except Exception as e:
            print(f"Otomatik analiz hatası: {e}")

if __name__ == "__main__":
    main()
