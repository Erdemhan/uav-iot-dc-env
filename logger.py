import os
import json
import datetime
import pandas as pd
from typing import Dict, Any, List
from tez_reporter import TezReporter

class SimulationLogger:
    """
    Simülasyon verilerini (Telemetry) kaydeder.
    """
    def __init__(self, config_dict: Dict[str, Any] = None):
        TezReporter("logger.py", "Logger Başlatıldı")
        
        # Log klasörünü oluştur: logs/EXP_YYYYMMDD_HHMMSS
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join("logs", f"EXP_{timestamp}")
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.csv_path = os.path.join(self.log_dir, "history.csv")
        self.buffer: List[Dict[str, Any]] = []
        
        if config_dict:
            self.save_config(config_dict)

    def save_config(self, config: Dict[str, Any]):
        """Config parametrelerini json olarak kaydeder."""
        path = os.path.join(self.log_dir, "config.json")
        # Parametreler içinde serialize edilemeyen nesneler varsa stringe çeviriyoruz
        safe_config = {k: str(v) for k, v in config.items()}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(safe_config, f, indent=4)

    def log_step(self, data_dict: Dict[str, Any]):
        """Her adımın verisini belleğe alır."""
        self.buffer.append(data_dict)

    def flush(self):
        """Bellekteki veriyi diske yazar."""
        if not self.buffer:
            return
            
        df = pd.DataFrame(self.buffer)
        
        # Dosya yoksa header ile yaz, varsa header olmadan ekle
        mode = 'w' if not os.path.exists(self.csv_path) else 'a'
        header = not os.path.exists(self.csv_path)
        
        df.to_csv(self.csv_path, mode=mode, header=header, index=False)
        self.buffer = [] # Bufferı temizle

    def close(self):
        """Kapanışta kalan verileri yaz."""
        self.flush()
        print(f"[SimulationLogger] Kayıtlar {self.log_dir} dizinine kaydedildi.")
