import os
import json
import datetime
import pandas as pd
from typing import Dict, Any, List


class SimulationLogger:
    """
    Records simulation data (Telemetry).
    """
    def __init__(self, config_dict: Dict[str, Any] = None, log_dir: str = None):

        # Create log folder: logs/EXP_YYYYMMDD_HHMMSS or use custom directory
        if log_dir is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_dir = os.path.join("logs", f"EXP_{timestamp}")
        else:
            self.log_dir = log_dir
            
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.csv_path = os.path.join(self.log_dir, "history.csv")
        self.buffer: List[Dict[str, Any]] = []
        
        if config_dict:
            self.save_config(config_dict)

    def save_config(self, config: Dict[str, Any]):
        """Saves config parameters as json."""
        path = os.path.join(self.log_dir, "config.json")
        # Convert non-serializable objects in parameters to string
        safe_config = {k: str(v) for k, v in config.items()}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(safe_config, f, indent=4)

    def log_step(self, data_dict: Dict[str, Any]):
        """Buffers data for each step."""
        self.buffer.append(data_dict)

    def flush(self):
        """Writes buffered data to disk."""
        if not self.buffer:
            return
            
        df = pd.DataFrame(self.buffer)
        
        # If file doesn't exist write with header, else append without header
        mode = 'w' if not os.path.exists(self.csv_path) else 'a'
        header = not os.path.exists(self.csv_path)
        
        df.to_csv(self.csv_path, mode=mode, header=header, index=False)
        self.buffer = [] # Clear buffer

    def close(self):
        """Write remaining data on close."""
        self.flush()
        print(f"[SimulationLogger] Logs saved to {self.log_dir}.")
