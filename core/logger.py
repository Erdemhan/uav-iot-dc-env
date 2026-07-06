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
        if os.path.exists(self.csv_path):
            try:
                os.remove(self.csv_path)
            except OSError:
                pass
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


class ProcessLogger:
    """
    Handles logging of subprocess stdout/stderr streams.
    Classifies lines and optionally writes them to a file.
    """
    def __init__(self, log_file_path: str = None):
        self.log_file_path = log_file_path
        if log_file_path:
            os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
            # Clear log file on startup
            try:
                with open(self.log_file_path, "w", encoding="utf-8") as f:
                    pass
            except Exception:
                pass
            
    def process_line(self, raw_line: str, stream_type: str = "stdout") -> tuple[str, str]:
        """
        Processes a raw line from a stream:
        Classifies it and writes it to a file if configured.
        Returns (prefix, clean_line).
        """
        clean_line = raw_line.strip()
        if not clean_line:
            return "", ""
            
        lower_line = clean_line.lower()
        
        if stream_type == "stderr":
            if any(k in lower_line for k in ["error", "fail", "exception", "critical", "traceback"]):
                prefix = "[ERROR]"
            elif any(k in lower_line for k in ["warning", "warn", "deprecation"]):
                prefix = "[WARN]"
            else:
                prefix = "[INFO]"
        else:
            # Stdout is usually info, unless it explicitly logs errors
            if any(k in lower_line for k in ["error", "exception", "critical", "traceback"]):
                prefix = "[ERROR]"
            elif any(k in lower_line for k in ["warning", "warn"]):
                prefix = "[WARN]"
            else:
                prefix = "[INFO]"
                
        formatted = f"{prefix} {clean_line}"
        
        if self.log_file_path:
            try:
                with open(self.log_file_path, "a", encoding="utf-8") as f:
                    f.write(formatted + "\n")
            except Exception:
                pass
                
        return prefix, clean_line

