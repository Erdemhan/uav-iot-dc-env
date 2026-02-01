import json
import os
import datetime
from typing import Dict, Any

class TezReporter:
    """
    Tez önerisi kapsamında yapılan sistemsel değişiklikleri proje tarihçesine kaydeder.
    """
    HISTORY_FILE = "project_history.json"

    def __init__(self, module_name: str, action: str = "Modül Oluşturuldu"):
        self.module_name = module_name
        self.log_entry(action)

    def log_entry(self, action: str, details: Dict[str, Any] = None):
        """
        Geçmiş dosyasına yeni bir kayıt ekler.
        """
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "module": self.module_name,
            "action": action,
            "details": details or {}
        }

        history = []
        if os.path.exists(self.HISTORY_FILE):
            try:
                with open(self.HISTORY_FILE, "r", encoding="utf-8") as f:
                    history = json.load(f)
            except json.JSONDecodeError:
                pass # Dosya bozuksa veya boşsa yeni liste başlat

        history.append(entry)

        with open(self.HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=4, ensure_ascii=False)
            
if __name__ == "__main__":
    # Test
    TezReporter("tez_reporter.py", "Test Kaydı")
