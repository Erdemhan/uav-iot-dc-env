import re

with open("RAPOR.md", "r", encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        if line.startswith("#"):
            print(f"{i}: {line.strip()}")
