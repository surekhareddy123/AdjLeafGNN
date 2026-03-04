from __future__ import annotations
import os
import json
from datetime import datetime

class Logger:
    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.log_path = os.path.join(out_dir, "logs.txt")

    def log(self, msg: str) -> None:
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{stamp}] {msg}"
        print(line)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def save_json(self, obj, filename: str) -> None:
        p = os.path.join(self.out_dir, filename)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)
