#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
from datetime import datetime
from typing import List, Dict, Optional, Any


class PathRegistry:
    def __init__(self, json_path: str) -> None:
        self.json_path = json_path
        self.data: Dict[str, Any] = {"history": []}
        self.load()

    def load(self) -> None:
        if os.path.exists(self.json_path):
            try:
                with open(self.json_path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
                    print(f"[PathRegistry] Loaded {len(self.data.get('history', []))} entries from {self.json_path}")
            except Exception as e:
                print(f"[PathRegistry] Error loading {self.json_path}: {e}")
                self.data = {"history": []}
        else:
            print(f"[PathRegistry] Registry file not found, initializing new: {self.json_path}")
            self.data = {"history": []}

    def save(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.json_path), exist_ok=True)
            with open(self.json_path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
            print(f"[PathRegistry] Saved registry to {self.json_path}")
        except Exception as e:
            print(f"[PathRegistry] Error saving {self.json_path}: {e}")

    def get_path_seed(self, source_id: str, target_id: str, action: str) -> Optional[str]:
        # ロギング追加
        # print(f"[PathRegistry] Query: src={source_id}, tgt={target_id}, action={action}")
        for entry in reversed(self.data.get("history", [])):
            if (entry.get("source_id") == source_id and 
                entry.get("target_id") == target_id and 
                entry.get("action") == action):
                res = entry.get("path_seed")
                print(f"[PathRegistry] Match found! -> {res}")
                return res
        return None

    def update(self, source_id: str, target_id: str, action: str, path_seed: str) -> None:
        print(f"[PathRegistry] Updating: src={source_id}, tgt={target_id}, action={action}, path={path_seed}")
        found = False
        if "history" not in self.data:
            self.data["history"] = []
            
        for entry in self.data["history"]:
            if (entry.get("source_id") == source_id and 
                entry.get("target_id") == target_id and 
                entry.get("action") == action):
                entry["path_seed"] = path_seed
                entry["last_used"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                entry["success_count"] = entry.get("success_count", 0) + 1
                found = True
                break
        
        if not found:
            new_entry = {
                "source_id": source_id,
                "target_id": target_id,
                "action": action,
                "path_seed": path_seed,
                "success_count": 1,
                "last_used": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            self.data["history"].append(new_entry)
        
        self.save()
