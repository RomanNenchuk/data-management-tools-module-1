from dataclasses import dataclass
from typing import Any
import yaml

@dataclass
class Config:
    raw: dict

    @staticmethod
    def load(path: str) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return Config(raw=data)

    def get(self, *keys, default=None):
        cur: Any = self.raw
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur
