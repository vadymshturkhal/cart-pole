import json, os
from threading import Lock


class ConfigManager:
    """Central manager for all persistent and runtime configuration."""
    _instance = None
    _lock = Lock()

    @classmethod
    def instance(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def __init__(self):
        main_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.path = os.path.join(main_dir, "user_config.json")
        self.data = self._load()

    # ------------------- IO -------------------
    def _load(self):
        if not os.path.exists(self.path):
            return {}
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=4)

    # ------------------- Panel helpers -------------------
    def get_section(self, name: str, fallback: dict):
        """Return a dict for a specific panel/section."""
        key = f"{name}_last_config"
        if key in self.data:
            return self.data[key]
        defaults_key = f"{name}_defaults"
        if defaults_key in self.data:
            return self.data[defaults_key]
        return fallback.copy()

    def set_section_runtime(self, name: str, values: dict):
        """Save temporary runtime settings for a panel."""
        self.data[f"{name}_last_config"] = values
        self.save()

    def set_section_defaults(self, name: str, values: dict):
        """Persist defaults for a panel."""
        self.data[f"{name}_defaults"] = values
        self.save()

    def clear_runtime_sections(self):
        """Clean up runtime-only entries for all panels."""
        to_remove = [k for k in self.data if k.endswith("_last_config")]
        for k in to_remove:
            del self.data[k]
        self.save()
