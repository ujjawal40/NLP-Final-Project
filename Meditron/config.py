import os
from pathlib import Path
import yaml


class Config:
    def __init__(self):
        self.project_root = Path(os.path.dirname(os.path.dirname(__file__)))
        self._load_paths()

    def _load_paths(self):
        config_path = self.project_root / "Meditron" / "path.yaml"
        with open(config_path) as f:
            self.paths = yaml.safe_load(f)

        # Create processed directory if needed
        (self.project_root / self.paths["data"]["processed"]).mkdir(parents=True, exist_ok=True)

    @property
    def artificial_path(self):
        return self.project_root / self.paths["data"]["raw"]["artificial"]

    @property
    def labeled_path(self):
        return self.project_root / self.paths["data"]["raw"]["labeled"]