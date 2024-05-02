"""
Configuration Management for Text2TuneAI
Handles loading and accessing configuration parameters
"""

import os
import yaml
from typing import Any, Dict, Optional
from pathlib import Path
from omegaconf import OmegaConf, DictConfig


class Config:
    """Configuration manager for Text2TuneAI project"""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration

        Args:
            config_path: Path to config.yaml file. If None, uses default path.
        """
        if config_path is None:
            # Default config path
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "configs" / "config.yaml"

        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._setup_directories()

    def _load_config(self) -> DictConfig:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Convert to OmegaConf for better access
        config = OmegaConf.create(config_dict)

        # Resolve paths relative to project root
        project_root = self.config_path.parent.parent
        config.project.root = str(project_root)

        return config

    def _setup_directories(self):
        """Create necessary directories if they don't exist"""
        project_root = Path(self.config.project.root)

        directories = [
            project_root / self.config.data.dali.root_dir,
            project_root / self.config.data.lakh.root_dir,
            project_root / self.config.data.processed_dir,
            project_root / self.config.data.cache_dir,
            project_root / self.config.data.samples_dir,
            project_root / self.config.paths.checkpoints,
            project_root / self.config.paths.logs,
            project_root / self.config.paths.outputs,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation

        Args:
            key: Configuration key (e.g., 'model.text_encoder.hidden_size')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        try:
            return OmegaConf.select(self.config, key, default=default)
        except Exception:
            return default

    def update(self, key: str, value: Any):
        """
        Update configuration value

        Args:
            key: Configuration key (e.g., 'training.batch_size')
            value: New value
        """
        OmegaConf.update(self.config, key, value)

    def save(self, path: Optional[str] = None):
        """
        Save configuration to file

        Args:
            path: Output path. If None, overwrites original config file.
        """
        if path is None:
            path = self.config_path

        with open(path, 'w') as f:
            OmegaConf.save(self.config, f)

    def to_dict(self) -> Dict:
        """Convert configuration to dictionary"""
        return OmegaConf.to_container(self.config, resolve=True)

    @property
    def model_config(self) -> DictConfig:
        """Get model configuration"""
        return self.config.model

    @property
    def training_config(self) -> DictConfig:
        """Get training configuration"""
        return self.config.training

    @property
    def data_config(self) -> DictConfig:
        """Get data configuration"""
        return self.config.data

    @property
    def generation_config(self) -> DictConfig:
        """Get generation configuration"""
        return self.config.generation

    def __repr__(self) -> str:
        return f"Config(config_path={self.config_path})"

    def __str__(self) -> str:
        return OmegaConf.to_yaml(self.config)


# Global config instance
_global_config: Optional[Config] = None


def get_config(config_path: Optional[str] = None) -> Config:
    """
    Get global configuration instance

    Args:
        config_path: Path to config file. Only used on first call.

    Returns:
        Config instance
    """
    global _global_config

    if _global_config is None:
        _global_config = Config(config_path)

    return _global_config


def reset_config():
    """Reset global configuration (useful for testing)"""
    global _global_config
    _global_config = None


if __name__ == "__main__":
    # Test configuration loading
    config = get_config()
    print("Configuration loaded successfully!")
    print(f"\nProject: {config.get('project.name')}")
    print(f"Model: {config.get('model.name')}")
    print(f"Batch size: {config.get('training.batch_size')}")
    print(f"\nFull config:\n{config}")
