"""Configuration management utilities."""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass
import torch


class ConfigLoader:
    """Load and manage configuration files."""
    
    def __init__(self, config_dir: Union[str, Path] = "config"):
        """Initialize config loader."""
        self.config_dir = Path(config_dir)
        self._configs = {}
    
    def load_config(self, config_name: str) -> Dict[str, Any]:
        """Load a configuration file."""
        if config_name in self._configs:
            return self._configs[config_name]
        
        config_path = self.config_dir / f"{config_name}.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self._configs[config_name] = config
        return config
    
    def get(self, config_name: str, key_path: str, default: Any = None) -> Any:
        """Get a configuration value by dot notation path."""
        config = self.load_config(config_name)
        
        keys = key_path.split('.')
        value = config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default


@dataclass
class TrainingConfig:
    """Training configuration dataclass."""
    epochs: int = 10
    batch_size: int = 4
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    validate_every: int = 1
    grad_clip_max_norm: float = 1.0
    val_split: float = 0.2
    stratify: bool = True
    num_workers: int = 4
    pin_memory: bool = True
    shuffle: bool = True
    drop_last: bool = True
    
    @classmethod
    def from_config(cls, config_loader: ConfigLoader) -> "TrainingConfig":
        """Create TrainingConfig from config files."""
        training_cfg = config_loader.load_config("training_config")
        
        return cls(
            epochs=config_loader.get("training_config", "training.epochs", cls.epochs),
            batch_size=config_loader.get("training_config", "training.batch_size", cls.batch_size),
            learning_rate=config_loader.get("training_config", "training.learning_rate", cls.learning_rate),
            weight_decay=config_loader.get("training_config", "training.weight_decay", cls.weight_decay),
            warmup_ratio=config_loader.get("training_config", "training.warmup_ratio", cls.warmup_ratio),
            validate_every=config_loader.get("training_config", "training.validate_every", cls.validate_every),
            grad_clip_max_norm=config_loader.get("training_config", "training.grad_clip_max_norm", cls.grad_clip_max_norm),
            val_split=config_loader.get("training_config", "data.val_split", cls.val_split),
            stratify=config_loader.get("training_config", "data.stratify", cls.stratify),
            num_workers=config_loader.get("training_config", "data.num_workers", cls.num_workers),
            pin_memory=config_loader.get("training_config", "data.pin_memory", cls.pin_memory),
            shuffle=config_loader.get("training_config", "data.shuffle", cls.shuffle),
            drop_last=config_loader.get("training_config", "data.drop_last", cls.drop_last),
        )


@dataclass
class ModelConfig:
    """Model configuration dataclass."""
    model_name: str = "facebook/wav2vec2-base-960h"
    num_classes: int = 2
    freeze_feature_extractor: bool = False
    freeze_transformer: bool = False
    dropout_rate: float = 0.1
    hidden_dim: int = 256
    target_sample_rate: int = 16000
    max_duration: float = 10.0
    normalize_audio: bool = True
    
    @classmethod
    def from_config(cls, config_loader: ConfigLoader) -> "ModelConfig":
        """Create ModelConfig from config files."""
        return cls(
            model_name=config_loader.get("model_config", "wav2vec2.model_name", cls.model_name),
            num_classes=config_loader.get("model_config", "wav2vec2.num_classes", cls.num_classes),
            freeze_feature_extractor=config_loader.get("model_config", "wav2vec2.freeze_feature_extractor", cls.freeze_feature_extractor),
            freeze_transformer=config_loader.get("model_config", "wav2vec2.freeze_transformer", cls.freeze_transformer),
            dropout_rate=config_loader.get("model_config", "wav2vec2.dropout_rate", cls.dropout_rate),
            hidden_dim=config_loader.get("model_config", "wav2vec2.hidden_dim", cls.hidden_dim),
            target_sample_rate=config_loader.get("model_config", "audio.target_sample_rate", cls.target_sample_rate),
            max_duration=config_loader.get("model_config", "audio.max_duration", cls.max_duration),
            normalize_audio=config_loader.get("model_config", "audio.normalize_audio", cls.normalize_audio),
        )


def detect_device(priority: list = None, force: str = None) -> str:
    """
    Detect the best available device with priority: MPS -> CUDA -> CPU.
    
    Args:
        priority: List of device preferences ["mps", "cuda", "cpu"]
        force: Force a specific device (overrides priority)
        
    Returns:
        Device string ("mps", "cuda", or "cpu")
    """
    if force:
        return force
    
    if priority is None:
        priority = ["mps", "cuda", "cpu"]
    
    for device in priority:
        if device == "mps" and torch.backends.mps.is_available():
            return "mps"
        elif device == "cuda" and torch.cuda.is_available():
            return "cuda"
        elif device == "cpu":
            return "cpu"
    
    # Fallback to CPU if nothing else works
    return "cpu"


def get_device_from_config(config_loader: ConfigLoader) -> str:
    """Get device from configuration."""
    priority = config_loader.get("training_config", "device.priority", ["mps", "cuda", "cpu"])
    force = config_loader.get("training_config", "device.force", None)
    
    device = detect_device(priority=priority, force=force)
    
    print(f"Device selected: {device}")
    if device == "mps":
        print("Using Apple Metal Performance Shaders (MPS)")
    elif device == "cuda":
        print(f"Using CUDA device: {torch.cuda.get_device_name()}")
    else:
        print("Using CPU (consider using GPU for faster training)")
    
    return device


# Global config loader instance
config_loader = ConfigLoader()