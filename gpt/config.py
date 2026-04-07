import torch
from typing import Dict, Any, Optional
import json
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseConfig:
    """Base configuration class for managing hyperparameters and settings"""
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        # Training parameters
        self.batch_size = 32
        self.learning_rate = 1e-3
        self.num_epochs = 10
        self.val_split = 0.1
        self.embd_size = 128
        self.num_heads = 8
        self.num_layers = 1
        self.weight_decay = 0.1
        self.betas = (0.9, 0.95)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Model parameters
        self.model_name = 'base_model'
        self.checkpoint_dir = Path('checkpoints')
        self.log_dir = Path('logs')
        
        # Data parameters
        self.data_dir = Path('data')
        self.num_workers = 1
        
        if config_dict:
            self.update(config_dict)
    
    def update(self, config_dict: Dict[str, Any]):
        """Update configuration from dictionary"""
        for key, value in config_dict.items():
            setattr(self, key, value)
    
    def save(self, path: Path):
        """Save configuration to JSON file"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=4, default=str)
    
    @classmethod
    def load(cls, path: Path):
        """Load configuration from JSON file"""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(config_dict)