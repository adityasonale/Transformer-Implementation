import logging
from pathlib import Path
from typing import Any

def setup_logger(name: str, log_file: Path = None, level=logging.INFO):
    """Setup logger"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def count_parameters(model):
    """Count model parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_dict_to_json(d: dict, path: Path):
    """Save dictionary to JSON"""
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(d, f, indent=4, default=str)


def load_dict_from_json(path: Path) -> dict:
    """Load dictionary from JSON"""
    import json
    with open(path, 'r') as f:
        return json.load(f)