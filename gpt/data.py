"""
Data handling classes
"""
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Callable,  Dict, List
from pathlib import Path
from abc import ABC, abstractmethod
from config import BaseConfig
import pickle

class BaseDataset(Dataset, ABC):
    """
    Abstract base dataset
    """
    
    def __init__(self, data_path: Path=None, data=None, transform: Optional[Callable] = None):
        self.transform = transform

        if data is not None:
            self.data = data
        elif data_path is not None:
            self.data_path = data_path
            self.data = self._load_data()
        else:
            raise ValueError("Provide either data or data_path")
    
    @abstractmethod
    def _load_data(self):
        """Load data from disk - implement in subclass"""
        pass
    
    @abstractmethod
    def __len__(self):
        """Return dataset size"""
        pass
    
    @abstractmethod
    def __getitem__(self, idx: int):
        """Get item by index"""
        pass

class GptDataset(BaseDataset):
    def __init__(self, data_path=None, data=None, transform=None):
        super().__init__(data_path=data_path, data=data, transform=transform)

    def _load_data(self):
        with open(self.data_path, encoding="utf-8") as f:
            return f.readlines()
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        return self.transform(x) if self.transform else x


class GptDataLoader:
    """Handles data loading and preprocessing"""
    
    def __init__(self, 
                 config: BaseConfig,
                 train_dataset: Dataset,
                 val_dataset: Optional[Dataset] = None,
                 test_dataset: Optional[Dataset] = None):
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True if self.config.device == 'cuda' else False
        )
    
    def val_dataloader(self) -> Optional[DataLoader]:
        if self.val_dataset is None:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True if self.config.device == 'cuda' else False
        )
    
    def test_dataloader(self) -> Optional[DataLoader]:
        if self.test_dataset is None:
            return None
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers
        )
    
class DataInfo:
    """
    Stores dataset metadata and vocabulary mappings
    """
    
    def __init__(self):
        # Vocabulary
        self.idx2token: Dict[int, str] = {}
        self.token2idx: Dict[str, int] = {}
        self.vocab_size: int = 0
        self.max_len = 0
        
        # Classification
        self.idx2class: Dict[int, str] = {}
        self.class2idx: Dict[str, int] = {}
        self.num_classes: int = 0
        
        # Special token indices (optional)
        self.pad_idx: Optional[int] = None
        self.unk_idx: Optional[int] = None
    
    def build_vocab(self, tokens: List[str]):
        """Build vocabulary from list of unique tokens"""
        self.token2idx = {token: idx for idx, token in enumerate(tokens)}
        self.idx2token = {idx: token for token, idx in self.token2idx.items()}
        self.vocab_size = len(tokens)
    
    def set_classes(self, class_names: List[str]):
        """Set class mappings from list of class names"""
        self.class2idx = {name: idx for idx, name in enumerate(class_names)}
        self.idx2class = {idx: name for name, idx in self.class2idx.items()}
        self.num_classes = len(class_names)
    
    def save(self, path: Path):
        """Save to pickle file"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path: Path) -> 'DataInfo':
        """Load from pickle file"""
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    def __repr__(self):
        return f"DataInfo(vocab_size={self.vocab_size}, num_classes={self.num_classes})"
