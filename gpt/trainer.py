import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Callable
from pathlib import Path

from config import BaseConfig
from models import BaseModel
from utils import setup_logger


class Trainer:
    """Handles model training and evaluation"""
    
    def __init__(self, 
                 model: BaseModel,
                 config: BaseConfig,
                 criterion: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: Optional[Any] = None,
                 metrics: Optional[Dict[str, Callable]] = None):
        self.model = model.to(config.device)
        self.config = config
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metrics = metrics or {}
        
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
        # Setup logging
        self.config.log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger('trainer', self.config.log_dir / 'train.log')
        
        # Create checkpoint directory
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        metric_values = {name: 0 for name in self.metrics.keys()}
        
        # for batch_idx, (inputs, targets) in enumerate(dataloader):
        for batch_idx, inputs in enumerate(dataloader):
            # print(inputs)
            inputs = torch.stack(inputs, dim=0)
            targets = inputs[:, 1:] # Shifted targets

            _, seq_len = inputs.shape
            self.mask = self.model.create_causal_mask(seq_len)

            inputs = inputs.to(self.config.device)
            targets = targets.to(self.config.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs, mask=self.mask)
            outputs = outputs[:, :-1, :]

            # loss = self.criterion(outputs, targets)
            loss = self.criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            for name, metric_fn in self.metrics.items():
                metric_values[name] += metric_fn(outputs, targets).item()
            
            if batch_idx % 10 == 0:
                self.logger.info(f'Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(dataloader)
        avg_metrics = {name: val / len(dataloader) for name, val in metric_values.items()}
        
        return {'loss': avg_loss, **avg_metrics}
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        metric_values = {name: 0 for name in self.metrics.keys()}
        
        with torch.no_grad():
            for inputs in dataloader:

                inputs = torch.stack(inputs, dim=0)
                targets = inputs[:, 1:] # Shifted targets

                _, seq_len = inputs.shape
                self.mask = self.model.create_causal_mask(seq_len)

                inputs = inputs.to(self.config.device)
                targets = targets.to(self.config.device)
                
                outputs = self.model(inputs, mask=self.mask)
                outputs = outputs[:, :-1, :]
                # loss = self.criterion(outputs, targets)
                loss = self.criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))
                total_loss += loss.item()
                
                for name, metric_fn in self.metrics.items():
                    metric_values[name] += metric_fn(outputs, targets).item()
        
        avg_loss = total_loss / len(dataloader)
        avg_metrics = {name: val / len(dataloader) for name, val in metric_values.items()}
        
        return {'loss': avg_loss, **avg_metrics}
    
    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        """Train the model"""
        self.logger.info(f'Starting training on {self.config.device}')
        self.logger.info(f'Model has {self.model.get_trainable_params():,} trainable parameters')

        # Tracking metrics
        self.train_metrics_history = {name: [] for name in self.metrics.keys()}
        self.val_metric_history = {name: [] for name in self.metrics.keys()}
        
        for epoch in range(self.config.num_epochs):
            self.logger.info(f'\nEpoch {epoch + 1}/{self.config.num_epochs}')
            
            train_metrics = self.train_epoch(train_loader)
            self.train_losses.append(train_metrics['loss'])

            # Store train metrics
            for name in self.metrics.keys():
                if name in train_metrics:
                    self.train_metrics_history[name].append(train_metrics[name])

            self.logger.info(f'Train Loss: {train_metrics["loss"]:.4f}')
            
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                self.val_losses.append(val_metrics['loss'])

                for name in self.metrics.keys():
                    if name in val_metrics:
                        self.val_metrics_history[name].append(val_metrics[name])
                        
                self.logger.info(f'Val Loss: {val_metrics["loss"]:.4f}')
                
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.save_checkpoint('best_model.pt', epoch, val_metrics)
                    self.logger.info('Saved best model')
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt', epoch)
    
    def save_checkpoint(self, filename: str, epoch: int, metrics: Optional[Dict] = None):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            # 'config': self.config.to_dict()
        }
        if metrics:
            checkpoint['metrics'] = metrics
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, self.config.checkpoint_dir / filename)
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint"""
        checkpoint = torch.load(
            self.config.checkpoint_dir / filename,
            map_location=self.config.device
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['epoch']