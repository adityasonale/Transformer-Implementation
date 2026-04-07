"""
Visualization utilities for deep learning projects
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import json

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 100


class Visualizer:
    """
    Handles all visualization tasks
    """
    
    def __init__(self, save_dir: Path = Path('visualizations')):
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_training_history(self, 
                             train_losses: List[float],
                             val_losses: Optional[List[float]] = None,
                             train_metrics: Optional[Dict[str, List[float]]] = None,
                             val_metrics: Optional[Dict[str, List[float]]] = None,
                             save_name: str = 'training_history.png'):
        """
        Plot training and validation losses/metrics over epochs
        
        Args:
            train_losses: List of training losses per epoch
            val_losses: List of validation losses per epoch
            train_metrics: Dict of metric_name -> list of values
            val_metrics: Dict of metric_name -> list of values
            save_name: Filename to save the plot
        """
        # Determine number of subplots needed
        n_plots = 1  # Always have loss
        if train_metrics:
            n_plots += len(train_metrics)
        
        fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 5))
        if n_plots == 1:
            axes = [axes]
        
        # Plot losses
        epochs = range(1, len(train_losses) + 1)
        axes[0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
        if val_losses:
            axes[0].plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Plot metrics
        if train_metrics:
            for idx, (metric_name, values) in enumerate(train_metrics.items()):
                ax = axes[idx + 1]
                ax.plot(epochs, values, 'b-', label=f'Train {metric_name}', linewidth=2)
                if val_metrics and metric_name in val_metrics:
                    ax.plot(epochs, val_metrics[metric_name], 'r-', 
                           label=f'Val {metric_name}', linewidth=2)
                ax.set_xlabel('Epoch', fontsize=12)
                ax.set_ylabel(metric_name.capitalize(), fontsize=12)
                ax.set_title(f'{metric_name.capitalize()} Over Time', 
                           fontsize=14, fontweight='bold')
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Training history saved to {self.save_dir / save_name}")
    
    def plot_batch_samples(self,
                          images: torch.Tensor,
                          labels: torch.Tensor,
                          predictions: Optional[torch.Tensor] = None,
                          class_names: Optional[List[str]] = None,
                          n_samples: int = 16,
                          save_name: str = 'batch_samples.png',
                          denormalize_fn: Optional[callable] = None):
        """
        Visualize a batch of images with labels and predictions
        
        Args:
            images: Tensor of shape (B, C, H, W)
            labels: Tensor of shape (B,)
            predictions: Optional tensor of predictions (B,)
            class_names: List of class names
            n_samples: Number of samples to show
            save_name: Filename to save
            denormalize_fn: Function to denormalize images
        """
        n_samples = min(n_samples, images.size(0))
        grid_size = int(np.ceil(np.sqrt(n_samples)))
        
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
        axes = axes.flatten()
        
        for idx in range(n_samples):
            img = images[idx].cpu()
            
            # Denormalize if function provided
            if denormalize_fn:
                img = denormalize_fn(img)
            
            # Convert to numpy and transpose to (H, W, C) for plotting
            if img.shape[0] == 1:  # Grayscale
                img = img.squeeze(0)
                axes[idx].imshow(img, cmap='gray')
            else:  # RGB
                img = img.permute(1, 2, 0)
                axes[idx].imshow(img)
            
            # Create title
            label = labels[idx].item()
            title = f"True: {class_names[label] if class_names else label}"
            
            if predictions is not None:
                pred = predictions[idx].item()
                pred_name = class_names[pred] if class_names else pred
                title += f"\nPred: {pred_name}"
                
                # Color title based on correct/incorrect
                color = 'green' if label == pred else 'red'
                axes[idx].set_title(title, fontsize=9, color=color)
            else:
                axes[idx].set_title(title, fontsize=9)
            
            axes[idx].axis('off')
        
        # Hide unused subplots
        for idx in range(n_samples, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Batch samples saved to {self.save_dir / save_name}")
    
    def plot_confusion_matrix(self,
                            y_true: np.ndarray,
                            y_pred: np.ndarray,
                            class_names: Optional[List[str]] = None,
                            save_name: str = 'confusion_matrix.png',
                            normalize: bool = False):
        """
        Plot confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: List of class names
            save_name: Filename to save
            normalize: Whether to normalize the confusion matrix
        """
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                   xticklabels=class_names if class_names else 'auto',
                   yticklabels=class_names if class_names else 'auto')
        plt.title(title, fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix saved to {self.save_dir / save_name}")
    
    def plot_data_distribution(self,
                             labels: np.ndarray,
                             class_names: Optional[List[str]] = None,
                             save_name: str = 'data_distribution.png',
                             title: str = 'Class Distribution'):
        """
        Plot distribution of classes in dataset
        
        Args:
            labels: Array of labels
            class_names: List of class names
            save_name: Filename to save
            title: Plot title
        """
        unique, counts = np.unique(labels, return_counts=True)
        
        plt.figure(figsize=(10, 6))
        x_labels = [class_names[i] if class_names else str(i) for i in unique]
        
        bars = plt.bar(x_labels, counts, color=sns.color_palette("husl", len(unique)))
        plt.xlabel('Class', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Data distribution saved to {self.save_dir / save_name}")
    
    def plot_feature_maps(self,
                         feature_maps: torch.Tensor,
                         save_name: str = 'feature_maps.png',
                         n_features: int = 16):
        """
        Visualize feature maps from a convolutional layer
        
        Args:
            feature_maps: Tensor of shape (B, C, H, W)
            save_name: Filename to save
            n_features: Number of feature maps to show
        """
        # Take first sample and first n_features channels
        feature_maps = feature_maps[0].cpu().detach()[:n_features]
        n_features = min(n_features, feature_maps.shape[0])
        
        grid_size = int(np.ceil(np.sqrt(n_features)))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
        axes = axes.flatten()
        
        for idx in range(n_features):
            axes[idx].imshow(feature_maps[idx], cmap='viridis')
            axes[idx].set_title(f'Feature {idx}', fontsize=9)
            axes[idx].axis('off')
        
        # Hide unused subplots
        for idx in range(n_features, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Feature maps saved to {self.save_dir / save_name}")
    
    def plot_learning_rate(self,
                          learning_rates: List[float],
                          save_name: str = 'learning_rate.png'):
        """
        Plot learning rate schedule over training
        
        Args:
            learning_rates: List of learning rates per epoch/iteration
            save_name: Filename to save
        """
        plt.figure(figsize=(10, 6))
        plt.plot(learning_rates, linewidth=2)
        plt.xlabel('Iteration/Epoch', fontsize=12)
        plt.ylabel('Learning Rate', fontsize=12)
        plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Learning rate plot saved to {self.save_dir / save_name}")
    
    def plot_gradient_flow(self,
                          named_parameters,
                          save_name: str = 'gradient_flow.png'):
        """
        Plot gradient flow through network layers
        
        Args:
            named_parameters: model.named_parameters()
            save_name: Filename to save
        """
        ave_grads = []
        max_grads = []
        layers = []
        
        for n, p in named_parameters:
            if p.requires_grad and p.grad is not None:
                layers.append(n)
                ave_grads.append(p.grad.abs().mean().cpu().item())
                max_grads.append(p.grad.abs().max().cpu().item())
        
        plt.figure(figsize=(12, 6))
        plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.5, lw=1, color="c", label="max")
        plt.bar(np.arange(len(ave_grads)), ave_grads, alpha=0.5, lw=1, color="b", label="mean")
        plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k")
        plt.xticks(range(0, len(ave_grads), 1), layers, rotation=90, fontsize=8)
        plt.xlim(left=0, right=len(ave_grads))
        plt.ylim(bottom=-0.001, top=max(max_grads)*1.1)
        plt.xlabel("Layers", fontsize=12)
        plt.ylabel("Average Gradient", fontsize=12)
        plt.title("Gradient Flow", fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Gradient flow saved to {self.save_dir / save_name}")
    
    def save_metrics_json(self, metrics: Dict[str, Any], filename: str = 'metrics.json'):
        """
        Save metrics to JSON file
        
        Args:
            metrics: Dictionary of metrics
            filename: Name of JSON file
        """
        filepath = self.save_dir / filename
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"Metrics saved to {filepath}")


# Example usage functions
def plot_dataset_overview(dataset, visualizer: Visualizer, n_samples: int = 16):
    """
    Quick overview of a dataset
    
    Args:
        dataset: PyTorch Dataset
        visualizer: Visualizer instance
        n_samples: Number of samples to show
    """
    # Get samples
    indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)
    images = []
    labels = []
    
    for idx in indices:
        img, label = dataset[idx]
        images.append(img)
        labels.append(label)
    
    images = torch.stack(images)
    labels = torch.tensor(labels)
    
    # Plot samples
    visualizer.plot_batch_samples(images, labels, n_samples=n_samples, 
                                 save_name='dataset_samples.png')
    
    # Plot distribution
    all_labels = [dataset[i][1] for i in range(len(dataset))]
    visualizer.plot_data_distribution(np.array(all_labels), 
                                     save_name='dataset_distribution.png')


def denormalize_imagenet(tensor):
    """Helper function to denormalize ImageNet normalized images"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return tensor * std + mean