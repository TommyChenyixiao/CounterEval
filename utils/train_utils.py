# train_utils.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, auc, roc_auc_score
)
from tqdm import tqdm
import os
import json
from datetime import datetime
from pathlib import Path

class MetricsTracker:
    """Tracks and stores training metrics over time."""
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.initialize_metrics()
        
    def initialize_metrics(self):
        metrics_list = ['loss', 'accuracy', 'f1', 'precision', 'recall', 'auc']
        self.metrics = {
            'train': {metric: [] for metric in metrics_list},
            'val': {metric: [] for metric in metrics_list}
        }
    
    def update(self, phase: str, epoch_metrics: dict):
        for metric, value in epoch_metrics.items():
            self.metrics[phase][metric].append(float(value))
    
    def save(self):
        with open(self.log_dir / 'training_metrics.json', 'w') as f:
            json.dump(self.metrics, f, indent=4)

def train_epoch(model, loader, optimizer, criterion, device, clip_grad_norm=1.0):
    """Trains the model for one epoch."""
    model.train()
    total_loss = 0
    all_labels, all_preds, all_probs = [], [], []
    
    progress_bar = tqdm(loader, ascii=True)
    
    for data in progress_bar:
        data = data.to(device)
        optimizer.zero_grad()
        
        outputs = model(data.x, data.edge_index, data.batch)
        loss = criterion(outputs.view(-1), data.y.float())
        
        loss.backward()
        if clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()
        
        probs = torch.sigmoid(outputs).view(-1)
        preds = (probs > 0.5).float()
        
        total_loss += loss.item() * data.num_graphs
        all_probs.extend(probs.cpu().detach().numpy())
        all_preds.extend(preds.cpu().detach().numpy())
        all_labels.extend(data.y.cpu().numpy())
        
        # Update progress bar description
        curr_loss = loss.item()
        progress_bar.set_description(f"Loss: {curr_loss:.3f}")
    
    return compute_metrics(
        np.array(all_labels), 
        np.array(all_preds), 
        np.array(all_probs),
        total_loss / len(loader.dataset)
    )

def validate_epoch(model, loader, criterion, device):
    """Evaluates the model on validation/test data."""
    model.eval()
    total_loss = 0
    all_labels, all_preds, all_probs = [], [], []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            outputs = model(data.x, data.edge_index, data.batch)
            loss = criterion(outputs.view(-1), data.y.float())
            
            probs = torch.sigmoid(outputs).view(-1)
            preds = (probs > 0.5).float()
            
            total_loss += loss.item() * data.num_graphs
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())
    
    return compute_metrics(
        np.array(all_labels), 
        np.array(all_preds), 
        np.array(all_probs),
        total_loss / len(loader.dataset)
    )

def compute_metrics(labels, predictions, probabilities, loss=None):
    """Computes all relevant metrics for binary classification."""
    metrics = {
        'accuracy': accuracy_score(labels, predictions),
        'precision': precision_score(labels, predictions),
        'recall': recall_score(labels, predictions),
        'f1': f1_score(labels, predictions),
        'auc': roc_auc_score(labels, probabilities)
    }
    if loss is not None:
        metrics['loss'] = loss
    return metrics

def plot_training_curves(metrics_tracker, model_name, save_dir):
    """Plots detailed training curves for all metrics."""
    plt.figure(figsize=(20, 10))
    metrics = ['loss', 'accuracy', 'f1', 'precision', 'recall', 'auc']
    
    for idx, metric in enumerate(metrics, 1):
        plt.subplot(2, 3, idx)
        plt.plot(metrics_tracker.metrics['train'][metric], label='Train')
        plt.plot(metrics_tracker.metrics['val'][metric], label='Validation')
        plt.title(f'{metric.capitalize()} Over Time')
        plt.xlabel('Epoch')
        plt.ylabel(metric.capitalize())
        plt.grid(True)
        plt.legend()

    plt.suptitle(f'{model_name} Training Progress', y=1.02)
    plt.tight_layout()
    plt.savefig(Path(save_dir) / 'training_curves.png')
    plt.close()

def plot_model_comparison(results_dict, models, test_loader, device, save_dir):
    """
    Creates a comprehensive model comparison visualization with accuracy curves and ROC curves.
    
    Args:
        results_dict: Dictionary containing training metrics for each model
        models: Dictionary of trained models
        test_loader: DataLoader for test data
        device: Training device (CPU/GPU)
        save_dir: Directory to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Training and Validation Accuracy Curves
    for model_name, metrics in results_dict.items():
        ax1.plot(metrics.metrics['train']['accuracy'], 
                linestyle='-', label=f'{model_name} (Train)')
        ax1.plot(metrics.metrics['val']['accuracy'], 
                linestyle='--', label=f'{model_name} (Val)')
    
    ax1.set_title('Training and Validation Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.grid(True)
    ax1.legend()

    # ROC Curves
    for model_name, model in models.items():
        model.eval()
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                outputs = model(data.x, data.edge_index, data.batch)
                probs = torch.sigmoid(outputs).view(-1).cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(data.y.cpu().numpy())

        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        roc_auc = auc(fpr, tpr)
        ax2.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')

    ax2.plot([0, 1], [0, 1], 'k--', label='Random')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curves')
    ax2.grid(True)
    ax2.legend()

    plt.suptitle('Model Performance Comparison', y=1.02)
    plt.tight_layout()
    plt.savefig(Path(save_dir) / 'model_comparison.png', 
                bbox_inches='tight', dpi=300)
    plt.close()

def setup_experiment(model_name: str, config: dict) -> Path:
    """Sets up experiment directory and saves configuration."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path('results') / f'{model_name}_{timestamp}'
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    return exp_dir

def save_results_summary(results_dict, models_test_metrics, training_time, save_dir, start_time):
    """Creates a detailed results summary with unique timestamp identifier."""
    results_file = Path(save_dir) / f'results_{start_time}.txt'
    
    with open(results_file, 'w') as f:
        f.write("Model Performance Summary\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Experiment Details:\n")
        f.write(f"Start Time: {start_time}\n")
        f.write(f"Total Training Duration: {training_time:.2f} seconds\n\n")
        
        f.write("Test Performance Metrics:\n")
        row_format = "{:<15} {:<10} {:<10} {:<10} {:<15} {:<10}"
        headers = ['Model', 'Test Acc', 'Test F1', 'Test AUC', 'Best Val Loss', 'Best Val F1']
        
        f.write(row_format.format(*headers) + "\n")
        f.write("-" * 80 + "\n")
        
        for model_name in results_dict.keys():
            test_metrics = models_test_metrics[model_name]
            best_val_idx = np.argmin(results_dict[model_name].metrics['val']['loss'])
            best_val_loss = results_dict[model_name].metrics['val']['loss'][best_val_idx]
            best_val_f1 = results_dict[model_name].metrics['val']['f1'][best_val_idx]
            
            row = [
                model_name,
                f"{test_metrics['accuracy']:.3f}",
                f"{test_metrics['f1']:.3f}",
                f"{test_metrics['auc']:.3f}",
                f"{best_val_loss:.3f}",
                f"{best_val_f1:.3f}"
            ]
            f.write(row_format.format(*row) + "\n")