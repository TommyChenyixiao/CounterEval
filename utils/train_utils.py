import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, roc_curve, auc
from tqdm import tqdm
import os
import json
from datetime import datetime

class TrainLogger:
    """Logging class for training metrics and visualization"""
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'train_f1': [], 'val_f1': []
        }
    
    def log(self, metrics, phase='train'):
        for key, value in metrics.items():
            self.history[f'{phase}_{key}'].append(value)
    
    def save_history(self):
        with open(f'{self.log_dir}/training_history.json', 'w') as f:
            json.dump(self.history, f)

def train_epoch(model, loader, optimizer, criterion, device):
    """Run one training epoch"""
    model.train()
    total_loss = 0
    predictions, labels, probs = [], [], []
    
    for data in tqdm(loader, desc='Training'):
        data = data.to(device)
        optimizer.zero_grad()
        
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out.view(-1), data.y.float())
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * data.num_graphs
        probs.extend(torch.sigmoid(out).view(-1).cpu().detach().numpy())
        predictions.extend((torch.sigmoid(out).view(-1) > 0.5).float().cpu().detach().numpy())
        labels.extend(data.y.cpu().numpy())
    
    return {
        'loss': total_loss / len(loader.dataset),
        'acc': accuracy_score(labels, predictions),
        'f1': f1_score(labels, predictions)
    }

def validate_epoch(model, loader, criterion, device):
    """Run one validation epoch"""
    model.eval()
    total_loss = 0
    predictions, labels, probs = [], [], []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out.view(-1), data.y.float())
            
            total_loss += loss.item() * data.num_graphs
            probs.extend(torch.sigmoid(out).view(-1).cpu().numpy())
            predictions.extend((torch.sigmoid(out).view(-1) > 0.5).float().cpu().numpy())
            labels.extend(data.y.cpu().numpy())
    
    return {
        'loss': total_loss / len(loader.dataset),
        'acc': accuracy_score(labels, predictions),
        'f1': f1_score(labels, predictions)
    }

def plot_training_progress(history, model_name, results_dir):
    """Plot training metrics"""
    plt.figure(figsize=(15, 5))
    
    # Loss plot
    plt.subplot(131)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Accuracy plot
    plt.subplot(132)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.title('Accuracy Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # F1 Score plot
    plt.subplot(133)
    plt.plot(history['train_f1'], label='Train')
    plt.plot(history['val_f1'], label='Validation')
    plt.title('F1 Score Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{results_dir}/training_progress.png')
    plt.close()

def evaluate_model(model, loader, device, model_name, results_dir):
    """Evaluate model and generate performance visualizations"""
    model.eval()
    predictions, labels, probs = [], [], []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            probs.extend(torch.sigmoid(out).view(-1).cpu().numpy())
            predictions.extend((torch.sigmoid(out).view(-1) > 0.5).float().cpu().numpy())
            labels.extend(data.y.cpu().numpy())
    
    predictions = np.array(predictions)
    labels = np.array(labels)
    probs = np.array(probs)
    
    # Generate and save classification report
    report = classification_report(labels, predictions)
    with open(f'{results_dir}/classification_report.txt', 'w') as f:
        f.write(report)
    
    # Plot confusion matrix
    plot_confusion_matrix(labels, predictions, model_name, results_dir)
    
    # Plot ROC curve
    plot_roc_curve(labels, probs, model_name, results_dir)
    
    return {
        'accuracy': accuracy_score(labels, predictions),
        'f1': f1_score(labels, predictions)
    }

def plot_confusion_matrix(y_true, y_pred, model_name, results_dir):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'{results_dir}/confusion_matrix.png')
    plt.close()

def plot_roc_curve(y_true, y_prob, model_name, results_dir):
    """Plot and save ROC curve"""
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} ROC Curve')
    plt.legend()
    plt.savefig(f'{results_dir}/roc_curve.png')
    plt.close()

def compare_models(results, save_dir):
    """Generate model comparison visualizations"""
    metrics = ['accuracy', 'f1']
    model_names = list(results.keys())
    
    plt.figure(figsize=(10, 6))
    x = np.arange(len(metrics))
    width = 0.8 / len(model_names)
    
    for i, model_name in enumerate(model_names):
        values = [results[model_name]['metrics'][m] for m in metrics]
        plt.bar(x + i*width, values, width, label=model_name)
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Model Comparison')
    plt.xticks(x + width*(len(model_names)-1)/2, metrics)
    plt.legend()
    plt.savefig(f'{save_dir}/model_comparison.png')
    plt.close()

def setup_training(model_name, config):
    """Setup training directories and save config"""
    results_dir = f'results/{model_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    os.makedirs(results_dir, exist_ok=True)
    
    with open(f'{results_dir}/config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    return results_dir