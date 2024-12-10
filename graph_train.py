import torch
from torch_geometric.loader import DataLoader
from pathlib import Path

from models.gnn_models import create_gnn_model
from utils.train_utils import (
    MetricsTracker, train_epoch, validate_epoch,
    plot_training_curves, plot_model_comparison, setup_experiment
)

def train_model(model, train_loader, val_loader, device, model_name, config, exp_dir):
    metrics_tracker = MetricsTracker(exp_dir)
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
        if config['scheduler'] == 'reduce_on_plateau'
        else torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    )
    
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([config['pos_weight']]).to(device)
        if 'pos_weight' in config else None
    )
    
    best_val_f1 = 0
    patience_counter = 0
    
    print(f"\n{model_name} Training Progress:")
    for epoch in range(config['epochs']):
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = validate_epoch(model, val_loader, criterion, device)
        
        metrics_tracker.update('train', train_metrics)
        metrics_tracker.update('val', val_metrics)
        
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            torch.save(model.state_dict(), exp_dir / 'best_model.pt')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f'Early stopping at epoch {epoch + 1}')
                break
        
        if config['scheduler'] == 'reduce_on_plateau':
            scheduler.step(val_metrics['f1'])
        else:
            scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1:03d} | Train Loss: {train_metrics["loss"]:.3f} F1: {train_metrics["f1"]:.3f} | Val Loss: {val_metrics["loss"]:.3f} F1: {val_metrics["f1"]:.3f}')
    
    plot_training_curves(metrics_tracker, model_name, exp_dir)
    metrics_tracker.save()
    return model, metrics_tracker
def main():
    train_data = torch.load('processed_data/men_balanced_train_graph_dataset.pt')
    val_data = torch.load('processed_data/men_imbalanced_val_graph_dataset.pt')
    test_data = torch.load('processed_data/men_imbalanced_test_graph_dataset.pt')
    
    model_configs = {
        'GCN': {'hidden_channels': 64, 'num_layers': 3, 'dropout': 0.5, 'pool_type': 'mean'},
        'GAT': {'hidden_channels': 32, 'num_layers': 2, 'heads': 4, 'dropout': 0.5, 'pool_type': 'mean'},
        'GraphSAGE': {'hidden_channels': 64, 'num_layers': 3, 'dropout': 0.5, 'pool_type': 'max'},
        'Transformer': {'hidden_channels': 32, 'num_layers': 2, 'heads': 4, 'dropout': 0.5, 'pool_type': 'mean'}
    }
    
    train_config = {
        'epochs': 100,
        'lr': 0.001,
        'weight_decay': 5e-4,
        'batch_size': 32,
        'patience': 10,
        'scheduler': 'reduce_on_plateau',
        'pos_weight': 2.0,
        'clip_grad_norm': 1.0
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device} for training')
    
    results_dict = {}
    trained_models = {}
    test_loader = DataLoader(test_data, batch_size=train_config['batch_size'])
    
    for model_name, model_config in model_configs.items():
        print(f'\n{"-"*50}\nTraining {model_name} Model\n{"-"*50}')
        
        exp_dir = setup_experiment(model_name, {**model_config, **train_config})
        
        train_loader = DataLoader(train_data, batch_size=train_config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_data, batch_size=train_config['batch_size'])
        
        model = create_gnn_model(model_name, train_data[0].num_features, **model_config)
        model = model.to(device)
        
        model, metrics_tracker = train_model(
            model, train_loader, val_loader, device,
            model_name, {**model_config, **train_config}, exp_dir
        )
        
        test_metrics = validate_epoch(model, test_loader, torch.nn.BCEWithLogitsLoss(), device)
        print(f'\nTest Results for {model_name}:')
        print(f'F1: {test_metrics["f1"]:.3f} | AUC: {test_metrics["auc"]:.3f} | Acc: {test_metrics["accuracy"]:.3f}')
        
        results_dict[model_name] = metrics_tracker
        trained_models[model_name] = model
    
    plot_model_comparison(results_dict, trained_models, test_loader, device, 'results')

if __name__ == '__main__':
    main()
