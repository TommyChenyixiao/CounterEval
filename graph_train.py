import torch
from torch_geometric.loader import DataLoader
import os
from datetime import datetime

from models.gnn_models import create_gnn_model
from utils.train_utils import (
    TrainLogger, train_epoch, validate_epoch, plot_training_progress,
    evaluate_model, compare_models, setup_training
)

def train_model(model, train_loader, val_loader, device, model_name, config, results_dir):
    """Train a GNN model"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), 
                               lr=config['lr'], 
                               weight_decay=config['weight_decay'])
    
    # Setup learning rate scheduler
    if config['scheduler'] == 'reduce_on_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )
    elif config['scheduler'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs']
        )
    
    # Setup loss function with class weights if specified
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([config['pos_weight']]).to(device) 
        if 'pos_weight' in config else None
    )
    
    logger = TrainLogger(results_dir)
    best_val_f1 = 0
    best_model = None
    patience_counter = 0
    
    print(f"\nTraining {model_name}")
    for epoch in range(config['epochs']):
        # Training phase
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validation phase
        val_metrics = validate_epoch(model, val_loader, criterion, device)
        
        # Update learning rate
        if config['scheduler'] == 'reduce_on_plateau':
            scheduler.step(val_metrics['f1'])
        elif config['scheduler'] == 'cosine':
            scheduler.step()
        
        # Log metrics
        logger.log(train_metrics, 'train')
        logger.log(val_metrics, 'val')
        
        # Print progress
        print(f'\nEpoch {epoch:03d}:')
        print(f'Train Loss: {train_metrics["loss"]:.4f}, '
              f'Acc: {train_metrics["acc"]:.4f}, F1: {train_metrics["f1"]:.4f}')
        print(f'Val Loss: {val_metrics["loss"]:.4f}, '
              f'Acc: {val_metrics["acc"]:.4f}, F1: {val_metrics["f1"]:.4f}')
        
        # Save best model and check early stopping
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_model = model.state_dict()
            torch.save(best_model, f'{results_dir}/best_model.pt')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f'\nEarly stopping triggered after {epoch} epochs')
                break
        
        # Plot progress
        if epoch % config['plot_interval'] == 0 or epoch == config['epochs'] - 1:
            plot_training_progress(logger.history, model_name, results_dir)
    
    # Load best model and evaluate
    model.load_state_dict(best_model)
    logger.save_history()
    final_metrics = evaluate_model(model, val_loader, device, model_name, results_dir)
    
    # Return the trained model, training history, and final metrics
    return model, logger.history, final_metrics

def main():
    # Load split datasets
    train_data = torch.load('processed_data/men_balanced_train_graph_dataset.pt')
    val_data = torch.load('processed_data/men_imbalanced_val_graph_dataset.pt')
    test_data = torch.load('processed_data/men_imbalanced_test_graph_dataset.pt')
    
    # Model configurations
    model_configs = {
        'GCN': {
            'hidden_channels': 64,
            'num_layers': 3,
            'dropout': 0.5,
            'pool_type': 'mean'
        },
        'GAT': {
            'hidden_channels': 32,
            'num_layers': 2,
            'heads': 4,
            'dropout': 0.5,
            'pool_type': 'mean'
        },
        'GraphSAGE': {
            'hidden_channels': 64,
            'num_layers': 3,
            'dropout': 0.5,
            'pool_type': 'max'
        }
    }
    
    # Training configuration
    train_config = {
        'epochs': 100,
        'lr': 0.001,
        'weight_decay': 5e-4,
        'batch_size': 32,
        'patience': 10,
        'plot_interval': 10,
        'scheduler': 'reduce_on_plateau',
        'pos_weight': 2.0  # Adjust this based on your class distribution if needed
    }
    
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=train_config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=train_config['batch_size'])
    test_loader = DataLoader(test_data, batch_size=train_config['batch_size'])
    
    # Get number of features from the first graph
    num_features = train_data[0].num_features
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Print dataset sizes
    print(f"\nDataset sizes:")
    print(f"Training: {len(train_data)} graphs")
    print(f"Validation: {len(val_data)} graphs")
    print(f"Test: {len(test_data)} graphs")
    
    # Train models
    results = {}
    for model_name, model_config in model_configs.items():
        print(f"\n{'='*50}\nTraining {model_name}\n{'='*50}")
        
        # Setup training
        config = {**model_config, **train_config}
        results_dir = setup_training(model_name, config)
        
        # Create and train model
        model = create_gnn_model(model_name, num_features, **model_config)
        trained_model, history, metrics = train_model(
            model, train_loader, val_loader, device, model_name,
            config, results_dir
        )
        
        # Evaluate on test set
        print(f"\nEvaluating {model_name} on test set:")
        test_metrics = evaluate_model(trained_model, test_loader, device, 
                                    f"{model_name}_test", results_dir)
        
        # Store results
        results[model_name] = {
            'model': trained_model,
            'history': history,
            'val_metrics': metrics,
            'test_metrics': test_metrics,
            'config': config
        }
        
        # Save model
        os.makedirs('models', exist_ok=True)
        torch.save(trained_model.state_dict(), f'models/{model_name.lower()}_model.pt')
        
        # Print test results
        print(f"\nTest Results for {model_name}:")
        print(f"Accuracy: {test_metrics['acc']:.4f}")
        print(f"F1 Score: {test_metrics['f1']:.4f}")
        print(f"Precision: {test_metrics['precision']:.4f}")
        print(f"Recall: {test_metrics['recall']:.4f}")
    
    # Compare models
    compare_models(results, 'results/model_comparison')
    
    return results

if __name__ == '__main__':
    main()