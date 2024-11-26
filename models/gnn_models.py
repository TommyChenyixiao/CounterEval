import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool, global_add_pool, global_max_pool
from torch.nn import Linear, ReLU, BatchNorm1d, Dropout, ModuleList

class BaseGNN(torch.nn.Module):
    """Base class for GNN models"""
    def __init__(self, num_node_features, hidden_channels, num_layers, dropout=0.5, 
                 pool_type='mean', num_classes=1):
        super().__init__()
        self.dropout = dropout
        
        # Set pooling function
        self.pool_type = pool_type
        if pool_type == 'mean':
            self.pool = global_mean_pool
        elif pool_type == 'add':
            self.pool = global_add_pool
        elif pool_type == 'max':
            self.pool = global_max_pool
        else:
            raise ValueError(f"Unsupported pooling type: {pool_type}")
        
        # Batch normalization layers
        self.batch_norms = ModuleList([
            BatchNorm1d(hidden_channels) for _ in range(num_layers)
        ])
        
        # Output MLP
        self.linear1 = Linear(hidden_channels, hidden_channels//2)
        self.linear2 = Linear(hidden_channels//2, num_classes)

    def forward(self, x, edge_index, batch):
        raise NotImplementedError("Base class doesn't implement forward")

class GCN(BaseGNN):
    def __init__(self, num_node_features, hidden_channels=64, num_layers=2, 
                 dropout=0.5, pool_type='mean', num_classes=1):
        super().__init__(num_node_features, hidden_channels, num_layers, 
                        dropout, pool_type, num_classes)
        
        self.convs = ModuleList()
        
        # First layer
        self.convs.append(GCNConv(num_node_features, hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
    
    def forward(self, x, edge_index, batch):
        # Graph convolution layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling
        x = self.pool(x, batch)
        
        # Final MLP
        x = self.linear1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.linear2(x)
        
        return x

class GAT(BaseGNN):
    def __init__(self, num_node_features, hidden_channels=64, num_layers=2, 
                 heads=4, dropout=0.5, pool_type='mean', num_classes=1):
        # Adjust hidden_channels to account for multi-head attention
        super().__init__(hidden_channels * heads, hidden_channels * heads, 
                        num_layers, dropout, pool_type, num_classes)
        
        self.convs = ModuleList()
        self.heads = heads
        
        # First layer
        self.convs.append(GATConv(num_node_features, hidden_channels, 
                                heads=heads, dropout=dropout))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, 
                                    heads=heads, dropout=dropout))
    
    def forward(self, x, edge_index, batch):
        # Graph convolution layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling
        x = self.pool(x, batch)
        
        # Final MLP
        x = self.linear1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.linear2(x)
        
        return x

class GraphSAGE(BaseGNN):
    def __init__(self, num_node_features, hidden_channels=64, num_layers=2, 
                 dropout=0.5, pool_type='mean', num_classes=1):
        super().__init__(num_node_features, hidden_channels, num_layers, 
                        dropout, pool_type, num_classes)
        
        self.convs = ModuleList()
        
        # First layer
        self.convs.append(SAGEConv(num_node_features, hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
    
    def forward(self, x, edge_index, batch):
        # Graph convolution layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling
        x = self.pool(x, batch)
        
        # Final MLP
        x = self.linear1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.linear2(x)
        
        return x

# Example usage
def create_gnn_model(model_type, num_features, **kwargs):
    """
    Factory function to create GNN models with specific configurations
    
    Args:
        model_type (str): 'GCN', 'GAT', or 'GraphSAGE'
        num_features (int): Number of input node features
        **kwargs: Additional model parameters
    
    Returns:
        BaseGNN: Configured GNN model
    """
    models = {
        'GCN': GCN,
        'GAT': GAT,
        'GraphSAGE': GraphSAGE
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return models[model_type](num_features, **kwargs)

# Usage example:
if __name__ == "__main__":
    # Example configurations
    configs = {
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
    
    # Create models with different configurations
    num_features = 10  # Example number of features
    models = {
        name: create_gnn_model(name, num_features, **config)
        for name, config in configs.items()
    }