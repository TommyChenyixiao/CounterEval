import torch
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv, GATConv, SAGEConv, TransformerConv,
    global_mean_pool, global_add_pool, global_max_pool
)
from torch.nn import Linear, BatchNorm1d, ModuleList

class BaseGNN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_layers, dropout=0.5, pool_type='mean'):
        super().__init__()
        self.dropout = dropout

        # Configure pooling function
        pooling_functions = {
            'mean': global_mean_pool,
            'add': global_add_pool,
            'max': global_max_pool
        }
        if pool_type not in pooling_functions:
            raise ValueError(f"Unsupported pooling type: {pool_type}. Choose from {list(pooling_functions.keys())}")
        self.pool = pooling_functions[pool_type]
        
        # Batch normalization for GNN layers
        self.gnn_batch_norms = ModuleList([
            BatchNorm1d(hidden_channels) for _ in range(num_layers)
        ])

        # Three-layer MLP with batch normalization
        self.mlp_layers = ModuleList([
            Linear(hidden_channels, hidden_channels),
            Linear(hidden_channels, hidden_channels // 2),
            Linear(hidden_channels // 2, 1)
        ])
        
        self.mlp_batch_norms = ModuleList([
            BatchNorm1d(hidden_channels),
            BatchNorm1d(hidden_channels // 2)
        ])

    def mlp_forward(self, x):
        # First MLP layer
        x = self.mlp_layers[0](x)
        x = self.mlp_batch_norms[0](x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Second MLP layer
        x = self.mlp_layers[1](x)
        x = self.mlp_batch_norms[1](x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Output layer
        return self.mlp_layers[2](x)

class GCN(BaseGNN):
    def __init__(self, num_node_features, hidden_channels=64, num_layers=3, dropout=0.5, pool_type='mean'):
        super().__init__(num_node_features, hidden_channels, num_layers, dropout, pool_type)
        
        self.convs = ModuleList()
        self.convs.append(GCNConv(num_node_features, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

    def forward(self, x, edge_index, batch):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.gnn_batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.pool(x, batch)
        return self.mlp_forward(x)

class GAT(BaseGNN):
    def __init__(self, num_node_features, hidden_channels=32, num_layers=2, heads=4, dropout=0.5, pool_type='mean'):
        super().__init__(hidden_channels * heads, hidden_channels * heads, num_layers, dropout, pool_type)
        
        self.convs = ModuleList()
        self.convs.append(GATConv(num_node_features, hidden_channels, heads=heads, dropout=dropout))
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout))

    def forward(self, x, edge_index, batch):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.gnn_batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.pool(x, batch)
        return self.mlp_forward(x)

class GraphSAGE(BaseGNN):
    def __init__(self, num_node_features, hidden_channels=64, num_layers=3, dropout=0.5, pool_type='max'):
        super().__init__(num_node_features, hidden_channels, num_layers, dropout, pool_type)
        
        self.convs = ModuleList()
        self.convs.append(SAGEConv(num_node_features, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))

    def forward(self, x, edge_index, batch):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.gnn_batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.pool(x, batch)
        return self.mlp_forward(x)

class GraphTransformer(BaseGNN):
    def __init__(self, num_node_features, hidden_channels=32, num_layers=2, heads=4, dropout=0.5, pool_type='mean'):
        super().__init__(hidden_channels * heads, hidden_channels * heads, num_layers, dropout, pool_type)
        
        self.convs = ModuleList()
        self.convs.append(TransformerConv(
            num_node_features, hidden_channels,
            heads=heads, dropout=dropout, edge_dim=None
        ))
        for _ in range(num_layers - 1):
            self.convs.append(TransformerConv(
                hidden_channels * heads, hidden_channels,
                heads=heads, dropout=dropout, edge_dim=None
            ))

    def forward(self, x, edge_index, batch):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.gnn_batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.pool(x, batch)
        return self.mlp_forward(x)

def create_gnn_model(model_type, num_features, **kwargs):
    """Creates a GNN model of the specified type with given parameters."""
    models = {
        'GCN': GCN,
        'GAT': GAT,
        'GraphSAGE': GraphSAGE,
        'Transformer': GraphTransformer
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(models.keys())}")
    
    return models[model_type](num_features, **kwargs)