import logging
from datasets import load_data
from datasets import preprocess_pyg
import sys
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, roc_curve
import numpy as np
import matplotlib.pyplot as plt

# Setting up the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
stdout_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stdout_handler)

# Load the data
data = load_data.get_data('./datasets/women.pkl', local=True)
filtered_data = preprocess_pyg.filter_features(data.copy())
data_list = preprocess_pyg.process_data(filtered_data)

# Hyperparameters
learning_rate = 1e-3  # Learning rate
epochs = 150  # Number of training epochs
batch_size = 16  # Batch size
channels = 128  # Hidden units for the neural network
layers = 3  # Number of GCN layers

# Train/validation/test split
train_size = int(0.7 * len(data_list))
val_size = int(0.15 * len(data_list))
test_size = len(data_list) - train_size - val_size

train_data, val_data, test_data = torch.utils.data.random_split(data_list, [train_size, val_size, test_size])

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)

class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=layers):
        super(GNN, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.dropout = torch.nn.Dropout(0.5)
        self.lin1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        x = global_mean_pool(x, batch)
        x = self.dropout(x)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x

# Model, optimizer, and loss function
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GNN(in_channels=data_list[0].num_node_features,
            hidden_channels=channels,
            out_channels=1,
            num_layers=layers).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.BCEWithLogitsLoss()

def train():
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        # Forward pass
        out = model(batch.x, batch.edge_index, batch.batch)

        # Align target shape with the output
        loss = criterion(out.squeeze(), batch.y.view(-1))

        # Backward pass
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs

    return total_loss / len(train_loader.dataset)

def compute_metrics(y_true, y_pred, threshold=0.5):
    """
    Compute evaluation metrics based on predictions.
    """
    y_pred_label = (y_pred >= threshold).astype(int)

    accuracy = accuracy_score(y_true, y_pred_label)
    precision = precision_score(y_true, y_pred_label)
    recall = recall_score(y_true, y_pred_label)
    f1 = f1_score(y_true, y_pred_label)

    return accuracy, precision, recall, f1

def evaluate(loader, threshold=0.5):
    model.eval()
    y_true = []
    y_pred = []
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out.squeeze(), batch.y.view(-1))
            total_loss += loss.item() * batch.num_graphs

            y_true.append(batch.y.cpu().numpy())
            y_pred.append(out.sigmoid().cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    auc_score = roc_auc_score(y_true, y_pred)

    # Compute additional metrics
    accuracy, precision, recall, f1 = compute_metrics(y_true, y_pred, threshold)

    return total_loss / len(loader.dataset), auc_score, accuracy, precision, recall, f1, y_true, y_pred

# Training loop with validation
for epoch in range(1, epochs + 1):
    train_loss = train()

    if epoch % 10 == 0:
        val_loss, val_auc, val_accuracy, val_precision, val_recall, val_f1, _, _ = evaluate(val_loader)
        logger.info(
            f"Epoch: {epoch:03d}, "
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Val AUC: {val_auc:.4f}, "
            f"Val Acc: {val_accuracy:.4f}, "
            f"Val Prec: {val_precision:.4f}, "
            f"Val Rec: {val_recall:.4f}, "
            f"Val F1: {val_f1:.4f}"
        )

# Final evaluation on the test set
test_loss, test_auc, test_accuracy, test_precision, test_recall, test_f1, y_true, y_pred = evaluate(test_loader)
fpr, tpr, _ = roc_curve(y_true, y_pred)

# Print test results
logger.info(
    f"Test Loss: {test_loss:.4f}, "
    f"Test AUC: {test_auc:.4f}, "
    f"Test Acc: {test_accuracy:.4f}, "
    f"Test Prec: {test_precision:.4f}, "
    f"Test Rec: {test_recall:.4f}, "
    f"Test F1: {test_f1:.4f}"
)

# Plot ROC Curve
plt.figure()
plt.plot(fpr, tpr, color='b', label=f'AUC = {test_auc:.2f}')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (Test Set)')
plt.legend(loc='lower right')
plt.show()