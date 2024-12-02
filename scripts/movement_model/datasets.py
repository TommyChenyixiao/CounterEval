import torch
from torch.utils.data import Dataset

class MovementDataset(Dataset):

    def __init__(self, Xs, ys, split='train'):
        self.Xs = Xs
        self.ys = ys
        self.split = split
    
    def __len__(self):
        return len(self.Xs)
    
    def __getitem__(self, idx):
        # X shape: (n_features, window_size, n_players)
        # y shape: (2, n_players)
        X = torch.tensor(self.Xs[idx], dtype=torch.float32)
        y = torch.tensor(self.ys[idx], dtype=torch.float32)
        return X, y