import torch
from torch_geometric.data import Data, Dataset
from pathlib import Path
from typing import List
import pandas as pd
import numpy as np

class SoccerDataset(Dataset):
    def __init__(self, df):
        super().__init__()
        # Group by game_id and frame_id to get unique graphs
        self.grouped = df.groupby(['game_id', 'frame_id'])
        self.graph_keys = list(self.grouped.groups.keys())
        
        # Define which columns to use as node features
        self.feature_cols = ['x', 'y', 'vx', 'vy', 'v', 'angle_v', 'dist_goal', 
                           'angle_goal', 'dist_ball', 'angle_ball', 'att_team',
                           'potential_receiver', 'sin_ax', 'cos_ay', 'a', 'sin_a', 
                           'cos_a', 'dist_to_left_boundary', 'dist_to_right_boundary']
        
    def len(self):
        return len(self.graph_keys)
    
    def get(self, idx):
        # Get the game_id and frame_id for this index
        game_id, frame_id = self.graph_keys[idx]
        
        # Get the data for this specific frame
        frame_data = self.grouped.get_group((game_id, frame_id))
        
        # Convert features to tensor
        x = torch.tensor(frame_data[self.feature_cols].values, dtype=torch.float)
        
        # Get the label (success) for this frame
        y = torch.tensor(frame_data['success'].iloc[0], dtype=torch.int32)
        
        # Create fully connected edges (every node connected to every other node)
        num_nodes = len(frame_data)
        edge_index = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:  # Exclude self-loops
                    edge_index.append([i, j])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        
        # Create the PyG Data object
        data = Data(x=x, edge_index=edge_index, y=y)
        
        return data

def create_pyg_dataset(df):
    """
    Convert DataFrame to PyTorch Geometric dataset
    
    Parameters:
    df (pandas.DataFrame): Input DataFrame with required columns
    
    Returns:
    SoccerDataset: PyTorch Geometric dataset
    """
    return SoccerDataset(df)

def save_graph_list(graph_list: List[Data], save_path: str, filename: str = "graph_data") -> None:
    """
    Save a list of PyG graphs to disk.
    
    Args:
        graph_list (List[Data]): List of PyG Data objects to save
        save_path (str): Directory path to save the graphs
        filename (str): Base filename to use (default: "graph_data")
    """
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    full_path = save_dir / f"{filename}.pt"
    torch.save(graph_list, str(full_path))

def load_graph_list(load_path: str, filename: str = "graph_data") -> List[Data]:
    """
    Load a list of PyG graphs from disk.
    
    Args:
        load_path (str): Directory path where graphs are saved
        filename (str): Base filename to load (default: "graph_data")
    
    Returns:
        List[Data]: List of PyG Data objects
    """
    load_dir = Path(load_path)
    full_path = load_dir / f"{filename}.pt"
    
    if not full_path.exists():
        raise FileNotFoundError(f"No saved graphs found at {full_path}")
    
    return torch.load(str(full_path))