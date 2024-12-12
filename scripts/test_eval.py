import torch
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from utils.counter_eval_utils import load_trained_gat_model, evaluate_single_graph, analyze_all_players
from utils.pyg_converter import create_pyg_dataset


test_data = torch.load('processed_data/men_imbalanced_test_graph_dataset.pt')

# Load the model
num_features = test_data[0].num_features
model_dir = Path('results/GAT/best_model.pt')
model = load_trained_gat_model(model_dir, num_features)

df_loc = pd.read_csv('experiments/pred_locations_29.csv')
df_actual = pd.read_parquet('processed_data/men_imbalanced_node_features_test.parquet')
df_loc_subset = df_loc[df_loc.game_id.isin([198])].copy()
# For all players:
all_results = analyze_all_players(
    df_actual, df_loc_subset, model, create_pyg_dataset, evaluate_single_graph
)
