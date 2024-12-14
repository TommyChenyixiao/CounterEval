import pandas as pd
import os
import sys
import networkx as nx
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from utils.pyg_converter import create_pyg_dataset, save_graph_list
from utils.graph_visualize import visualize_head_tail_pyg_data, save_sample_visualizations

train_val_df = pd.read_parquet('processed_data/men_imbalanced_node_features_train.parquet')
test_df = pd.read_parquet('processed_data/men_imbalanced_node_features_test.parquet')

unique_games = train_val_df['game_id'].unique()
test_games = test_df['game_id'].unique()
train_games, val_games = train_test_split(unique_games, test_size=0.17, random_state=42)
train_df = train_val_df[train_val_df['game_id'].isin(train_games)].copy()
val_df = train_val_df[train_val_df['game_id'].isin(val_games)].copy()


# Drop Playernumber labels
train_df.drop(columns=['player_num_label'], inplace=True)
val_df.drop(columns=['player_num_label'], inplace=True)
test_df.drop(columns=['player_num_label'], inplace=True)

# Now balance only the training set
train_unique_frames = train_df.drop_duplicates(['game_id', 'frame_id', 'success'])
class_counts = train_unique_frames.groupby('success').size()
min_class_count = min(class_counts)

balanced_train_frames = pd.concat([
    train_unique_frames[train_unique_frames['success'] == 0].sample(n=min_class_count, random_state=42),
    train_unique_frames[train_unique_frames['success'] == 1].sample(n=min_class_count, random_state=42)
])

# Get all rows for the balanced frame_id/game_id combinations
balanced_train_df = pd.merge(
    train_df,
    balanced_train_frames[['game_id', 'frame_id']],
    on=['game_id', 'frame_id']
)

# Print statistics
print("=== Dataset Split Statistics ===")
print("\nUnique game_ids in each split:")
print(f"Train: {len(train_games)}")
print(f"Val: {len(val_games)}")
print(f"Test: {len(test_games)}")

print("\n=== Original Training Set ===")
print(train_df.drop_duplicates(['game_id', 'frame_id', 'success']).groupby('success').size())

print("\n=== Balanced Training Set ===")
print(balanced_train_df.drop_duplicates(['game_id', 'frame_id', 'success']).groupby('success').size())

print("\n=== Validation Set ===")
print(val_df.drop_duplicates(['game_id', 'frame_id', 'success']).groupby('success').size())

print("\n=== Test Set ===")
print(test_df.drop_duplicates(['game_id', 'frame_id', 'success']).groupby('success').size())

# Create PyG datasets
train_pyg_data_list = create_pyg_dataset(balanced_train_df)
val_pyg_data_list = create_pyg_dataset(val_df)
test_pyg_data_list = create_pyg_dataset(test_df)

# Call the visualization function for training set
print("\nVisualizing training set graphs:")
visualize_head_tail_pyg_data(train_pyg_data_list)
save_sample_visualizations(train_pyg_data_list, save_dir='experiments', num_samples=1)

# Save the PyG datasets for each split
save_graph_list(train_pyg_data_list, 'processed_data', "men_balanced_train_graph_dataset")
save_graph_list(val_pyg_data_list, 'processed_data', "men_imbalanced_val_graph_dataset")
save_graph_list(test_pyg_data_list, 'processed_data', "men_imbalanced_test_graph_dataset")

# Print sizes of each split
print(f"\nSaved {len(train_pyg_data_list)} training graphs")
print(f"Saved {len(val_pyg_data_list)} validation graphs")
print(f"Saved {len(test_pyg_data_list)} test graphs")