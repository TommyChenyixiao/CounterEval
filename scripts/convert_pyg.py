import pandas as pd
import os
import sys
import networkx as nx
import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.model_selection import train_test_split

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from utils.pyg_converter import create_pyg_dataset, save_graph_list
from utils.graph_visualize import visualize_head_tail_pyg_data, save_sample_visualizations

df = pd.read_parquet('processed_data/men_imbalanced_node_features_checked.parquet')
# reprocessed some features
# ['x', 'y', 'vx', 'vy', 'v', 'dist_goal', 
                        #    'angle_goal', 'dist_ball', 'angle_ball', 'att_team',
                        #    'potential_receiver', 'sin_ax', 'cos_ay', 'a', 'sin_a', 
                        #    'cos_a', 'dist_to_left_boundary', 'dist_to_right_boundary']
df_group_by_team = df.loc[df["att_team"] != -1].groupby(['game_id', 'frame_id', 'att_team']).agg({'x': ['min', 'max']})
df_group_by_team.columns = ['min_x', 'max_x']
df_group_by_team.reset_index(inplace=True)
pivot_df = df_group_by_team.pivot(index=['game_id', 'frame_id'], columns='att_team', values=['min_x', 'max_x'])
pivot_df.columns = [f'{col[0]}_{int(col[1])}' for col in pivot_df.columns]
pivot_df.reset_index(inplace=True)
df = pd.merge(df, pivot_df, on=['game_id', 'frame_id'])
df.loc[:, 'team_direction'] = None
df.loc[(df.loc[:, "att_team"] == 0) & (df.loc[:, "min_x_0"] < df.loc[:, "min_x_1"]) & (df.loc[:, "max_x_0"] < df.loc[:, "max_x_1"]), 'team_direction'] = 1
df.loc[(df.loc[:, "att_team"] == 1) & (df.loc[:, "min_x_1"] < df.loc[:, "min_x_0"]) & (df.loc[:, "max_x_1"] < df.loc[:, "max_x_0"]), 'team_direction'] = 1
df.loc[(df.loc[:, "att_team"] == 0) & (df.loc[:, "min_x_0"] > df.loc[:, "min_x_1"]) & (df.loc[:, "max_x_0"] > df.loc[:, "max_x_1"]), 'team_direction'] = -1
df.loc[(df.loc[:, "att_team"] == 1) & (df.loc[:, "min_x_1"] > df.loc[:, "min_x_0"]) & (df.loc[:, "max_x_1"] > df.loc[:, "max_x_0"]), 'team_direction'] = -1
df.loc[(df.loc[:, "att_team"] == -1) & (df.loc[:, "min_x_1"] > df.loc[:, "min_x_0"]) & (df.loc[:, "max_x_1"] > df.loc[:, "max_x_0"]), 'team_direction'] = -1
df.loc[(df.loc[:, "att_team"] == -1) & (df.loc[:, "min_x_0"] > df.loc[:, "min_x_1"]) & (df.loc[:, "max_x_0"] > df.loc[:, "max_x_1"]), 'team_direction'] = 1

df.loc[(df.loc[:, "team_direction"] == 1), "dist_goal"] = np.sqrt((df.loc[(df.loc[:, "team_direction"] == 1), "x"] - 1)**2 + (df.loc[(df.loc[:, "team_direction"] == 1), "y"] - 0.5)**2)
df.loc[(df.loc[:, "team_direction"] == -1), "dist_goal"] = np.sqrt((df.loc[(df.loc[:, "team_direction"] == -1), "x"] - 0)**2 + (df.loc[(df.loc[:, "team_direction"] == -1), "y"] - 0.5)**2)
df.loc[(df.loc[:, "team_direction"] == 1), "angle_goal"] = np.arctan2(0.5 - df.loc[(df.loc[:, "team_direction"] == 1), "y"], 1 - df.loc[(df.loc[:, "team_direction"] == 1), "x"])
df.loc[(df.loc[:, "team_direction"] == -1), "angle_goal"] = np.arctan2(0.5 - df.loc[(df.loc[:, "team_direction"] == -1), "y"], 0 - df.loc[(df.loc[:, "team_direction"] == -1), "x"])

ball_df = df[df['att_team'] == -1].drop_duplicates(['game_id', 'frame_id'])
ball_df = ball_df[['game_id', 'frame_id', 'x', 'y']]
ball_df.columns = ['game_id', 'frame_id', 'ball_x', 'ball_y']
df = pd.merge(df, ball_df, on=['game_id', 'frame_id'])
df.loc[:, 'dist_ball'] = np.sqrt((df.loc[:, 'x'] - df.loc[:, 'ball_x'])**2 + (df.loc[:, 'y'] - df.loc[:, 'ball_y'])**2)
df.loc[:, 'angle_ball'] = np.arctan2(df.loc[:, 'ball_y'] - df.loc[:, 'y'], df.loc[:, 'ball_x'] - df.loc[:, 'x'])

df.drop(columns=['min_x_0', 'min_x_1', 'max_x_0', 'max_x_1'], inplace=True)

# First get unique game_ids
unique_games = df['game_id'].unique()

# Split game_ids into train (70%), validation (15%), and test (15%)
train_games, temp_games = train_test_split(unique_games, test_size=0.3, random_state=42)
val_games, test_games = train_test_split(temp_games, test_size=0.5, random_state=42)

# Split the original dataframe based on game_ids
train_df = df[df['game_id'].isin(train_games)].copy()
val_df = df[df['game_id'].isin(val_games)].copy()
test_df = df[df['game_id'].isin(test_games)].copy()

# Combine the training and validation sets
combined_train_val_df = pd.concat([train_df, val_df])

# output the parquet files
combined_train_val_df.to_parquet('processed_data/men_imbalanced_node_features_train.parquet')
test_df.to_parquet('processed_data/men_imbalanced_node_features_test.parquet')

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