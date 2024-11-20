import os
import pandas as pd
import numpy as np
import h5py
from loguru import logger
from multiprocessing import Pool

def process_one_frame(df):
    """
    Process a single frame to construct node features and edge features.
    - id
        'game_id', 'frame_id'
    - Node features
        'x', 'y', 'vx', 'vy', 'v', 'angle_v', 'dist_goal', 'angle_goal',
       'dist_ball', 'angle_ball', 'att_team', 'potential_receiver', 'sin_ax',
       'cos_ay', 'a', 'sin_a', 'cos_a', 'dist_to_left_boundary', 'dist_to_right_boundary', 'player_num_label'
    - Edge features (between each player)
        * Euclidean distance between players: 'dist'  \text{dist}(i, j) = \sqrt{(x_i - x_j)^2 + (y_i - y_j)^2} 
        * Angle between players: 'angle' \text{angle}(i, j) = \arctan2(y_j - y_i, x_j - x_i)
        * Relative velocity magnitude: 'rel_v' \text{rel\_v}(i, j) = \sqrt{(v_{x_i} - v_{x_j})^2 + (v_{y_i} - v_{y_j})^2}
        * Velocity alignment: 'v_align' vx_i * vx_j + vy_i * vy_j
        * Acceleration magnitude: 'a' \text{a}(i, j) = \sqrt{(a_{x_i} - a_{x_j})^2 + (a_{y_i} - a_{y_j})^2}
        * Edge type: 'type' 0 if players are from the same team, 1 if players are from different teams and 2 if one of the players is the ball
    - Label
        'success'

    Parameters:
        - df: A pandas DataFrame containing player tracking data for a single frame.

    Returns:
        None but save node features, edge features, id, label to a .h5 file
    """
    # Calculate the Euclidean distance matrix between players
    distance_matrix = np.sqrt(
        (df['x'].values[:, np.newaxis] - df['x'].values) ** 2 +
        (df['y'].values[:, np.newaxis] - df['y'].values) ** 2
    )

    # Calculate the angle matrix between players
    angle_matrix = np.arctan2(
        df['y'].values[:, np.newaxis] - df['y'].values,
        df['x'].values[:, np.newaxis] - df['x'].values
    )

    # Calculate the relative velocity magnitude matrix between players
    rel_v_matrix = np.sqrt(
        (df['vx'].values[:, np.newaxis] - df['vx'].values) ** 2 +
        (df['vy'].values[:, np.newaxis] - df['vy'].values) ** 2
    )

    # Calculate the velocity alignment matrix between players
    v_align_matrix = df['vx'].values[:, np.newaxis] * df['vx'].values + df['vy'].values[:, np.newaxis] * df['vy'].values

    # Calculate the acceleration magnitude matrix between players
    a_matrix = np.abs(
        df['a'].values[:, np.newaxis] - df['a'].values
    )

    # Create a matrix to store the edge type
    type_matrix = np.zeros_like(distance_matrix)

    # Assign edge types
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            if i == j:
                type_matrix[i, j] = 0
            elif df['att_team'].iloc[i] == df['att_team'].iloc[j]:
                type_matrix[i, j] = 0
            elif df['att_team'].iloc[i] == -1 or df['att_team'].iloc[j] == -1:
                type_matrix[i, j] = 2
            else:
                type_matrix[i, j] = 1

    # Save the node features, edge features, id, and label to a .h5 file
    h5_filename = f'processed_data/features/{df["game_id"].iloc[0]}_{df["frame_id"].iloc[0]}.h5'
    with h5py.File(h5_filename, 'w') as f:
        f.create_dataset('id', data=df[['game_id', 'frame_id']])
        f.create_dataset('node_features', data=df.drop(columns=['game_id', 'frame_id', 'success']))
        f.create_dataset('edge_features', data=np.stack([distance_matrix, angle_matrix, rel_v_matrix, v_align_matrix, a_matrix, type_matrix], axis=-1))
        f.create_dataset('label', data=df['success'])
    
    logger.info(f"Processed frame {df['frame_id'].iloc[0]} for game {df['game_id'].iloc[0]}")

def construct_edges(filename):
    df = pd.read_parquet(filename)
    

    # Create a list of DataFrames, where each dataframe corresponding to one game_id and one frame_id
    frames = []
    for game_id in df['game_id'].unique()[:10]:
        for frame_id in df[df['game_id'] == game_id]['frame_id'].unique():
            frames.append(df[(df['game_id'] == game_id) & (df['frame_id'] == frame_id)])
    
    logger.info(f"Processing {len(frames)} frames")

    # Process each frame in parallel
    with Pool() as pool:
        frames = pool.map(process_one_frame, frames)


def main():
    logger.add('edge_construction.log')
    os.makedirs('processed_data/features', exist_ok=True)

    # Construct edges for the training data
    construct_edges('processed_data/men_imbalanced_node_features_numbered.parquet')

if __name__ == '__main__':
    main()