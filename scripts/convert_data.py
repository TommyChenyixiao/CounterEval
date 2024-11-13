"""
The original data is in the form of pickle files. We will convert them to parquet files 
so we can use them in both Python and R.
"""

import pandas as pd
import os
import pickle
import numpy as np
import h5py
from loguru import logger

def convert_data_full_frame(file):
    """
    This function will convert the data of pickle format to other formats for the full frames data.
    In full frames data, we have the following keys:
    'id', 'node_feature_names', 'edge_feature_names', 'label', 'a', 'x', 'e'

    - Node Features will be converted into both h5 and parquet format.
    - Edge Features will be converted into h5 format.
    - Adjacency Matrix will be converted into both h5.

    For h5 data, the keys will be (game_id, success) because we find that one game may contain multiple counterattack samples. The value will be a 
    3D array where the first dimension is the frame, the second dimension is the node (or edge), and the third dimension is the feature.
    """
    with open(file, 'rb') as handle:
        og_data = pickle.load(handle)
    logger.info(f"{file} - Length: {len(og_data)}")
    
    # Node Features
    df_list = []
    id_list = []
    arr_dict = {}
    for idx, array in enumerate(og_data['x']):
        ball_mask = (array[:, 8] == 0) & (array[:, 9] == 0)
        assert array[ball_mask, 10] == 0, "Ball att_team flag is not set to 0"
        array[ball_mask, 10] = -1 # Change the att_team flag of ball from 0 to -1

        df = pd.DataFrame(array, columns=og_data['node_feature_names'])
        df['game_id'] = og_data['id'][idx]
        df['frame_id'] = idx 
        df['success'] = og_data['label'][idx][0]

        df_list.append(df)
        id_list.append(og_data['id'][idx])
        game_id, success = og_data['id'][idx], og_data['label'][idx][0]
        if f"{game_id}_{success}" in arr_dict:
            arr_dict[f"{game_id}_{success}"].append(array)
        else:
            arr_dict[f"{game_id}_{success}"] = [array]
    
    final_df = pd.concat(df_list, ignore_index=True)
    final_df['new_id'] = final_df.groupby(['game_id', 'success']).ngroup()
    final_df['adjusted_frame_id'] = final_df['frame_id'] - final_df.groupby(['game_id', 'success'])['frame_id'].transform('min')
    # game_id mapping
    game_id_map = final_df[['game_id', 'success', 'new_id']].drop_duplicates()
    
    final_df['game_id'] = final_df['new_id'].astype(int)
    final_df['frame_id'] = final_df['adjusted_frame_id'].astype(int)
    final_df.drop(columns=['new_id', 'adjusted_frame_id'], inplace=True)

    filename = file.split('/')[-1][:-4]
    final_df.to_parquet(f'processed_data/{filename}_node_features.parquet', index=False)
    logger.info(f"{filename} - Node Features: {final_df.shape}")

    with h5py.File(f"processed_data/{filename}_node_features.h5", "w") as f:
        for idx, (key, arr) in enumerate(arr_dict.items()):
            arr = np.stack(arr, axis=0)
            game_id, success = key.split('_')
            new_id = game_id_map.loc[(game_id_map['game_id'] == int(game_id)) & (game_id_map['success'] == int(success)), 'new_id'].values[0]
            suffix = f"{new_id}_{success}"
            f.create_dataset(f"node_features_{suffix}", data=arr)
    logger.info(f"{filename} - Node Features: {len(arr_dict)}")

    # Adjcency Matrix
    adj_dict = {}
    for idx, array in enumerate(og_data['a']):
        game_id, success = og_data['id'][idx], og_data['label'][idx][0]
        if f"{game_id}_{success}" in adj_dict:
            continue

        a_coo = array.tocoo()
        a = np.vstack([a_coo.row, a_coo.col])
        adj_dict[f"{game_id}_{success}"] = a
    
    with h5py.File(f"processed_data/{filename}_edge_lists.h5", "w") as f:
        for idx, (key, arr) in enumerate(adj_dict.items()):
            arr = np.stack([arr.reshape((-1,2))], axis=0)
            game_id, success = key.split('_')
            new_id = game_id_map.loc[(game_id_map['game_id'] == int(game_id)) & (game_id_map['success'] == int(success)), 'new_id'].values[0]
            suffix = f"{new_id}_{success}"
            f.create_dataset(f"edge_list_{suffix}", data=arr)
    
    # Edge Features
    edge_dict = {}
    for idx, array in enumerate(og_data['e']):
        game_id, success = og_data['id'][idx], og_data['label'][idx][0]
        if f"{game_id}_{success}" in edge_dict:
            edge_dict[f"{game_id}_{success}"].append(array)
        else:
            edge_dict[f"{game_id}_{success}"] = [array]
    
    with h5py.File(f"processed_data/{filename}_edge_features.h5", "w") as f:
        for idx, (key, arr) in enumerate(edge_dict.items()):
            arr = np.stack(arr, axis=0)
            game_id, success = key.split('_')
            new_id = game_id_map.loc[(game_id_map['game_id'] == int(game_id)) & (game_id_map['success'] == int(success)), 'new_id'].values[0]
            suffix = f"{new_id}_{success}"
            f.create_dataset(f"edge_features_{suffix}", data=arr)

def convert_data_balanced(file):
    """
    This function will convert the data of pickle format to other formats for the balanced data.
    In balanced data, we have the following keys:
    'delaunay', 'normal', 'dense', 'dense_ap', 'dense_dp', 'binary'
    The first five keys indicate the different types of adjacency matrices while the last one is the label. 
    In this project, we only keep 'delaunay', 'normal', 'dense' types of adjacency matrices.
    Under each key, we have the following keys:
    'a', 'x', 'e'.
    The data is also fully shuffled and there is no id key.

    For h5 data, the keys will be (game_id, success) because we find that one game may contain multiple counterattack samples. The value will be a 
    3D array where the first dimension is the frame, the second dimension is the node (or edge), and the third dimension is the feature.
    """
    with open(file, 'rb') as handle:
        og_data = pickle.load(handle)
    logger.info(f"{file} - Length: {len(og_data)}")
    keys = ['delaunay', 'normal', 'dense']
    filename = file.split('/')[-1][:-4]

    labels = np.array(og_data['binary'])
    np.save(f"processed_data/{filename}_labels.npy", labels)

    # Node Features
    df_list = []
    arr_list = []
    for idx, array in enumerate(og_data['normal']['x']):
        ball_mask = (array[:, 8] == 0) & (array[:, 9] == 0)
        assert array[ball_mask, 10] == 0, f"Ball att_team flag is not set to 0 at pos {idx} of {filename}/{key}"
        array[ball_mask, 10] = -1 # Change the att_team flag of ball from 0 to -1

        df = pd.DataFrame(array, columns=['x', 'y', 'vx', 'vy', 'v', 'angle_v', 'dist_goal', 'angle_goal', 'dist_ball', 'angle_ball', 'att_team', 'potential_receiver'])
        df['success'] = og_data['binary'][idx][0]
        df['frame_id'] = idx

        df_list.append(df)
        arr_list.append(array)

    final_df = pd.concat(df_list, ignore_index=True)
    final_df.to_parquet(f'processed_data/{filename}_node_features.parquet', index=False)
    logger.info(f"{filename} - Node Features: {final_df.shape}")

    with h5py.File(f"processed_data/{filename}_node_features.h5", "w") as f:
        for idx, arr in enumerate(arr_list):
            arr = np.stack([arr], axis=0)
            f.create_dataset(f"node_features_{idx}", data=arr)
    logger.info(f"{filename} - Node Features: {len(arr_list)}")

    for key in keys:
        og_data_key = og_data[key]
        
        # Adjcency Matrix
        adj_list = []
        for idx, array in enumerate(og_data_key['a']):
            a_coo = array.tocoo()
            a = np.vstack([a_coo.row, a_coo.col])
            adj_list.append(a)

        with h5py.File(f"processed_data/{filename}_{key}_edge_lists.h5", "w") as f:
            for idx, arr in enumerate(adj_list):
                arr = np.stack([arr.reshape((-1,2))], axis=0)
                f.create_dataset(f"edge_list_{idx}", data=arr)
        
        # Edge Features
        edge_list = []
        for idx, array in enumerate(og_data_key['e']):
            edge_list.append(array)
        
        with h5py.File(f"processed_data/{filename}_{key}_edge_features.h5", "w") as f:
            for idx, arr in enumerate(edge_list):
                arr = np.stack([arr], axis=0)
                f.create_dataset(f"edge_features_{idx}", data=arr)

def main():
    files = os.listdir('raw_data')
    for f in files:
        if f.endswith('.pkl'):
            if "imbalanced" in f: convert_data_full_frame(f'raw_data/{f}')
            else: convert_data_balanced(f'raw_data/{f}')

if __name__ == '__main__':
    main()
