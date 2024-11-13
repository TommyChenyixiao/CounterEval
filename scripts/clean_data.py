"""
Data Cleaning
=============
We found several problems with the original dataset. We sincerely wish that people treat their papers as products with quality assurance.

1. There are frames with less than 23 nodes (22 players + 1 ball) due to red cards or other reasons. This causes the number of edges from 
edge lists and edge features to be different. We will exclude these frames.

2. The team_flag of ball nodes is the same as the defending team. We will change the team_flag of ball nodes to -1. This is finished in convert_data.py.

3. In full frames data, we found that one game may contain multiple counterattack samples. We use success to further distinguish them.

4. In full frames data, we found that the team with team_flag == 1 may actually be the defending team. We relabel the team_flag 
by evaluating the position of goalkeepers and the average direction of the players. 

5. In full frames data, we found that some samples with success == 0 are actually successful counterattacks. 

6. There is no id for players it is hard to analyze the movement of players. We will add an id for each player by assuming that the order of players in the dataset
is consistent in different frames. This requires further investigation on the continuity of the position of players in different frames.

7. Due to the above problems, we think the balanced dataset is not reliable. We will regenerate a balanced dataset from the cleaned full frames data.
"""

import sys
import os
import h5py
import pandas as pd
import numpy as np
from loguru import logger

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.helper import h5_to_dict

def exclude_red_card_data(filename):
    """
    We only consider frames with 23 nodes (22 players + 1 ball)
    """
    if "imbalanced" in filename:
        with h5py.File(f"processed_data/{filename}_edge_lists.h5", "r") as f:
            edge_lists = h5_to_dict(f)
        with h5py.File(f"processed_data/{filename}_edge_features.h5", "r") as f:
            edge_features = h5_to_dict(f)
    else:
        edge_lists_dict = {}
        edge_features_dict = {}
        for adj in ["normal", "dense", "delaunay"]:
            with h5py.File(f"processed_data/{filename}_{adj}_edge_lists.h5", "r") as f:
                edge_lists_dict[adj] = h5_to_dict(f)
            with h5py.File(f"processed_data/{filename}_{adj}_edge_features.h5", "r") as f:
                edge_features_dict[adj] = h5_to_dict(f)
        edge_lists = edge_lists_dict['normal']
        edge_features = edge_features_dict['normal']
    
    with h5py.File(f"processed_data/{filename}_node_features.h5", "r") as f:
        node_features = h5_to_dict(f)
    
    red_card_game_number = 0
    keys_to_delete = []
    for k in edge_lists.keys():
        if "imbalanced" in filename:
            game_id, success = k.split("_")[-2], k.split("_")[-1]
            suffix = f"{game_id}_{success}"
        else:
            suffix = k.split("_")[-1]
        if edge_features[f"edge_features_{suffix}"].shape[1] != edge_lists[k].shape[1] \
            or node_features[f"node_features_{suffix}"].shape[1] != 23:
            red_card_game_number += 1
            keys_to_delete.append(k)
    
    df = pd.read_parquet(f"processed_data/{filename}_node_features.parquet")
    for k in keys_to_delete:
        if "imbalanced" in filename:
            game_id, success = k.split("_")[-2], k.split("_")[-1]
            suffix = f"{game_id}_{success}"
        else:
            suffix = k.split("_")[-1]
        
        if "imbalanced" in filename:
            del edge_lists[k]
            del edge_features[f"edge_features_{suffix}"]
            df = df.loc[(df.loc[:, "game_id"] != int(game_id)) | (df.loc[:, "success"] != int(success)), :]
        else:
            for adj in ["normal", "dense", "delaunay"]:
                del edge_lists_dict[adj][k]
                del edge_features_dict[adj][f"edge_features_{suffix}"]
                df = df.loc[df.loc[:, "frame_id"] != int(suffix), :]
        del node_features[f"node_features_{suffix}"]
    
    df.to_parquet(f"processed_data/{filename}_node_features.parquet", index=False)

    logger.info(f"{filename} - Number of games with red card: {red_card_game_number}")
    if "imbalanced" in filename:
        with h5py.File(f"processed_data/{filename}_edge_lists.h5", "w") as f:
            for key, arr in edge_lists.items():
                f.create_dataset(key, data=arr)
        with h5py.File(f"processed_data/{filename}_edge_features.h5", "w") as f:
            for key, arr in edge_features.items():
                f.create_dataset(key, data=arr)
    else:
        for adj in ["normal", "dense", "delaunay"]:
            with h5py.File(f"processed_data/{filename}_{adj}_edge_lists.h5", "w") as f:
                for key, arr in edge_lists_dict[adj].items():
                    f.create_dataset(key, data=arr)
            with h5py.File(f"processed_data/{filename}_{adj}_edge_features.h5", "w") as f:
                for key, arr in edge_features_dict[adj].items():
                    f.create_dataset(key, data=arr)
    with h5py.File(f"processed_data/{filename}_node_features.h5", "w") as f:
        for key, arr in node_features.items():
            f.create_dataset(key, data=arr)
    
def relabel_attack_team(filename):
    """
    We found that the team with team_flag == 1 may actually be the defending team. We relabel the team_flag
    by evaluating the position of goalkeepers and the average direction of the players. This is only applicable for full frames data.
    """
    df = pd.read_parquet(f"processed_data/{filename}_node_features.parquet")
    df.loc[:, "dist_to_left_boundary"] = df.loc[:, "x"]
    df.loc[:, "dist_to_right_boundary"] = 1 - df.loc[:, "x"]

    team_position = 




def main():
    full_filenames = ["men_imbalanced", "women_imbalanced"]
    balanced_filenames = ["men", "women", "combined"]
    for filename in full_filenames + balanced_filenames:
        exclude_red_card_data(filename)
    
if __name__ == "__main__":
    main()