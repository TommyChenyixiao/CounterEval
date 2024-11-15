"""
Data Cleaning
=============
We found several problems with the original dataset. We sincerely wish that people treat their papers as products with quality assurance.

1. There are frames with less than 23 nodes (22 players + 1 ball) due to red cards or other reasons. This causes the number of edges from 
edge lists and edge features to be different. We will exclude these frames. [SOLVED]

2. The team_flag of ball nodes is the same as the defending team. We will change the team_flag of ball nodes to -1. This is finished in convert_data.py. [SOLVED]

3. In full frames data, we found that one game may contain multiple counterattack samples. We use success to further distinguish them. [SOLVED]

4. In full frames data, we found that the team with team_flag == 1 may actually be the defending team. We relabel the team_flag 
by evaluating the position of goalkeepers and the average direction of the players. [SOLVED]

5. In full frames data, we found that some samples with success == 0 are actually successful counterattacks. 

6. There is no id for players it is hard to analyze the movement of players. We will add an id for each player by assuming that the order of players in the dataset
is consistent in different frames. This requires further investigation on the continuity of the position of players in different frames.

7. Some games have too few frames to be considered as a counterattack. We may exclude these games.

8. Due to the above problems, we think the balanced dataset is not reliable. We will regenerate a balanced dataset from the cleaned full frames data.
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
    number_of_games = df.loc[:, "game_id"].nunique()
    logger.info(f"{filename} - Before relabeling attack team - Number of games: {number_of_games}")
    
    # Determine the location of team based on the position of goalkeepers
    # Basically, we assume that the team with the goalkeeper closer to the left boundary is the left team
    # We will calculate the average distance of players to the left and right boundaries
    team_position = (
        df.groupby(['game_id', 'frame_id', 'success', 'att_team'])
        .agg(
            min_dist_to_left_boundary=('dist_to_left_boundary', 'min'),
            min_dist_to_right_boundary=('dist_to_right_boundary', 'min')
        )
        .reset_index()
        .groupby(['game_id', 'success', 'att_team'])
        .agg(
            avg_min_dist_to_left_boundary=('min_dist_to_left_boundary', 'mean'),
            avg_min_dist_to_right_boundary=('min_dist_to_right_boundary', 'mean')
        )
        .reset_index()
        .query("att_team != -1")
        .pivot(index=['game_id', 'success'], columns='att_team', 
            values=['avg_min_dist_to_left_boundary', 'avg_min_dist_to_right_boundary'])
    )
    team_position.columns = [
        f"{val}_{int(flag)}" for val, flag in team_position.columns
    ]
    team_position = team_position.reset_index()
    team_position['left_diff'] = (
        team_position['avg_min_dist_to_left_boundary_0'] - team_position['avg_min_dist_to_left_boundary_1']
    )
    team_position['right_diff'] = (
        team_position['avg_min_dist_to_right_boundary_0'] - team_position['avg_min_dist_to_right_boundary_1']
    )
    team_position['left_team'] = team_position.apply(
        lambda row: (
            1 if abs(row['left_diff']) > abs(row['right_diff']) and row['left_diff'] > 0 else
            0 if abs(row['left_diff']) > abs(row['right_diff']) else
            0 if row['right_diff'] > 0 else 1
        ),
        axis=1
    )
    
    # The direction of the team is determined by the average direction of the players
    # If the average direction of the two teams are in the same direction, we will include this game
    game_direction = (
        df.loc[df.loc[:, "att_team"] != -1, :].groupby(['game_id', 'success', 'att_team']).agg(
            avg_vx=('vx', 'mean')
        ).reset_index()
        .pivot(index=['game_id', 'success'], columns='att_team', values='avg_vx')
    )
    game_direction.columns = [
        f"avg_vx_{int(flag)}" for flag in game_direction.columns
    ]
    game_direction = game_direction.reset_index()
    # Filter those games with avg_vx_0 * avg_vx_1 < 0
    game_direction = game_direction.query("avg_vx_0 * avg_vx_1 > 0")
    
    # Merge the two dataframes
    # If the average direction is to the right, then the left team is the attacking team
    indicative_team = pd.merge(team_position, game_direction, on=['game_id', 'success'])
    indicative_team['indicative_att_team'] = indicative_team.apply(
        lambda row: row['left_team'] if row['avg_vx_0'] > 0 else 1 - row['left_team'],
        axis=1
    )
    indicative_team = indicative_team.loc[:, ['game_id', 'success', 'indicative_att_team']]

    relabeled_df = pd.merge(df, indicative_team, on=['game_id', 'success'], how='inner')
    relabeled_df.loc[(relabeled_df.loc[:, "indicative_att_team"] == 0) & (relabeled_df.loc[:, "att_team"] != -1), "att_team"] = 1 - \
        relabeled_df.loc[(relabeled_df.loc[:, "indicative_att_team"] == 0) & (relabeled_df.loc[:, "att_team"] != -1), "att_team"]
    relabeled_df.drop(columns=['indicative_att_team'], inplace=True)
    relabeled_df.loc[:, "game_id"] = relabeled_df.loc[:, "game_id"].astype(int)
    relabeled_df.loc[:, "success"] = relabeled_df.loc[:, "success"].astype(int)

    number_of_games = relabeled_df.loc[:, "game_id"].nunique()
    logger.info(f"{filename} - After relabeling attack team - Number of games after relabeling: {number_of_games}")
    
    # h5 data also need to be updated
    with h5py.File(f"processed_data/{filename}_node_features.h5", "r") as f:
        node_features = h5_to_dict(f)
    with h5py.File(f"processed_data/{filename}_edge_features.h5", "r") as f:
        edge_features = h5_to_dict(f)
    with h5py.File(f"processed_data/{filename}_edge_lists.h5", "r") as f:
        edge_lists = h5_to_dict(f)
    
    keys_to_delete = []
    for key, arr in node_features.items():
        game_id, success = key.split("_")[-2], key.split("_")[-1]
        suffix = f"{game_id}_{success}"
        tmp_df = indicative_team.loc[(indicative_team.loc[:, "game_id"] == int(game_id)) & (indicative_team.loc[:, "success"] == int(success)), :]
        if tmp_df.shape[0] == 0:
            keys_to_delete.append(f"node_features_{suffix}")
        else:
            assert tmp_df.shape[0] == 1, "Multiple rows found"
            att_team = tmp_df.loc[:, "indicative_att_team"].iloc[0]
            if att_team == 0:
                mask = (arr[:, :, 10] != -1)
                node_features[f"node_features_{suffix}"][mask, 10] = 1 - node_features[f"node_features_{suffix}"][mask, 10]
    for key in keys_to_delete:
        game_id, success = key.split("_")[-2], key.split("_")[-1]
        suffix = f"{game_id}_{success}"
        del node_features[key]
        del edge_features[f"edge_features_{suffix}"]
        del edge_lists[f"edge_list_{suffix}"]

    with h5py.File(f"processed_data/{filename}_node_features.h5", "w") as f:
        for key, arr in node_features.items():
            f.create_dataset(key, data=arr)
    with h5py.File(f"processed_data/{filename}_edge_features.h5", "w") as f:
        for key, arr in edge_features.items():
            f.create_dataset(key, data=arr)
    with h5py.File(f"processed_data/{filename}_edge_lists.h5", "w") as f:
        for key, arr in edge_lists.items():
            f.create_dataset(key, data=arr)
    
    relabeled_df.to_parquet(f"processed_data/{filename}_node_features.parquet", index=False)

def relabel_success(filename):
    """
    We found that there are some samples labeled with unsuccessful counterattack actually are successful counterattacks 
    i.e. the offensive team successfully brings the ball to the opponent's penalty area. We use the last frame of the
    full frame data to determine the success of the counterattack.
    """
    field_length, field_width = 105, 68
    penalty_length, penalty_width = 16.5, 40.3

    df = pd.read_parquet(f"processed_data/{filename}_node_features.parquet")
    # Assume that we have fixed the team_flag of the ball nodes
    # Game direction
    game_direction = (
        df.loc[df.loc[:, "att_team"] != -1, :].groupby(['game_id']).agg(
            avg_vx=('vx', 'mean')
        ).reset_index()
    )
    game_direction = game_direction.reset_index()
    game_direction.loc[:, "game_direction"] = np.sign(game_direction.loc[:, "avg_vx"]) # +1 => right, -1 => left

    # First frame when the ball is within the opponent's penalty area
    #NOTE We also try the last frame, but given the goal of understanding whether the counterattack can
    # pass the ball into the opponent's penalty area i.e. the event within the penalty area is not considered,
    # we think as long as there is one (two? three?) frame that the ball is within the opponent's penalty area, the counterattack is successful
    frame_df = df.copy(deep=True)

    ball_df = frame_df.loc[frame_df.loc[:, "att_team"] == -1, :]
    ball_df.loc[:, "x_actual"] = ball_df.loc[:, "x"] * field_length
    ball_df.loc[:, "y_actual"] = ball_df.loc[:, "y"] * field_width
    ball_df = pd.merge(ball_df, game_direction, on=['game_id'], how='inner')
    # Determine whether the ball is in the opponent's penalty area
    ball_df.loc[:, "in_penalty_area"] = (
        (ball_df.loc[:, "game_direction"] == 1) & 
        (ball_df.loc[:, "x_actual"] >= field_length - penalty_length) &
        (ball_df.loc[:, "y_actual"] >= (field_width - penalty_width) / 2) &
        (ball_df.loc[:, "y_actual"] <= (field_width + penalty_width) / 2)
    ) | (
        (ball_df.loc[:, "game_direction"] == -1) & 
        (ball_df.loc[:, "x_actual"] <= penalty_length) &
        (ball_df.loc[:, "y_actual"] >= (field_width - penalty_width) / 2) &
        (ball_df.loc[:, "y_actual"] <= (field_width + penalty_width) / 2)
    )
    # Determine whether the ball is controlled by the attacking team
    players_df = frame_df.loc[frame_df.loc[:, "att_team"] != -1, :]
    players_df = players_df.groupby(['game_id', 'frame_id', 'att_team']).agg(
        min_dist_to_ball=('dist_ball', 'min')
    ).reset_index().pivot(index=['game_id', 'frame_id'], columns='att_team', values='min_dist_to_ball')
    players_df.columns = [f"min_dist_to_ball_{int(flag)}" for flag in players_df.columns]
    players_df = players_df.reset_index()
    players_df.loc[:, "controlled_by_att_team"] = (
        players_df.loc[:, "min_dist_to_ball_0"] > players_df.loc[:, "min_dist_to_ball_1"]
    )
    # AND the two conditions
    assert ball_df.loc[:, "game_id"].nunique() == players_df.loc[:, "game_id"].nunique()
    condition_df = pd.merge(ball_df, players_df, on=['game_id', 'frame_id'], how='inner')
    assert condition_df.loc[:, "game_id"].nunique() == df.loc[:, "game_id"].nunique()

    # As long as there is one frame that the ball is in the opponent's penalty area and controlled by the attacking team, the counterattack is successful
    condition_df.loc[:, "indicative_success"] = condition_df.apply(
        lambda row: 1 if row['in_penalty_area'] and row['controlled_by_att_team'] else 0,
        axis=1
    )
    condition_df = condition_df.loc[:, ['game_id', 'frame_id', 'indicative_success']]
    condition_df = condition_df.groupby(['game_id']).agg(
        indicative_success=('indicative_success', 'max')
    ).reset_index()

    # Assert that after the above process, the number of games is the same
    assert condition_df.loc[:, "game_id"].nunique() == df.loc[:, "game_id"].nunique()
    relabeled_df = pd.merge(df, condition_df, on=['game_id'], how='inner')
    relabeled_df.loc[:, "success"] = relabeled_df.loc[:, "indicative_success"]

    ## TEST first....


def main():
    full_filenames = ["men_imbalanced", "women_imbalanced"]
    balanced_filenames = ["men", "women", "combined"]
    # balanced_filenames = []
    # for filename in full_filenames + balanced_filenames:
    #     exclude_red_card_data(filename)
    
    # for filename in full_filenames:
    #     relabel_attack_team(filename)

    for filename in full_filenames:
        relabel_success(filename)
    
if __name__ == "__main__":
    main()