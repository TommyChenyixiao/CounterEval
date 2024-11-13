import sys
import os
import pytest
import h5py
import pandas as pd
import numpy as np

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.helper import h5_to_dict

def test_balanced_node_features():
    filename = ["men", "women", "combined"]
    for name in filename: 
        h5_size = 0
        with h5py.File(f"processed_data/{name}_node_features.h5", "r") as f:
            node_features = h5_to_dict(f)
            for key in node_features.keys():
                h5_size += node_features[key].shape[0] * node_features[key].shape[1]
                # No NaN values
                assert not np.sum(~np.isfinite(node_features[key][:]))
                # There is one ball (f[key][:, 10] == -1) for each frame
                assert np.sum(node_features[key][:, :, 10] == -1) == 1
        
        df = pd.read_parquet(f"processed_data/{name}_node_features.parquet")
        check_df = df.groupby(["frame_id"])["x"].count().reset_index().apply(lambda x: x["x"] == 23, axis=1)
        assert check_df.all()
        # There is one ball (f[key][:, 10] == -1) for each frame
        assert np.sum(df.loc[df["att_team"] == -1, :].groupby(["frame_id"])["x"].count() == 1)
        # The sample size of parquer and h5 should be the same
        assert df.shape[0] == h5_size


def test_balanced_edge_features_lists():
    filename = ["men", "women", "combined"]
    adj_type = ["normal", "dense", "delaunay"]
    for name in filename:
        for adj in adj_type:
            with h5py.File(f"processed_data/{name}_{adj}_edge_lists.h5", "r") as f:
                edge_lists = h5_to_dict(f)
            with h5py.File(f"processed_data/{name}_{adj}_edge_features.h5", "r") as f:
                edge_features = h5_to_dict(f)

            for k in edge_lists.keys():
                num = k.split("_")[-1]
                assert edge_features[f"edge_features_{num}"].shape[1] == edge_lists[k].shape[1] # Make sure that the number of edge features is the same as the number of edges

def test_full_node_features():
    filename = ["men_imbalanced", "women_imbalanced"]
    for name in filename: 
        h5_size = 0
        with h5py.File(f"processed_data/{name}_node_features.h5", "r") as f:
            node_features = h5_to_dict(f)
            for key in node_features.keys():
                h5_size += node_features[key].shape[0] * node_features[key].shape[1]
                # No NaN values
                assert not np.sum(~np.isfinite(node_features[key][:]))
                # There is one ball (f[key][:, 10] == -1) for each frame
                assert np.sum(node_features[key][:, :, 10] == -1) == node_features[key].shape[0]

        df = pd.read_parquet(f"processed_data/{name}_node_features.parquet")
        check_df = df.groupby(["game_id", "frame_id", "success"])["x"].count().reset_index().apply(lambda x: x["x"] == 23, axis=1)
        assert check_df.all()
        # There is one ball (f[key][:, 10] == -1) for each frame
        assert np.sum(df.loc[df["att_team"] == -1, :].groupby(["game_id", "frame_id", "success"])["x"].count() == 1)

        # The sample size of parquer and h5 should be the same
        assert df.shape[0] == h5_size
def test_full_edge_features_lists():
    filename = ["men_imbalanced", "women_imbalanced"]
    for name in filename:
        with h5py.File(f"processed_data/{name}_edge_lists.h5", "r") as f:
            edge_lists = h5_to_dict(f)
        with h5py.File(f"processed_data/{name}_edge_features.h5", "r") as f:
            edge_features = h5_to_dict(f)
        
        for k in edge_lists.keys():
            game_id, success = k.split("_")[-2], k.split("_")[-1]
            assert edge_features[f"edge_features_{game_id}_{success}"].shape[1] == edge_lists[k].shape[1]

