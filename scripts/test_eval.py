import torch
import os
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from utils.counter_eval_utils import load_trained_gat_model, evaluate_single_graph, analyze_all_players
from utils.pyg_converter import create_pyg_dataset, save_graph_list


if __name__ == "__main__":
    df_actual = pd.read_parquet('processed_data/men_imbalanced_node_features_test.parquet')
    test_pyg_data_list = create_pyg_dataset(df_actual)
    save_graph_list(test_pyg_data_list, 'processed_data', "men_imbalanced_test_graph_dataset")
    
    test_data = torch.load('processed_data/men_imbalanced_test_graph_dataset.pt')

    # Load the model
    num_features = test_data[0].num_features
    model_dir = Path('results/GAT/best_model.pt')
    model = load_trained_gat_model(model_dir, num_features)

    df_loc = pd.read_csv('processed_data/pred_locations_29.csv')
    df_actual = pd.read_parquet('processed_data/men_imbalanced_node_features_test.parquet')
    df_loc_subset = df_loc[df_loc.game_id.isin([198])].copy()
    # For all players:
    all_results = analyze_all_players(
        df_actual, df_loc_subset, model, create_pyg_dataset, evaluate_single_graph
    )

    # Save the results (dictinary)
    summary_df = pd.DataFrame(columns=["game_id", "player_num_label", "avg_perf", "max_perf", "min_perf", "num_frames", "frames_above_avg"])
    ts_df = pd.DataFrame(columns=["game_id", "frame_id", "player_num_label", "actual_prob", "cf_prob", "perf"])
    for key, value in all_results.items():
        game_id, player_num_label = key
        game_id, player_num_label = int(game_id), int(player_num_label)
        avg_perf, max_perf, min_perf = value['aggregate_metrics']['average_performance'], value['aggregate_metrics']['max_performance'], value['aggregate_metrics']['min_performance']
        num_frames, frames_above_avg = value['aggregate_metrics']['num_frames_analyzed'], value['aggregate_metrics']['frames_above_average']
        
        tmp_df = pd.DataFrame({
            "game_id": [game_id],
            "player_num_label": [player_num_label],
            "avg_perf": [avg_perf],
            "max_perf": [max_perf],
            "min_perf": [min_perf],
            "num_frames": [num_frames],
            "frames_above_avg": [frames_above_avg]
        })
        summary_df = pd.concat([summary_df, tmp_df], axis=0)
        
        for frame_id, frame_data in value['frame_results'].items():
            tmp_df = pd.DataFrame({
                "game_id": [game_id],
                "frame_id": [int(frame_id)],
                "player_num_label": [player_num_label],
                "actual_prob": [frame_data['actual_probability']],
                "cf_prob": [frame_data['avg_counterfactual_probability']],
                "perf": [frame_data['performance_score']]
            })

            ts_df = pd.concat([ts_df, tmp_df], axis=0)
        
    summary_df.to_csv('processed_data/summary_results.csv', index=False)
    ts_df.to_csv('processed_data/time_series_results.csv', index=False)