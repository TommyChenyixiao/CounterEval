import torch
import os
import sys
from models.gnn_models import create_gnn_model
from pathlib import Path
import numpy as np
from torch_geometric.data import Data, Batch
from utils.pyg_converter import create_pyg_dataset
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_trained_gat_model(model_path, num_features):
    # Define the same model configuration used during training
    model_config = {
        'hidden_channels': 32,
        'num_layers': 2,
        'heads': 4,
        'dropout': 0.5,
        'pool_type': 'mean'
    }
    
    # Create model with same architecture
    model = create_gnn_model('GAT', num_features, **model_config)
    
    # Load the saved state dict
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Put model in evaluation mode
    model = model.to(device)
    model.eval()
    
    return model


def evaluate_single_graph(model, graph: Data) -> dict:
    """
    Evaluate model on a single graph and return predictions with metrics.
    
    Args:
        model: Trained GAT model
        graph: PyG Data object containing a single graph
        
    Returns:
        dict: Dictionary containing prediction probabilities and actual label
    """
    device = next(model.parameters()).device
    model.eval()
    
    # Move graph to same device as model
    graph = graph.to(device)
    
    # Create a batch with single graph
    # This ensures edge_index and batch indices are properly formatted
    batch = Batch.from_data_list([graph])
    
    with torch.no_grad():
        # Get model prediction
        output = model(
            x=batch.x,
            edge_index=batch.edge_index,
            batch=batch.batch
        )
        prob = torch.sigmoid(output).item()
        pred = int(prob > 0.5)
        
        return {
            'probability': prob,
            'prediction': pred,
            'actual': graph.y.item(),
            'correct': pred == graph.y.item()
        }


def create_counterfactual_dataset(df_actual, df_loc, game_id, frame_id, player_num_label):
    """
    Creates counterfactual datasets by replacing actual x,y coordinates with sampled coordinates
    for a specific game_id, frame_id, and player_num_label combination.
    
    Parameters:
    -----------
    df_actual : pd.DataFrame
        Original dataset with actual player positions and features
    df_loc : pd.DataFrame
        Dataset containing sampled positions from movement model
    game_id : int
        Game ID to process
    frame_id : int
        Frame ID to process
    player_num_label : int
        Player number to process
    
    Returns:
    --------
    list of pd.DataFrame
        List of dataframes, each containing one counterfactual scenario
    """
    # Filter samples for specific game, frame, and player
    samples = df_loc[
        (df_loc['game_id'] == game_id) & 
        (df_loc['frame_id'] == frame_id) &
        (df_loc['player_num_label'] == player_num_label)
    ].copy()
    
    # Create list to store counterfactual datasets
    counterfactual_datasets = []
    
    # Filter the actual data for the specific frame
    base_data = df_actual[(df_actual['game_id'] == game_id) & 
        (df_actual['frame_id'] == frame_id)].copy()
    
    # For each sample
    for idx, sample in samples.iterrows():
        # Create a copy of the filtered base data
        cf_data = base_data.copy()
        
        # Find the corresponding row in the actual data
        mask = (
            (cf_data['game_id'] == sample['game_id']) & 
            (cf_data['player_num_label'] == sample['player_num_label'])
        )
        
        # Replace x, y coordinates for the specific player
        cf_data.loc[mask, 'x'] = sample['x']
        cf_data.loc[mask, 'y'] = sample['y']
        
        # Add to list
        counterfactual_datasets.append(cf_data)
    
    return counterfactual_datasets

def calculate_average_counterfactual_probability(counterfactual_datasets, model, create_pyg_dataset, evaluate_single_graph):
    """
    Calculates average probability across all counterfactual datasets
    
    Parameters:
    -----------
    counterfactual_datasets : list of pd.DataFrame
        List of counterfactual scenarios
    model : torch.nn.Module
        The trained graph neural network model
    create_pyg_dataset : function
        Function to convert DataFrame to PyG dataset
    evaluate_single_graph : function
        Function to evaluate a single graph
        
    Returns:
    --------
    dict
        Dictionary containing average probability and all individual probabilities
    """
    probabilities = []
    
    for dataset in counterfactual_datasets:
        # Convert to PyG dataset
        sample_graph = create_pyg_dataset(dataset)
        
        # Get evaluation results
        results = evaluate_single_graph(model, sample_graph[0])
        
        # Store probability
        probabilities.append(results['probability'])
    
    # Calculate average
    avg_probability = sum(probabilities) / len(probabilities)
    
    return {
        'average_probability': avg_probability,
        'individual_probabilities': probabilities
    }

def compare_movement_performance(df_actual, df_loc, model, create_pyg_dataset, evaluate_single_graph, 
                               game_id, frame_id, player_num_label):
    """
    Compares actual movement performance against counterfactual alternatives
    
    Parameters:
    -----------
    df_actual : pd.DataFrame
        Original dataset with actual player positions
    df_loc : pd.DataFrame
        Dataset containing sampled positions
    model : torch.nn.Module
        The trained graph neural network model
    create_pyg_dataset : function
        Function to convert DataFrame to PyG dataset
    evaluate_single_graph : function
        Function to evaluate a single graph
    game_id : int
        Game ID to process
    frame_id : int
        Frame ID to process
    player_num_label : int
        Player number to process
        
    Returns:
    --------
    dict
        Dictionary containing performance metrics
    """
    # Get base probability
    base_data = df_actual[
        (df_actual['game_id'] == game_id) & 
        (df_actual['frame_id'] == frame_id)
    ].copy()
    base_graph = create_pyg_dataset(base_data)
    base_results = evaluate_single_graph(model, base_graph[0])
    P_actual = base_results['probability']
    
    # Get counterfactual datasets
    counterfactual_datasets = create_counterfactual_dataset(
        df_actual, df_loc, game_id, frame_id, player_num_label
    )
    
    # Calculate average counterfactual probability
    cf_results = calculate_average_counterfactual_probability(
        counterfactual_datasets, model, create_pyg_dataset, evaluate_single_graph
    )
    P_cf_avg = cf_results['average_probability']
    
    # Calculate performance metric m(p,t)
    performance_score = P_actual - P_cf_avg
    
    return {
        'actual_probability': P_actual,
        'avg_counterfactual_probability': P_cf_avg,
        'performance_score': performance_score,
        'base_prediction': base_results['prediction'],
        'base_actual': base_results['actual'],
        'base_correct': base_results['correct'],
        'num_counterfactuals': len(cf_results['individual_probabilities']),
        'counterfactual_probabilities': cf_results['individual_probabilities']
    }

def analyze_player_game_contribution(df_actual, df_loc, model, create_pyg_dataset, evaluate_single_graph,
                                   game_id, player_num_label):
    """
    Analyzes a player's contribution throughout an entire game by calculating 
    performance scores for all frames.
    
    Parameters:
    -----------
    df_actual : pd.DataFrame
        Original dataset with actual player positions
    df_loc : pd.DataFrame
        Dataset containing sampled positions
    model : torch.nn.Module
        The trained graph neural network model
    create_pyg_dataset : function
        Function to convert DataFrame to PyG dataset
    evaluate_single_graph : function
        Function to evaluate a single graph
    game_id : int
        Game ID to analyze
    player_num_label : int
        Player number to analyze
    
    Returns:
    --------
    dict
        Dictionary containing frame-by-frame and aggregate performance metrics
    """
    # Get all unique frames for this game
    frames = sorted(df_actual[
        (df_actual['game_id'] == game_id) & (df_actual['frame_id'] > 0)
    ]['frame_id'].unique())
    
    # Store results for each frame
    frame_results = {}
    
    # Analyze each frame
    for frame_id in frames:
        try:
            results = compare_movement_performance(
                df_actual, df_loc, model, create_pyg_dataset, evaluate_single_graph,
                game_id, frame_id, player_num_label
            )
            frame_results[frame_id] = results
        except Exception as e:
            print(f"Error processing frame {frame_id}: {str(e)}")
            continue
    
    # Calculate aggregate metrics
    performance_scores = [res['performance_score'] for res in frame_results.values()]
    avg_performance = sum(performance_scores) / len(performance_scores) if performance_scores else 0
    
    return {
        'frame_results': frame_results,
        'aggregate_metrics': {
            'average_performance': avg_performance,
            'max_performance': max(performance_scores) if performance_scores else 0,
            'min_performance': min(performance_scores) if performance_scores else 0,
            'num_frames_analyzed': len(performance_scores),
            'frames_above_average': sum(1 for score in performance_scores if score > 0)
        }
    }

def print_player_analysis(results):
    """
    Prints a formatted analysis of a player's game contribution
    """
    agg = results['aggregate_metrics']
    print("\nPlayer Game Contribution Analysis")
    print("================================")
    print(f"Frames Analyzed: {agg['num_frames_analyzed']}")
    print(f"Average Performance Score: {agg['average_performance']:.3f}")
    print(f"Best Performance: {agg['max_performance']:.3f}")
    print(f"Worst Performance: {agg['min_performance']:.3f}")
    print(f"Frames Above Average: {agg['frames_above_average']} ({(agg['frames_above_average']/agg['num_frames_analyzed']*100):.1f}%)")
    
    # Print frame-by-frame details for significant moments
    print("\nSignificant Moments:")
    print("-------------------")
    for frame_id, frame_data in results['frame_results'].items():
        if abs(frame_data['performance_score']) > abs(agg['average_performance']):
            print(f"Frame {frame_id}:")
            print(f"  Performance Score: {frame_data['performance_score']:.3f}")
            print(f"  Actual Probability: {frame_data['actual_probability']:.3f}")
            print(f"  Avg Counterfactual Probability: {frame_data['avg_counterfactual_probability']:.3f}")


from tqdm import tqdm

def analyze_all_players(df_actual, df_loc, model, create_pyg_dataset, evaluate_single_graph):
    """
    Analyzes all available player-game combinations in df_loc with progress tracking
    """
    # Get unique combinations
    combinations = df_loc.groupby(['game_id', 'player_num_label']).size().reset_index()[['game_id', 'player_num_label']]
    
    # Store results
    all_results = {}
    
    # Create progress bar for all combinations
    pbar = tqdm(combinations.iterrows(), total=len(combinations), 
                desc="Analyzing player-game combinations", unit="combination")
    
    for _, row in pbar:
        game_id = row['game_id']
        player_num_label = row['player_num_label']
        
        pbar.set_description(f"Analyzing Game {game_id}, Player {player_num_label}")
        
        try:
            results = analyze_player_game_contribution(
                df_actual, df_loc, model, create_pyg_dataset, evaluate_single_graph,
                game_id, player_num_label
            )
            
            all_results[(game_id, player_num_label)] = results
            # print_player_analysis(results)
            create_performance_visualization(results, game_id, player_num_label)
            
        except Exception as e:
            print(f"\nError analyzing game {game_id}, player {player_num_label}: {str(e)}")
            continue
    
    return all_results

def analyze_player_game_contribution(df_actual, df_loc, model, create_pyg_dataset, evaluate_single_graph,
                                   game_id, player_num_label):
    """
    Analyzes a player's contribution throughout an entire game with progress tracking
    """
    # Get all unique frames for this game
    frames = sorted(df_actual[
        (df_actual['game_id'] == game_id) & (df_actual['frame_id'] > 0)
    ]['frame_id'].unique())
    
    # Store results
    frame_results = {}
    
    # Create progress bar for frames
    pbar = tqdm(frames, desc=f"Processing frames", unit="frame", leave=False)
    
    for frame_id in pbar:
        pbar.set_description(f"Processing frame {frame_id}")
        try:
            results = compare_movement_performance(
                df_actual, df_loc, model, create_pyg_dataset, evaluate_single_graph,
                game_id, frame_id, player_num_label
            )
            frame_results[frame_id] = results
        except Exception as e:
            print(f"\nError processing frame {frame_id}: {str(e)}")
            continue
    
    # Calculate aggregate metrics
    performance_scores = [res['performance_score'] for res in frame_results.values()]
    avg_performance = sum(performance_scores) / len(performance_scores) if performance_scores else 0
    
    return {
        'frame_results': frame_results,
        'aggregate_metrics': {
            'average_performance': avg_performance,
            'max_performance': max(performance_scores) if performance_scores else 0,
            'min_performance': min(performance_scores) if performance_scores else 0,
            'num_frames_analyzed': len(performance_scores),
            'frames_above_average': sum(1 for score in performance_scores if score > 0)
        }
    }

def get_single_player_analysis(df_actual, df_loc, model, create_pyg_dataset, evaluate_single_graph, 
                             game_id, player_num_label):
    """
    Analyzes a specific player in a specific game with progress tracking
    """
    if not df_loc[(df_loc['game_id'] == game_id) & 
                  (df_loc['player_num_label'] == player_num_label)].empty:
        try:
            print(f"\nAnalyzing Game {game_id}, Player {player_num_label}")
            results = analyze_player_game_contribution(
                df_actual, df_loc, model, create_pyg_dataset, evaluate_single_graph,
                game_id, player_num_label
            )
            print_player_analysis(results)
            create_performance_visualization(results, game_id, player_num_label)
            return results
        except Exception as e:
            print(f"Error analyzing game {game_id}, player {player_num_label}: {str(e)}")
            return None
    else:
        print(f"No data found for game {game_id}, player {player_num_label}")

def create_performance_visualization(results, game_id, player_num_label):
    """
    Creates and saves a visualization of player performance over time
    """
    frame_results = results['frame_results']
    frames = sorted(frame_results.keys())
    performance_scores = [frame_results[f]['performance_score'] for f in frames]
    actual_probs = [frame_results[f]['actual_probability'] for f in frames]
    cf_probs = [frame_results[f]['avg_counterfactual_probability'] for f in frames]
    
    plt.figure(figsize=(15, 8))
    
    # Plot performance scores
    plt.subplot(2, 1, 1)
    plt.plot(frames, performance_scores, 'b-', label='Performance Score')
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.title(f'Player {player_num_label} Performance in Game {game_id}')
    plt.xlabel('Frame')
    plt.ylabel('Performance Score')
    plt.legend()
    
    # Plot probabilities
    plt.subplot(2, 1, 2)
    plt.plot(frames, actual_probs, 'g-', label='Actual Probability')
    plt.plot(frames, cf_probs, 'r-', label='Avg Counterfactual Probability')
    plt.xlabel('Frame')
    plt.ylabel('Probability')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'vis/player_{player_num_label}_game_{game_id}_analysis.png')
    plt.close()

# Example usage:
# For a single player:
# results = get_single_player_analysis(
#     df_actual, df_loc, model, create_pyg_dataset, evaluate_single_graph,
#     game_id=137, player_num_label=4
# )

# For all players:
# all_results = analyze_all_players(
#     df_actual, df_loc_subset, model, create_pyg_dataset, evaluate_single_graph
# )