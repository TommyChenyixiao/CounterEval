import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotsoccer
import tempfile
from pathlib import Path
import torch
import os
import sys
import shutil

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import custom utilities
from utils.counter_eval_utils import load_trained_gat_model, evaluate_single_graph
from utils.pyg_converter import create_pyg_dataset

def setup_title():
    """Setup the Streamlit page title and sidebar header"""
    st.title("Soccer Game Analysis")
    st.sidebar.header("Game Selection")

def load_data():
    """Load the test dataset"""
    df_actual = pd.read_parquet('processed_data/men_imbalanced_node_features_test.parquet')
    return df_actual

def evaluate_game(data, game_id, model):
    """
    Evaluate success probabilities for each frame in a game
    
    Args:
        data: DataFrame containing game data
        game_id: ID of the game to evaluate
        model: Trained model for prediction
    
    Returns:
        tuple: Lists of frames and corresponding probabilities
    """
    game_data = data[data["game_id"] == game_id]
    frames = sorted(game_data["frame_id"].unique())
    probabilities = []
    
    for frame in frames:
        frame_data = game_data[game_data["frame_id"] == frame]
        graph = create_pyg_dataset(frame_data)
        result = evaluate_single_graph(model, graph[0])
        probabilities.append(result['probability'])
    
    return frames, probabilities

def create_synchronized_animations(data, game_id, probabilities, frames):
    """
    Create synchronized animations with two-row layout and labeled players
    """
    game_data = data[data["game_id"] == game_id].copy()
    
    # Scale coordinates
    game_data["x"] = game_data["x"] * 105
    game_data["y"] = game_data["y"] * 68
    
    # Create figure with two row subplots
    fig = plt.figure(figsize=(12, 16))  # Adjusted figure size for vertical layout
    
    # Soccer field subplot (top)
    ax1 = plt.subplot2grid((2, 1), (0, 0))
    matplotsoccer.field("green", figsize=8, show=False, ax=ax1)
    ax1.set_xlim(0, 105)
    ax1.set_ylim(0, 68)
    ax1.set_title(f"Game {game_id}", fontsize=14, pad=20)
    
    # Initialize soccer plot elements
    defense_scatter = ax1.scatter([], [], c='white', edgecolor='blue', 
                                s=400, linewidth=2, label="Defense")
    offense_scatter = ax1.scatter([], [], c='white', edgecolor='red', 
                                s=400, linewidth=2, label="Offense")
    ball_scatter = ax1.scatter([], [], c='black', s=200, label="Ball")
    
    # Initialize text annotations
    defense_labels = []
    offense_labels = []
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
                  markeredgecolor='red', label='Offense', markersize=10, linewidth=0),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
                  markeredgecolor='blue', label='Defense', markersize=10, linewidth=0),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black',
                  label='Ball', markersize=10, linewidth=0)
    ]
    ax1.legend(handles=legend_elements, loc='upper right')
    
    # Probability plot subplot (bottom)
    ax2 = plt.subplot2grid((2, 1), (1, 0))
    ax2.set_xlim(min(frames), max(frames))
    ax2.set_ylim(0, 1)
    ax2.set_xlabel('Frame', fontsize=12)
    ax2.set_ylabel('Success Probability', fontsize=12)
    ax2.set_title('Predicted Success Probability', fontsize=14, pad=20)
    ax2.grid(True, alpha=0.3)
    
    # Initialize probability plot elements
    line, = ax2.plot([], [], 'b-', linewidth=2)
    point, = ax2.plot([], [], 'ro', markersize=10)
    
    def update(frame_idx):
        frame = frames[frame_idx]
        current_frame_data = game_data[game_data["frame_id"] == frame]
        
        # Clear previous labels
        for label in defense_labels + offense_labels:
            label.remove()
        defense_labels.clear()
        offense_labels.clear()
        
        # Update soccer field
        defense_data = current_frame_data[current_frame_data["att_team"] == 0]
        offense_data = current_frame_data[current_frame_data["att_team"] == 1]
        ball_data = current_frame_data[current_frame_data["att_team"] == -1]
        
        # Update scatter positions
        defense_scatter.set_offsets(np.c_[defense_data["x"], defense_data["y"]])
        offense_scatter.set_offsets(np.c_[offense_data["x"], offense_data["y"]])
        ball_scatter.set_offsets(np.c_[ball_data["x"], ball_data["y"]])
        
        # Add labels for players
        for x, y in zip(defense_data["x"], defense_data["y"]):
            label = ax1.text(x, y, "D", color='blue', ha='center', va='center', 
                           fontweight='bold', fontsize=10)
            defense_labels.append(label)
            
        for x, y in zip(offense_data["x"], offense_data["y"]):
            label = ax1.text(x, y, "O", color='red', ha='center', va='center', 
                           fontweight='bold', fontsize=10)
            offense_labels.append(label)
        
        # Update probability plot
        line.set_data(frames[:frame_idx+1], probabilities[:frame_idx+1])
        point.set_data([frame], [probabilities[frame_idx]])
        
        return [defense_scatter, offense_scatter, ball_scatter, line, point] + defense_labels + offense_labels
    
    ani = animation.FuncAnimation(fig, update, frames=len(frames), 
                                blit=True, interval=100)
    
    plt.tight_layout()
    
    # Save animation
    temp_file = tempfile.NamedTemporaryFile(suffix=".gif", delete=False)
    ani.save(temp_file.name, writer="pillow", fps=10, dpi=100)
    plt.close()
    
    return temp_file.name

def main():
    """Main function to run the Streamlit app"""
    setup_title()
    df_actual = load_data()
    
    # Create output directory if it doesn't exist
    output_dir = "saved_animations"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    model_dir = Path('results/GAT/best_model.pt')
    num_features = 18
    model = load_trained_gat_model(model_dir, num_features)
    
    # Game selection
    game_ids = sorted(df_actual["game_id"].unique())
    selected_game_id = st.sidebar.selectbox("Select Game ID", game_ids)
    
    # Create two columns in sidebar for buttons
    col1, col2 = st.sidebar.columns(2)
    
    if col1.button("Analyze Game"):
        with st.spinner('Generating animation...'):
            # Get probability predictions
            frames, probabilities = evaluate_game(df_actual, selected_game_id, model)
            
            # Generate synchronized animations
            animation_file = create_synchronized_animations(
                df_actual, selected_game_id, probabilities, frames
            )
            
            # Store the animation file path in session state
            st.session_state.animation_file = animation_file
            st.session_state.selected_game_id = selected_game_id
        
        # Display the combined animation
        st.image(animation_file, caption=f"Game {selected_game_id} Analysis", use_container_width=True)
    
    # Add save button
    if col2.button("Save Animation") and hasattr(st.session_state, 'animation_file'):
        output_path = os.path.join(output_dir, f"game_{st.session_state.selected_game_id}_analysis.gif")
        shutil.copy2(st.session_state.animation_file, output_path)
        st.sidebar.success(f"Animation saved to {output_path}")

if __name__ == "__main__":
    main()