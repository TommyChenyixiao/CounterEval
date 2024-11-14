"""
This is a helper script to visualize the data. It is easy to check data integrity and consistency by visualizing the data.
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotsoccer
import tempfile
import plotly.express as px

def setup_title():
    st.title("Soccer Position Animation")
    st.sidebar.header("Select Dataset and ID")

def load_data():
    # Dataset selection
    live_datasets = ["./processed_data/men_imbalanced_node_features.parquet", 
                     "./processed_data/women_imbalanced_node_features.parquet",
                     "./processed_data/women_imbalanced_node_features_numbered.parquet",
                     "./processed_data/men_imbalanced_node_features_numbered.parquet"]
    static_datasets = ["./processed_data/men_node_features.parquet", "./processed_data/women_node_features.parquet"]
    dataset_type = st.sidebar.radio("Select Dataset", ["Live", "Static"])
    
    if dataset_type == "Live":
        dataset = st.sidebar.selectbox("Dataset", live_datasets)
    elif dataset_type == "Static":
        dataset = st.sidebar.selectbox("Dataset", static_datasets)

    # Load the selected dataset
    data = pd.read_parquet(dataset)

    return dataset_type, data

def get_game_ids(data, current_game_id):
    game_ids = sorted(data["game_id"].unique())
    selected_game_id = st.sidebar.selectbox("Game ID", game_ids, index=game_ids.index(current_game_id))
    return selected_game_id, game_ids

def create_animation(data, game_id):
    
    game_data = data[data["game_id"] == game_id]
    assert len(game_data) > 0, f"Game ID {game_id} not found in the dataset"

    # Scale x and y coordinates to the size of the soccer field
    game_data["x"] = game_data["x"] * 105
    game_data["y"] = game_data["y"] * 68
    game_data["v"] = game_data["v"] * 20  # For visualization purpose
    game_data["vx"] = game_data["vx"] * game_data["v"]
    game_data["vy"] = game_data["vy"] * game_data["v"]

    game_data["Node Type"] = game_data["att_team"].apply(lambda x: "Ball" if x == -1 else "Offense" if x == 1 else "Defense")
    
    has_player_num_label = 'player_num_label' in game_data.columns
    if has_player_num_label:
        game_data['player_num_label'] = game_data['player_num_label'].astype(str)

    assert game_data["success"].nunique() == 1, "Game data contains multiple success values"
    success = game_data["success"].iloc[0]
    success_text = "Success" if success == 1 else "Failure"

    # Set up the plot
    fig, ax = plt.subplots()
    matplotsoccer.field("green", figsize=8, show=False, ax=ax)
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 68)
    ax.set_title(f"Game ID: {game_id} - Counterattack: {success_text}")


    # Legend elements
    ball_legend = plt.Line2D([0], [0], marker='o', color='black', markersize=10, label="Ball")
    defense_legend = plt.Line2D([0], [0], marker='o', color='blue', markersize=10, label="Defense", markerfacecolor='white')
    offense_legend = plt.Line2D([0], [0], marker='o', color='red', markersize=10, label="Offense", markerfacecolor='white')

    # Add the legend to the plot
    ax.legend(handles=[ball_legend, defense_legend, offense_legend], loc='upper right')
    if has_player_num_label:
        # Initialize plot elements
        ball_node, = ax.plot([], [], 'ko', markersize=10, label="Ball")

        # Circle elements for players
        defense_circles = []
        offense_circles = []
        player_labels = []

        # Define custom legend elements
        ball_legend = plt.Line2D([0], [0], marker='o', color='black', markersize=10, label="Ball")
        defense_legend = plt.Line2D([0], [0], marker='o', color='blue', markersize=10, label="Defense", markerfacecolor='white')
        offense_legend = plt.Line2D([0], [0], marker='o', color='red', markersize=10, label="Offense", markerfacecolor='white')

        # Add the legend to the plot
        ax.legend(handles=[ball_legend, defense_legend, offense_legend], loc='upper right')

        def update(frame):
            for circle in defense_circles + offense_circles + player_labels:
                circle.remove()
            defense_circles.clear()
            offense_circles.clear()
            player_labels.clear()

            current_frame_data = game_data[game_data["frame_id"] == frame]
            defense_data = current_frame_data[current_frame_data["Node Type"] == "Defense"]
            offense_data = current_frame_data[current_frame_data["Node Type"] == "Offense"]
            ball_data = current_frame_data[current_frame_data["Node Type"] == "Ball"]

            # Draw ball node
            if not ball_data.empty:
                ball_node.set_data([ball_data["x"].iloc[0]], [ball_data["y"].iloc[0]])
            else:
                ball_node.set_data([], [])

            # Draw circles and labels for defense players
            for _, player in defense_data.iterrows():
                circle = plt.Circle((player["x"], player["y"]), 2, edgecolor='blue', facecolor='white', linewidth=2)
                ax.add_patch(circle)
                defense_circles.append(circle)
                label = ax.text(player["x"], player["y"], player["player_num_label"], color='blue', fontsize=8, ha='center', va='center', fontweight='bold')
                player_labels.append(label)

            # Draw circles and labels for offense players
            for _, player in offense_data.iterrows():
                circle = plt.Circle((player["x"], player["y"]), 2, edgecolor='red', facecolor='white', linewidth=2)
                ax.add_patch(circle)
                offense_circles.append(circle)
                label = ax.text(player["x"], player["y"], player["player_num_label"], color='red', fontsize=8, ha='center', va='center', fontweight='bold')
                player_labels.append(label)

            return ball_node, *defense_circles, *offense_circles, *player_labels

    else:
        # Initialize plot elements for points
        defense, = ax.plot([], [], 'bo', label="Defense")
        offense, = ax.plot([], [], 'ro', label="Offense")
        ball, = ax.plot([], [], 'ko', label="Ball")

        def update(frame):
            current_frame_data = game_data[game_data["frame_id"] == frame]
            defense_data = current_frame_data[current_frame_data["Node Type"] == "Defense"]
            offense_data = current_frame_data[current_frame_data["Node Type"] == "Offense"]
            ball_data = current_frame_data[current_frame_data["Node Type"] == "Ball"]

            # Position
            defense.set_data(defense_data["x"], defense_data["y"])
            offense.set_data(offense_data["x"], offense_data["y"])
            ball.set_data(ball_data["x"], ball_data["y"])

            return defense, offense, ball

    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=game_data["frame_id"].unique(), blit=True, repeat=False)
    
    # Save animation to a temporary file and return the file path
    temp_file = tempfile.NamedTemporaryFile(suffix=".gif", delete=False)
    ani.save(temp_file.name, writer="pillow", fps=10)
    return temp_file.name


if __name__ == "__main__":
    setup_title()
    dataset_type, data = load_data()
    if dataset_type == "Live":
        # Initialize the last generated game ID in session state if not already set
        if "last_generated_game_id" not in st.session_state:
            st.session_state.last_generated_game_id = data["game_id"].iloc[0]

        selected_game_id, game_ids = get_game_ids(data, st.session_state.last_generated_game_id)

        # Button to generate animation for the selected game ID from the dropdown
        if st.sidebar.button("Generate Animation", help="Generate GIF for selected game ID", key="generate"):
            # Generate the animation for the selected game ID
            animation_file = create_animation(data, selected_game_id)
            st.image(animation_file)
            # Update the last generated game ID
            st.session_state.last_generated_game_id = selected_game_id

        # Button to generate animation for the next game ID
        if st.sidebar.button("▶️ Next Game ID", help="Generate GIF for the next game ID", key="next"):
            # Get the index of the last generated game ID
            current_index = game_ids.index(st.session_state.last_generated_game_id)
            # Determine the next index, wrapping around if necessary
            next_index = (current_index + 1) % len(game_ids)
            # Get the next game ID
            next_game_id = game_ids[next_index]

            # Generate animation for the next game ID
            animation_file = create_animation(data, next_game_id)
            st.image(animation_file)
            # Update the last generated game ID
            st.session_state.last_generated_game_id = next_game_id

