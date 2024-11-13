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
                     "./processed_data/women_imbalanced_node_features.parquet"]
    static_datasets = ["./processed_data/men_node_features.parquet", "./processed_data/women_node_features.parquet"]
    dataset_type = st.sidebar.radio("Select Dataset", ["Live", "Static"])
    
    if dataset_type == "Live":
        dataset = st.sidebar.selectbox("Dataset", live_datasets)
    elif dataset_type == "Static":
        dataset = st.sidebar.selectbox("Dataset", static_datasets)

    # Load the selected dataset
    data = pd.read_parquet(dataset)

    return dataset_type, data

def get_game_ids(data):
    game_ids = data["game_id"].unique()
    selected_game_id = st.sidebar.selectbox("Game ID", game_ids)
    return selected_game_id

def create_animation(data, game_id):
    
    game_data = data[data["game_id"] == game_id]
    assert len(game_data) > 0, f"Game ID {game_id} not found in the dataset"

    # Scale x and y coordinates to the size of the soccer field
    game_data["x"] = game_data["x"] * 105
    game_data["y"] = game_data["y"] * 68
    game_data["v"] = game_data["v"] * 20 #NOTE For visualization purpose
    game_data["vx"] = game_data["vx"] * game_data["v"]
    game_data["vy"] = game_data["vy"] * game_data["v"]

    game_data["Node Type"] = game_data["att_team"].apply(lambda x: "Ball" if x == -1 else "Offense" if x == 1 else "Defense")
    
    assert game_data["success"].nunique() == 1, "Game data contains multiple success values"
    success = game_data["success"].iloc[0]
    success_text = "Success" if success == 1 else "Failure"
    # Set up the plot
    fig, ax = plt.subplots()
    matplotsoccer.field("green", figsize=8, show=False, ax=ax)
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 68)
    ax.set_title(f"Game ID: {game_id} - Counterattack: {success_text}")
    
    # Initialize plot elements
    defense, = ax.plot([], [], 'bo', label="Defense")
    offense, = ax.plot([], [], 'ro', label="Offense")
    ball, = ax.plot([], [], 'ko', label="Ball")

    ax.legend(loc='upper right')
    
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
        selected_game_id = get_game_ids(data)
        if st.sidebar.button("Generate Animation"):
            # Generate the animation for the selected game_id
            animation_file = create_animation(data, selected_game_id)
            
            # Display the animation in the app
            st.image(animation_file)