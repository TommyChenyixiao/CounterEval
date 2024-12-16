import matplotlib.pyplot as plt
import matplotsoccer
import pandas as pd

def generate_movement_prediction_plot(df, actual_locations, pred_locations, game_id, frame_id):
    actual_frame = actual_locations[(actual_locations["game_id"] == game_id) & (actual_locations["frame_id"] == frame_id)]
    pred_frame = pred_locations[(pred_locations["game_id"] == game_id) & (pred_locations["frame_id"] == frame_id)]
    curr_frame = df[(df["game_id"] == game_id) & (df["frame_id"] == frame_id)]

    fig, ax = plt.subplots(figsize=(10, 7))
    for i, row in actual_frame.iterrows():
        player_num_label = row["player_num_label"]
        x = row["x"] * 105
        y = row["y"] * 68
        if curr_frame.loc[curr_frame["player_num_label"] == player_num_label, "att_team"].iloc[0] == 0:
            circle = plt.Circle((x, y), 1, edgecolor='blue', facecolor='white', linewidth=1)
            ax.add_patch(circle)
            label = ax.text(x, y, int(player_num_label), color='blue', fontsize=5, ha='center', va='center', fontweight='bold')
        else:
            circle = plt.Circle((x, y), 1, edgecolor='red', facecolor='white', linewidth=1)
            ax.add_patch(circle)
            label = ax.text(x, y, int(player_num_label), color='red', fontsize=5, ha='center', va='center', fontweight='bold')

    for i, row in pred_frame.iterrows():
        player_num_label = row["player_num_label"]
        row["x"] = max(0, min(1, row["x"]))
        row["y"] = max(0, min(1, row["y"]))
        x = row["x"] * 105
        y = row["y"] * 68
        ax.plot(x, y, 'k+', markersize=3)

    matplotsoccer.field("green", figsize=8, show=False, ax=ax)
    plt.title(f"Game ID: {game_id}, Frame ID: {frame_id}")

    # plt.savefig(f"images/movement_prediction_{game_id}_{frame_id}.png")
    plt.show()

def generate_movement_prediction_plot_player(df, actual_locations, pred_locations, game_id, frame_id, player_id):
    actual_frame = actual_locations[(actual_locations["game_id"] == game_id) & (actual_locations["frame_id"] == frame_id) & (actual_locations["player_num_label"] == player_id)]
    pred_frame = pred_locations[(pred_locations["game_id"] == game_id) & (pred_locations["frame_id"] == frame_id) & (pred_locations["player_num_label"] == player_id)]
    
    x_actual = actual_frame["x"].values[0] * 105
    y_actual = actual_frame["y"].values[0] * 68

    x_pred = pred_frame["x"].values * 105
    y_pred = pred_frame["y"].values * 68

    # Plotting
    plt.figure(figsize=(8, 6))
    # plt.contourf(x_grid, y_grid, density, levels=30, cmap='Blues')
    # plt.colorbar(label='Density')
    plt.scatter(x_pred, y_pred, s=10, color='gray', alpha=0.5, label='Predicted Points')
    plt.scatter(x_actual, y_actual, color='red', label='Actual Point', zorder=5)
    plt.title(f"Game ID: {game_id}, Frame ID: {frame_id}, Player ID: {player_id}")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.show()

if __name__ == "__main__":

    actual_locations = pd.read_csv("processed_data/actual_locations_19.csv")  
    pred_locations = pd.read_csv("processed_data/pred_locations_19.csv")
    df = pd.read_parquet("processed_data/men_imbalanced_node_features_test.parquet")
    game_id = pred_locations["game_id"].unique()[1]
        
    # for frame_id in [5, 50, 100]:
    #     generate_movement_prediction_plot(df, actual_locations, pred_locations, game_id, frame_id)
    frame_id = 100
    player_id = 8
    generate_movement_prediction_plot_player(df, actual_locations, pred_locations, game_id, frame_id, player_id)