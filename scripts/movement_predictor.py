import torch 
import numpy as np
import pandas as pd
import warnings
from movement_model.model import MovementModel
warnings.filterwarnings("ignore")

def movement_predictor(df, model, window_size, features, X_mean, X_std, y_mean, y_std, samples=50):
    """
    df: pandas.DataFrame. This is a data frame of one game data.
    model: torch.nn.Module. This is the model that will be used to predict the movement of the players 
    """
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()
    assert df.loc[:, "game_id"].nunique() == 1, "The data frame should only contain one game"
    
    id_cols = ["game_id", "frame_id", "player_num_label"]
    actual_locations = df.loc[:, id_cols + ["x", "y"]]
    pred_locations = pd.DataFrame(columns=id_cols + ["x", "y"])
    
    frames = df['frame_id'].unique()
    for f in frames:
        # For each frame, we will predict the next location for each player
        # We will do this for samples number of times
        
        curr_frame = df[df['frame_id'] == f]
        
        # Get the previous window_size frames
        if window_size is None:
            prev_frames = df[(df['frame_id'] < f) & (df['frame_id'] >= f - 1)]
        else:
            prev_frames = df[(df['frame_id'] < f) & (df['frame_id'] >= f - window_size)]
        
        X = []
        players = prev_frames['player_num_label'].unique()
        for p in players:
            player = prev_frames[prev_frames['player_num_label'] == p]
            # Create one hot encoding for att_team
            if player.shape[0] == 0:
                continue
            x = player[features]
            num_features = len(features)                
            if window_size is None:
                x = x.values.reshape(num_features,)
            else:
                x = x.values.T
            
            X.append(x)
            # If window size is None, the data size is (n_features, n_players)
            # If else, the data size is (n_features, window_size, n_players)
        if len(X) == 0:
            continue

        X = np.stack(X, axis=0)
        X = torch.tensor(X, dtype=torch.float32)
        X = X.unsqueeze(0)
        X[:, :, :-3] = (X[:, :, :-3] - X_mean) / X_std
        X = X.to(device)

        with torch.no_grad():
            pred = []
            for _ in range(samples):
                eps = torch.randn(X.shape[0], X.shape[1], 2)
                eps = eps.to(device)
                recon_loc = model.decode(eps, X)
                recon_loc = (recon_loc.cpu().numpy() * y_std) + y_mean
                
                pred.append(recon_loc)
            # pred is of shape (samples, 1, n_players, 2)
            pred = np.stack(pred, axis=0)
        for i, p in enumerate(players):
            player_pred = pred[:, 0, i, :]
            tmp_df = pd.DataFrame({"game_id": curr_frame["game_id"].iloc[0], 
                                   "frame_id": curr_frame["frame_id"].iloc[0], 
                                   "player_num_label": p, 
                                   "x": player_pred[:, 0], 
                                   "y": player_pred[:, 1]})
            pred_locations = pd.concat([pred_locations, tmp_df], axis=0)
        
    return actual_locations, pred_locations

if __name__ == "__main__":
    df = pd.read_parquet("processed_data/men_imbalanced_node_features_test.parquet")
    X_mean = np.load("models/X_mean.npy")
    X_std = np.load("models/X_std.npy")
    y_mean = np.load("models/y_mean.npy")
    y_std = np.load("models/y_std.npy")
    with open("scripts/movement_model/features.txt", 'r') as f:
        features = f.read().splitlines()

    df["def_team"] = 0
    df["off_team"] = 0
    df.loc[df["att_team"] == 1, "off_team"] = 1
    df.loc[df["att_team"] == 0, "def_team"] = 1
    df = df.drop(columns=["att_team"])

    context_dim = 10
    model = MovementModel(2, context_dim, 2, 2, 3)
    model.load_state_dict(torch.load("models/checkpoint_21.pt"))

    game_ids = df["game_id"].unique()
    total_actual_locations = pd.DataFrame(columns=["game_id", "frame_id", "player_num_label", "x", "y"])
    total_pred_locations = pd.DataFrame(columns=["game_id", "frame_id", "player_num_label", "x", "y"])
    batch_size = 10
    for i, g in enumerate(game_ids):
        game_df = df[df["game_id"] == g]
        actual_locations, pred_locations = movement_predictor(game_df, model, window_size=None, features=features, X_mean=X_mean, X_std=X_std, y_mean=y_mean, y_std=y_std, samples=50)
        total_actual_locations = pd.concat([total_actual_locations, actual_locations], axis=0)
        total_pred_locations = pd.concat([total_pred_locations, pred_locations], axis=0)
        if (i+1) % batch_size == 0:
            print(f"Processed {i+1} games")    
            total_actual_locations.to_csv(f"processed_data/actual_locations_{i}.csv", index=False)
            total_pred_locations.to_csv(f"processed_data/pred_locations_{i}.csv", index=False)
            total_actual_locations = pd.DataFrame(columns=["game_id", "frame_id", "player_num_label", "x", "y"])
            total_pred_locations = pd.DataFrame(columns=["game_id", "frame_id", "player_num_label", "x", "y"])


