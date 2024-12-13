import torch
import numpy as np
from model import MovementModel
from train import process_game

def visualize_model(model, df):
    # data only contains one game
    model.eval()

    df = df.sort_values(['game_id', 'frame_id', 'player_num_label'])
    df = df.loc[df["att_team"] != -1]
    df["def_team"] = 0
    df["off_team"] = 0
    df.loc[df["att_team"] == 1, "off_team"] = 1
    df.loc[df["att_team"] == 0, "def_team"] = 1
    df = df.drop(columns=["att_team"])

    window = None
    with open("./features.txt", 'r') as f:
        features = f.read().splitlines()
    
    Xs, ys = process_game(df, window, features)
    Xs = torch.tensor(Xs, dtype=torch.float32)
    ys = torch.tensor(ys, dtype=torch.float32)

    X_mean = np.load("../../models/X_mean.npy")
    X_std = np.load("../../models/X_std.npy")
    y_mean = np.load("../../models/y_mean.npy")
    y_std = np.load("../../models/y_std.npy")

    Xs[:, :, :-3] = (Xs[:, :, :-3] - X_mean) / X_std
    ys = (ys - y_mean) / y_std

    with torch.no_grad():
        recon_batch, mu, logvar = model(ys, Xs)



    