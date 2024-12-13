import pandas as pd
import numpy as np
import argparse
import os
import torch
import multiprocessing as mp
from loguru import logger
from sklearn.model_selection import train_test_split
from datasets import MovementDataset
from model import MovementModel
from learner import Trainer

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def process_game(game_df, window_size, features):
    Xs = []
    ys = []
    frames = game_df['frame_id'].unique()
    for f in frames:
        curr_frame = game_df[game_df['frame_id'] == f]
        y = curr_frame[['x', 'y']].values.reshape(-1, 2)
        
        # Get the previous window_size frames
        if window_size is None:
            prev_frames = game_df[(game_df['frame_id'] < f) & (game_df['frame_id'] >= f - 1)]
        else:
            prev_frames = game_df[(game_df['frame_id'] < f) & (game_df['frame_id'] >= f - window_size)]
        
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
        
        ys.append(y)
        Xs.append(X)
    
    # print(f"{game_df['game_id'].iloc[0]}: {len(Xs)}")
    return Xs, ys

def prepare_data(df, args):
    """
    Prepare the data for training the model. For each player in each frame and game, we use its (x,y) as targets and the features (from features_path) args in the 
    previous window_size frames as input. We still keep the game_id, frame_id and player_id to be able to reconstruct the data later. The output X is of size
    (n_samples, window_size, n_features, number of player in the same game) and y is of size (n_samples, 2, number of player in the same game). 
    """
    # Window size
    window_size = args.window_size
    # Load the features
    with open(args.features_path, 'r') as f:
        features = f.read().splitlines()
    
    df = df.sort_values(['game_id', 'frame_id', 'player_num_label'])
    df = df.loc[df["att_team"] != -1]

    df["def_team"] = 0
    df["off_team"] = 0
    df.loc[df["att_team"] == 1, "off_team"] = 1
    df.loc[df["att_team"] == 0, "def_team"] = 1
    df = df.drop(columns=["att_team"])

    # Initialize the data
    Xs = []
    ys = []
    # Loop through the data
    games = df['game_id'].unique()
    
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.starmap(process_game, [(df[df['game_id'] == g], window_size, features) for g in games])
    
    for X, y in results:
        Xs.extend(X)
        ys.extend(y)

    return Xs, ys


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train a player movement model.')

    # Macro arguments
    parser.add_argument('-d', '--data_path', type=str, help='Path to the player tracking data.', default='../../processed_data/men_imbalanced_node_features_numbered.parquet')
    parser.add_argument('-o', '--output_dir', type=str, help='Path to save the model.', default='../../models/')
    parser.add_argument('-l', '--log_path', type=str, help='Path to save the log file.', default='movement_model.log')
    parser.add_argument('-r', '--reprocess', type=bool, help='Reprocess', default=False)
    # Data arguments
    parser.add_argument('-w', '--window_size', type=int, help='Window size for the model.', default=None)
    parser.add_argument('--features_path', type=str, help='Path to the feature file.', default='./features.txt')

    parser.add_argument("--betas", type=tuple, default=(0.9,0.9999))
    parser.add_argument("--regularizer", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--num_steps", type=int, default=0)

    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_grad_clip", type=float, default=None)
    parser.add_argument("--max_grad_norm", type=float, default=None)

    args = parser.parse_args()

    # check using cude, cpu or mps
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    # Load the data
    
    if not args.reprocess and ('Xs.npy' in os.listdir(os.path.dirname(args.data_path)) and 'ys.npy' in os.listdir(os.path.dirname(args.data_path))):
        Xs = np.load(os.path.join(os.path.dirname(args.data_path), 'Xs.npy'), allow_pickle=True)
        ys = np.load(os.path.join(os.path.dirname(args.data_path), 'ys.npy'), allow_pickle=True)
        logger.info(f"Data loaded. Xs shape: {len(Xs)}, ys shape: {len(ys)}")
    else:
        df = pd.read_parquet(args.data_path)
        Xs, ys = prepare_data(df, args)
        logger.info(f"Data prepared. Xs shape: {len(Xs)}, ys shape: {len(ys)}")
        # Save under data_path
        Xs_path = os.path.join(os.path.dirname(args.data_path), 'Xs.npy')
        ys_path = os.path.join(os.path.dirname(args.data_path), 'ys.npy')
        np.save(Xs_path, Xs)
        np.save(ys_path, ys)

    Xs_train, Xs_test, ys_train, ys_test = train_test_split(Xs, ys, test_size=0.2, random_state=42)
    Xs_train, Xs_val, ys_train, ys_val = train_test_split(Xs_train, ys_train, test_size=0.2, random_state=42)

    # normalize the data for continuous features
    X_mean = np.mean(Xs_train[:, :, :-3], axis=(0, 1))
    X_std = np.std(Xs_train[:, :, :-3], axis=(0, 1))
    # save the mean and std for later use
    np.save(os.path.join(args.output_dir, 'X_mean.npy'), X_mean)
    np.save(os.path.join(args.output_dir, 'X_std.npy'), X_std)

    Xs_train[:, :, :-3] = (Xs_train[:, :, :-3] - X_mean) / X_std
    Xs_val[:, :, :-3] = (Xs_val[:, :, :-3] - X_mean) / X_std
    Xs_test[:, :, :-3] = (Xs_test[:, :, :-3] - X_mean) / X_std

    y_mean = np.mean(ys_train, axis=(0, 1))
    y_std = np.std(ys_train, axis=(0, 1))
    np.save(os.path.join(args.output_dir, 'y_mean.npy'), y_mean)
    np.save(os.path.join(args.output_dir, 'y_std.npy'), y_std)

    ys_train = (ys_train - y_mean) / y_std
    ys_val = (ys_val - y_mean) / y_std
    ys_test = (ys_test - y_mean) / y_std

    logger.info(f"Train: {len(Xs_train)}, Val: {len(Xs_val)}, Test: {len(Xs_test)}")

    train_dataset = MovementDataset(Xs_train, ys_train, split='train')
    val_dataset = MovementDataset(Xs_val, ys_val, split='val')
    test_dataset = MovementDataset(Xs_test, ys_test, split='test')

    context_dim = Xs_train[0].shape[-1]
    model = MovementModel(2, context_dim, 2, 1, 2)
    model.to(device)

    logger.info("Number of param: {}".format(count_parameters(model)))

    optim = torch.optim.Adam(model.parameters(), lr=args.lr,betas=args.betas, weight_decay=args.regularizer)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.70)

    trainer = Trainer(model, optim, scheduler, train_dataset, val_dataset, args, device, logger)
    trainer.train()


