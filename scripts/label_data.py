import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

def process_data(df):
    """
    Process the DataFrame to assign player labels and calculate Euclidean distances.

    Parameters:
    - df: A pandas DataFrame containing game data.

    Returns:
    - df: The processed DataFrame with labeled players.
    - df_test: A DataFrame with calculated Euclidean distances for player tracking.
    """
    # Define the criteria to identify the ball (based on specific conditions)
    ball_criteria = (df['dist_ball'] == 0) & (df['angle_ball'] == 0)
    df['player_num_label'] = None  # Initialize a column for player labels

    # Assign label '0' for the ball
    df.loc[ball_criteria, 'player_num_label'] = 0

    # Iterate through each unique game_id
    for game_id in tqdm(df['game_id'].unique(), desc="Processing Game IDs"):
        # Extract the first frame (frame_id = 0) for the current game, excluding the ball
        first_frame = df[(df['game_id'] == game_id) & (df['frame_id'] == 0) & (~ball_criteria)].copy()

        # Assign labels 1 to 22 for the first 22 non-ball players
        first_frame['player_num_label'] = np.arange(1, 23)
        df.update(first_frame[['player_num_label']])
        prev_frame_data = first_frame
        frame_ids = sorted(df[df['game_id'] == game_id]['frame_id'].unique())

        # Iterate through each frame (excluding the first frame)
        for frame_id in tqdm(frame_ids, desc=f"Processing Frames for Game ID {game_id}", leave=False):
            if frame_id == 0:
                continue

            # Extract the current frame data, excluding the ball
            current_frame_data = df[(df['game_id'] == game_id) & (df['frame_id'] == frame_id) & (~ball_criteria)].copy()

            # Skip if there are no players in the current frame
            if current_frame_data.empty:
                continue

            # Calculate the Euclidean distance matrix between current and previous frame players
            distance_matrix = np.sqrt(
                (current_frame_data['x'].values[:, np.newaxis] - prev_frame_data['x'].values) ** 2 +
                (current_frame_data['y'].values[:, np.newaxis] - prev_frame_data['y'].values) ** 2
            )

            # Iterate through each player in the previous frame to assign labels
            for i, prev_row in prev_frame_data.iterrows():
                if len(current_frame_data) == 0:
                    break

                # Filter current frame players by the same attacking team as the previous player
                team_filtered_data = current_frame_data[current_frame_data['att_team'] == prev_row['att_team']]

                # Skip if no matching players are found
                if team_filtered_data.empty:
                    continue

                # Calculate the Euclidean distance for the filtered players
                distances = np.sqrt(
                    (team_filtered_data['x'].values - prev_row['x']) ** 2 +
                    (team_filtered_data['y'].values - prev_row['y']) ** 2
                )

                # Skip if there are no distances to compare
                if len(distances) == 0:
                    continue

                # Find the index of the closest player
                closest_idx = np.argmin(distances)
                closest_point_idx = team_filtered_data.index[closest_idx]

                # Assign the label from the previous frame to the closest player
                df.loc[closest_point_idx, 'player_num_label'] = prev_row['player_num_label']

                # Remove the labeled player from the current frame data to avoid duplication
                current_frame_data = current_frame_data.drop(index=closest_point_idx)

            # Update prev_frame_data for the next iteration
            prev_frame_data = df[(df['game_id'] == game_id) & (df['frame_id'] == frame_id) & (~ball_criteria)]

    # Convert player labels to integers
    df['player_num_label'] = df['player_num_label'].astype(int)

    # Calculate the Euclidean distance between current and previous positions for each player
    df_test = df[~ball_criteria].copy()
    df_test = df_test.sort_values(['game_id', 'frame_id', 'player_num_label'])
    df_test['x_prev'] = df_test.groupby(['game_id', 'player_num_label'])['x'].shift(1)
    df_test['y_prev'] = df_test.groupby(['game_id', 'player_num_label'])['y'].shift(1)
    df_test['euclidean_dist'] = np.sqrt((df_test['x'] - df_test['x_prev'])**2 + (df_test['y'] - df_test['y_prev'])**2)

    return df, df_test

def plot_histogram_and_print_table(df_test, gender):
    """
    Plot the histogram of Euclidean distances and print the bin statistics table.

    Parameters:
    - df_test: A DataFrame containing calculated Euclidean distances.
    - gender: The gender-specific title for the histogram ('Women' or 'Men').
    """
    # Plot the histogram of Euclidean distances
    weights = np.ones(len(df_test)) / len(df_test) * 100
    n, bins, patches = plt.hist(df_test['euclidean_dist'], bins=10, weights=weights)
    plt.xlabel('Euclidean Distance')
    plt.ylabel('Percentage')
    plt.title(f'Distribution of Euclidean Distances ({gender})')
    plt.show()

    # Calculate counts for each bin
    counts, _ = np.histogram(df_test['euclidean_dist'], bins=bins)

    # Create a DataFrame for bin statistics
    bin_ranges = [f"{bins[i]:.2f} - {bins[i+1]:.2f}" for i in range(len(bins) - 1)]
    percentages = [f"{n[i]:.1f}%" for i in range(len(n))]
    raw_counts = counts

    # Construct the DataFrame for non-zero bins
    df_bins = pd.DataFrame({
        'Bin Range': bin_ranges,
        'Percentage': percentages,
        'Count': raw_counts
    })

    # Display only non-zero bins
    df_bins_non_zero = df_bins[df_bins['Count'] > 0]
    print(df_bins_non_zero)

def plot_histograms_and_print_tables(df_tests, genders):
    """
    Plot side-by-side histograms for Euclidean distances and print the bin statistics tables.

    Parameters:
    - df_tests: List of DataFrames containing calculated Euclidean distances.
    - genders: List of gender-specific titles ('Women', 'Men').
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    # Iterate through each DataFrame and gender for plotting
    for i, (df_test, gender) in enumerate(zip(df_tests, genders)):
        weights = np.ones(len(df_test)) / len(df_test) * 100
        n, bins, patches = axs[i].hist(df_test['euclidean_dist'], bins=10, weights=weights)
        axs[i].set_xlabel('Euclidean Distance')
        axs[i].set_ylabel('Percentage' if i == 0 else '')
        axs[i].set_title(f'Distribution of Euclidean Distances ({gender})')

        # Calculate bin statistics
        counts, _ = np.histogram(df_test['euclidean_dist'], bins=bins)
        bin_ranges = [f"{bins[j]:.2f} - {bins[j+1]:.2f}" for j in range(len(bins) - 1)]
        percentages = [f"{n[j]:.1f}%" for j in range(len(n))]
        raw_counts = counts

        # Construct the DataFrame for bin statistics
        df_bins = pd.DataFrame({
            'Bin Range': bin_ranges,
            'Percentage': percentages,
            'Count': raw_counts
        })

        # Display only non-zero bins
        df_bins_non_zero = df_bins[df_bins['Count'] > 0]
        print(f"\nStatistics for {gender} Dataset:")
        print(df_bins_non_zero)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Check if the input file names are provided
    if len(sys.argv) != 3:
        print("Usage: python main.py <file_name_1> <file_name_2>")
        sys.exit(1)

    input_files = sys.argv[1:3]
    genders = []

    # Determine gender based on file names
    for input_file in input_files:
        if 'women' in input_file.lower():
            genders.append('Women')
        elif 'men' in input_file.lower():
            genders.append('Men')
        else:
            genders.append('Unknown')

    df_tests = []

    # Process each input file
    for input_file in input_files:
        print(f"Processing {input_file}...")
        df = pd.read_parquet(input_file)
        _, df_test = process_data(df)
        df_tests.append(df_test)

    # Plot histograms and print statistics tables
    plot_histograms_and_print_tables(df_tests, genders)

    # Save the processed DataFrames
    for input_file, df in zip(input_files, df_tests):
        output_path = input_file.replace(".parquet", "_numbered_test.parquet")
        print(f"Saving processed data to {output_path}...")
        df.to_parquet(output_path, index=False, engine='pyarrow')