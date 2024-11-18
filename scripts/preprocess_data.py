import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data and merge with the manual check dataset
df = pd.read_parquet('processed_data/men_imbalanced_node_features_numbered.parquet')
check_df = pd.read_csv('experiments/men_manual_check_dataset.csv')
check = check_df.rename(columns={'index': 'game_id'})
df_combined = df.merge(check, on='game_id', how='inner')

# Remove ball flying game
df_processed = df_combined[(df_combined["ball_flying_issue"] != "X") & (df_combined["player_number_issue"] != "X")]

# Reverse the success label for the marked games
df_processed.loc[df_processed['success_label_issue'] == 'X', 'success'] = 1 - df_processed['success']

# Drop game with short number of frames
df_processed = df_processed.groupby('game_id').filter(lambda x: x['frame_id'].nunique() >= 50).reset_index(drop=True).drop(columns=['ball_flying_issue', 'player_number_issue', 'success_label_issue', 'dataset_name', 'comments', 'manual_check_passed'])

# Print percentage of dropped games
print(f"Percentage of dropped games: {100 * (1 - len(df_processed) / len(df)):.2f}%")

# Save the processed data to parquet
df_processed.to_parquet('processed_data/men_imbalanced_node_features_checked.parquet')

# Define a function to visualize mistakes with a heatmap
def visualize_mistakes_heatmap(data):
    # Define columns with potential issues
    issue_columns = ["ball_flying_issue", "player_number_issue", "success_label_issue"]
    
    # Create a binary matrix (1 if marked with "X", 0 otherwise)
    binary_matrix = data[issue_columns].map(lambda x: 1 if x == "X" else 0)
    
    # Calculate the sum of issues per game ID
    issue_summary = binary_matrix.sum(axis=1)

    # Create a heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(binary_matrix.T, cmap="Reds", cbar_kws={'label': 'Issue Marked (1 = X, 0 = OK)'}, linewidths=0.5)
    plt.title("Heatmap of Marked Issues in Dataset")
    plt.xlabel("Game Index")
    plt.ylabel("Issue Type")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.show()

if __name__ == "__main__":   
    # Call the function
    visualize_mistakes_heatmap(check_df[:300])