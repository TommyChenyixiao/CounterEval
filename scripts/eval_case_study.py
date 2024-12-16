import pandas as pd
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

def case_study_summary(game_id):
    df = pd.read_csv('processed_data/summary_results.csv')
    # columns: game_id,player_num_label,avg_perf,max_perf,min_perf,num_frames,frames_above_avg
    plt.figure(figsize=(12, 6))

    # Plot each performance metric
    plt.plot(df["player_num_label"], df["avg_perf"], marker="o", label="Average Performance")
    plt.plot(df["player_num_label"], df["max_perf"], marker="o", label="Max Performance")
    plt.plot(df["player_num_label"], df["min_perf"], marker="o", label="Min Performance")

    x_values = range(1, len(df) + 1) 
    plt.xticks(x_values, df["player_num_label"], rotation=45, fontsize=10)
    # Adding titles and labels
    plt.title(f"Game ID: {game_id} Players' Performance Summary", fontsize=16)
    plt.xlabel("Players", fontsize=12)
    plt.ylabel("Performance Value", fontsize=12)
    plt.legend(title="Metrics", fontsize=10)
    plt.grid(alpha=0.5)

    # Show plot
    plt.show()

def case_study_time_series(game_id, player_num_label):
    df = pd.read_csv('processed_data/time_series_results.csv')
    # columns: game_id,frame_id,player_num_label,actual_prob,cf_prob,perf
    # two plot stack together (2,1)
    # on (1,1) frame_id vs perf
    # on (2,1) frame_id vs actual_prob, cf_prob
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    df_player = df[(df["game_id"] == game_id) & (df["player_num_label"] == player_num_label)]
    plt.plot(df_player["frame_id"], df_player["perf"], label="Performance")
    plt.title(f"Game ID: {game_id} Player {player_num_label} Performance Time Series", fontsize=16)
    plt.xlabel("Frame ID", fontsize=12)
    plt.ylabel("Performance Value", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.5)

    plt.subplot(2, 1, 2)
    plt.plot(df_player["frame_id"], df_player["actual_prob"], label="Actual Probability")
    plt.plot(df_player["frame_id"], df_player["cf_prob"], label="Counterfactual Probability")
    plt.title(f"Game ID: {game_id} Player {player_num_label} Probability Time Series", fontsize=16)
    plt.xlabel("Frame ID", fontsize=12)
    plt.ylabel("Probability Value", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.5)

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    # case_study_summary(198)
    case_study_time_series(198, 1)
    case_study_time_series(198, 8)
    case_study_time_series(198, 14)
