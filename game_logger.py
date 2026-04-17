import csv
import os
from datetime import datetime

# ================================================================
# game_logger.py — Records every game the ML plays to a CSV file
#
# HOW TO USE:
#   1. Place this file in your MLProject folder
#   2. In DotsAgent.cs, the Unity side sends game results
#      via the reward signal — this script reads the 
#      TensorBoard stats files and converts them to CSV
#
# Run AFTER training (or during):
#   python game_logger.py
# ================================================================

RESULTS_DIR = "results"
RUN_ID      = "dots_v1"
OUTPUT_CSV  = "game_history.csv"

def parse_tensorboard_to_csv():
    """
    Reads ML-Agents training stats and writes them to a CSV.
    ML-Agents saves stats in results/run_id/run_id-0.csv automatically.
    This script cleans and reformats that data.
    """

    # ML-Agents already generates a stats CSV — find it
    stats_path = os.path.join(RESULTS_DIR, RUN_ID, f"{RUN_ID}-0.csv")
    
    if not os.path.exists(stats_path):
        # Try finding any csv in results folder
        for root, dirs, files in os.walk(RESULTS_DIR):
            for f in files:
                if f.endswith(".csv"):
                    stats_path = os.path.join(root, f)
                    break

    if not os.path.exists(stats_path):
        print(f"No stats file found yet.")
        print(f"Train first, then run this script.")
        print(f"Expected location: {stats_path}")
        return

    print(f"Reading stats from: {stats_path}")

    rows = []
    with open(stats_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        print("Stats file is empty — train more first.")
        return

    # Write cleaned CSV
    output_path = OUTPUT_CSV
    with open(output_path, "w", newline="") as f:
        fieldnames = [
            "step",
            "cumulative_reward",
            "episode_length",
            "policy_loss",
            "value_loss",
            "learning_rate",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow({
                "step":               row.get("Step", ""),
                "cumulative_reward":  row.get("Environment/Cumulative Reward", ""),
                "episode_length":     row.get("Environment/Episode Length", ""),
                "policy_loss":        row.get("Losses/Policy Loss", ""),
                "value_loss":         row.get("Losses/Value Loss", ""),
                "learning_rate":      row.get("Policy/Learning Rate", ""),
            })

    print(f"\nSaved {len(rows)} training steps to: {output_path}")
    print(f"\nFirst few rows:")
    print(f"{'Step':<12} {'Reward':<12} {'Ep Length':<12}")
    print("-" * 36)
    for row in rows[:5]:
        step    = row.get("Step", "?")
        reward  = row.get("Environment/Cumulative Reward", "?")
        ep_len  = row.get("Environment/Episode Length", "?")
        try:
            print(f"{int(float(step)):<12} {float(reward):<12.3f} {float(ep_len):<12.1f}")
        except:
            print(f"{step:<12} {reward:<12} {ep_len:<12}")


def create_manual_game_log():
    """
    Creates a separate CSV that logs individual game outcomes.
    Call this from your own analysis after training.
    """
    output_path = "individual_games.csv"
    
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "game_number",
            "winner",           # 1=Player1, 2=AI, 0=Draw
            "ai_score",
            "player_score",
            "total_moves",
            "timestamp"
        ])

    print(f"Created empty game log: {output_path}")
    print("This gets filled in when you add logging to DotsAgent.cs")
    return output_path


if __name__ == "__main__":
    print("\n" + "="*50)
    print("  Dots and Boxes — Game Logger")
    print("="*50 + "\n")
    
    parse_tensorboard_to_csv()
    print()
    create_manual_game_log()