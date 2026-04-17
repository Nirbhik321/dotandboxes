import socket
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import os
import csv
from datetime import datetime

class DotsAndBoxesEnv(gym.Env):
    def __init__(self, host="127.0.0.1", port=5555):
        super().__init__()
        self.host = host
        self.port = port
        self.sock = None
        self.observation_space = spaces.Box(low=0, high=2, shape=(57,), dtype=np.float32)
        self.action_space = spaces.Discrete(40)
        self.game_number = 0
        self.csv_path = "game_history.csv"
        self._prev_ai_score = 0
        self._total_moves = 0
        self._init_csv()
        self._connect()

        # Load frozen opponent (past version of self)
        opponent_path = "results/dots_opponent.zip"
        if os.path.exists(opponent_path):
            print("Loaded frozen opponent for self-play!")
            self.opponent = PPO.load(opponent_path)
        else:
            print("No opponent found - using random moves for Player 1")
            self.opponent = None

    def _init_csv(self):
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["game_number","winner","ai_score",
                                  "player_score","total_moves","timestamp"])

    def _connect(self):
        print(f"Connecting to Unity on port {self.port}...")
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.host, self.port))
        self.sock.settimeout(120)
        print("Connected to Unity!")

    def _recv_state(self):
        data = b""
        while b"\n" not in data:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise ConnectionError("Unity disconnected")
            data += chunk
        line = data.decode("utf-8").strip()
        values = [float(x) for x in line.split(",")]
        obs = np.array(values[:57], dtype=np.float32)
        done = bool(values[57])
        p1_score = int(values[58])
        p2_score = int(values[59])
        return obs, done, p1_score, p2_score

    def _send_action(self, action):
        self.sock.sendall(f"{int(action)}\n".encode("utf-8"))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sock.sendall(b"reset\n")
        obs, done, p1_score, p2_score = self._recv_state()
        self._prev_ai_score = 0
        self._total_moves = 0

        # If it's Player 1's turn first, let opponent play
        obs = self._handle_opponent_turns(obs, done)
        return obs, {}

    def step(self, action):
        # Send AI (Player 2) action
        self._send_action(action)
        self._total_moves += 1
        obs, done, p1_score, p2_score = self._recv_state()

        ai_score = p2_score
        boxes_captured = ai_score - self._prev_ai_score
        self._prev_ai_score = ai_score

        reward = 0.0
        if boxes_captured > 0:
            reward += boxes_captured * 1.0

        if done:
            if p2_score > p1_score:   reward += 5.0;  winner = 2
            elif p2_score < p1_score: reward -= 5.0;  winner = 1
            else:                     reward += 1.0;  winner = 0
            self._log_game(winner, p2_score, p1_score)
        else:
            # Handle opponent (Player 1) turns
            obs = self._handle_opponent_turns(obs, done)

        return obs, reward, done, False, {}

    def _handle_opponent_turns(self, obs, done):
        # Keep playing opponent moves until it's Player 2's turn
        current_player = int(obs[56])
        while current_player == 1 and not done:
            if self.opponent is not None:
                p1_action, _ = self.opponent.predict(obs, deterministic=False)
            else:
                # Random move
                legal = [i for i in range(40) if obs[i] == 0]
                p1_action = np.random.choice(legal) if legal else 0

            self._send_action(p1_action)
            obs, done, p1_score, p2_score = self._recv_state()
            current_player = int(obs[56])

        return obs

    def _log_game(self, winner, ai_score, player_score):
        self.game_number += 1
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                self.game_number, winner, ai_score, player_score,
                self._total_moves,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ])

    def close(self):
        if self.sock:
            self.sock.close()


def train():
    print("\n" + "="*50)
    print("  Dots and Boxes — Stage 3 Self-Play Training")
    print("="*50)
    print("\n  1. Press Play in Unity first")
    print("  2. Wait for 'Listening on port 5555'")
    print("  3. Press Enter here\n")
    input("  Press Enter when Unity is ready...")

    env = DotsAndBoxesEnv()
    os.makedirs("results/checkpoints", exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="results/checkpoints/",
        name_prefix="dots_stage3",
        verbose=1
    )

    # Load existing model to continue training
    checkpoint_path = "results/dots_final_model.zip"
    if os.path.exists(checkpoint_path):
        print(f"\nLoading model from Stage 2...")
        model = PPO.load(checkpoint_path, env=env)
    else:
        print(f"\nNo model found - starting fresh...")
        model = PPO("MlpPolicy", env, verbose=1,
            learning_rate=3e-4, n_steps=2048, batch_size=128,
            n_epochs=10, gamma=0.99,
            tensorboard_log="results/tensorboard/")

    print("\nStage 3 training started — AI vs past self!")
    print("This is where real strategy emerges.\n")

    try:
        model.learn(
            total_timesteps=800_000,
            callback=checkpoint_callback,
            reset_num_timesteps=False,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\nStopping — saving...")

    model.save("results/dots_final_model")
    print("\nStage 3 complete! Model saved to results/dots_final_model.zip")
    env.close()


if __name__ == "__main__":
    train()