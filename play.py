import socket
import numpy as np
from stable_baselines3 import PPO

# ================================================================
# play.py — Human vs Trained AI
# 
# 1. Press Play in Unity
# 2. Run: python play.py
# 3. Click edges in Unity to make your moves (you are Player 1)
# 4. AI automatically responds as Player 2
# ================================================================

HOST = "127.0.0.1"
PORT = 5555
MODEL_PATH = "results/dots_final_model.zip"

def play():
    print("\n" + "="*50)
    print("  Dots and Boxes — Human vs AI")
    print("="*50)
    print(f"\n  Loading model from {MODEL_PATH}...")

    model = PPO.load(MODEL_PATH)
    print("  Model loaded!")

    print("\n  1. Press Play in Unity")
    print("  2. Press Enter here when Unity says 'Listening on port 5555'\n")
    input("  Press Enter when Unity is ready...")

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((HOST, PORT))
    sock.settimeout(300)  # 5 min timeout — plenty of time for human moves
    print("\n  Connected! You are Player 1 — click edges in Unity.")
    print("  AI is Player 2 — it will respond automatically.\n")

    games_played = 0
    ai_wins = 0
    human_wins = 0
    draws = 0

    while True:
        try:
            # Get initial state
            data = b""
            while b"\n" not in data:
                data += sock.recv(4096)
            line = data.decode("utf-8").strip()
            values = [float(x) for x in line.split(",")]

            obs = np.array(values[:57], dtype=np.float32)
            done = bool(values[57])
            p1_score = int(values[58])
            p2_score = int(values[59])
            current_player = int(obs[56])

            if done:
                games_played += 1
                if p2_score > p1_score:
                    ai_wins += 1
                    print(f"\n  Game {games_played}: AI wins! ({p2_score} vs {p1_score})")
                elif p1_score > p2_score:
                    human_wins += 1
                    print(f"\n  Game {games_played}: You win! ({p1_score} vs {p2_score})")
                else:
                    draws += 1
                    print(f"\n  Game {games_played}: Draw! ({p1_score} vs {p2_score})")

                print(f"  Record — You: {human_wins} | AI: {ai_wins} | Draws: {draws}")
                print("\n  Starting new game...")
                sock.sendall(b"reset\n")
                continue

            # If it's AI's turn (Player 2), pick action
            if current_player == 2:
                action, _ = model.predict(obs, deterministic=True)
                sock.sendall(f"{int(action)}\n".encode("utf-8"))
                print(f"  AI plays edge {int(action)}")
            # If it's human's turn (Player 1), Unity waits for you to click
            # Nothing to send — Unity handles the click and sends new state

        except KeyboardInterrupt:
            print(f"\n\nFinal Record:")
            print(f"  You:   {human_wins} wins")
            print(f"  AI:    {ai_wins} wins")
            print(f"  Draws: {draws}")
            break
        except Exception as e:
            print(f"\nError: {e}")
            break

    sock.close()

if __name__ == "__main__":
    play()