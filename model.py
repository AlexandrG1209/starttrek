import sys

import gymnasium as gym
import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

    def forward(self, state):
        return self.network(state)


class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, action_size),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, state):
        return self.actor(state), self.critic(state).squeeze(-1)


def load_model(algorithm, path):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    weights = saved.get("model_state_dict", saved)
    if algorithm == "dqn":
        hidden_size, state_size = weights["network.0.weight"].shape
        action_size = weights["network.4.weight"].shape[0]
        model = DQN(state_size, action_size, hidden_size)
    elif algorithm == "ppo":
        hidden_size, state_size = weights["actor.0.weight"].shape
        action_size = weights["actor.4.weight"].shape[0]
        model = ActorCritic(state_size, action_size, hidden_size)
    else:
        raise ValueError("Algorithm must be dqn or ppo")
    model.load_state_dict(weights)
    model.eval()
    return model


def main():
    if len(sys.argv) == 2 and sys.argv[1] in ["-h", "--help"]:
        print("USAGE\n    python model.py ALGO MODEL [EPISODES]\n\nDESCRIPTION\n"
              "    ALGO        algorithm of the saved model: dqn or ppo\n"
              "    MODEL       path to the saved model file\n"
              "    EPISODES    number of episodes to run (default: 1)")
        return 0
    if len(sys.argv) not in [3, 4]:
        raise ValueError("Usage: python model.py ALGO MODEL [EPISODES]")
    algorithm = sys.argv[1]
    if algorithm not in ["dqn", "ppo"]:
        raise ValueError("Algorithm must be dqn or ppo")
    episodes = int(sys.argv[3]) if len(sys.argv) == 4 else 1
    if episodes < 1:
        raise ValueError("Episodes must be positive")

    torch.set_num_threads(1)
    model = load_model(algorithm, sys.argv[2])
    env = gym.make("LunarLander-v3", render_mode="human")
    try:
        for episode in range(episodes):
            state, info = env.reset()
            score = 0.0
            done = False
            with torch.no_grad():
                while not done:
                    values = model(torch.tensor(state, dtype=torch.float32))
                    if algorithm == "ppo":
                        values = values[0]
                    action = values.argmax().item()
                    state, reward, terminated, truncated, info = env.step(action)
                    score += reward
                    done = terminated or truncated
            print(f"Episode {episode + 1}: score = {score:.2f}")
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(84)
