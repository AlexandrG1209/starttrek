import argparse

import gymnasium as gym
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Tester LunarLander avec des actions aleatoires")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    if args.episodes < 1:
        parser.error("Le nombre d'episodes doit etre positif")
    if args.seed < 0:
        parser.error("La graine doit etre positive ou nulle")

    render_mode = None if args.no_render else "human"
    env = gym.make("LunarLander-v3", render_mode=render_mode)
    scores = []

    print(f"Etat : {env.observation_space.shape[0]} valeurs")
    print(f"Actions : {env.action_space.n}")

    try:
        for episode in range(args.episodes):
            seed = args.seed + episode
            state, info = env.reset(seed=seed)
            env.action_space.seed(seed)
            total_reward = 0.0
            steps = 0
            done = False

            while not done:
                # Random actions give us a score to compare with later.
                action = env.action_space.sample()
                state, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                steps += 1
                done = terminated or truncated

            scores.append(total_reward)
            print(f"Episode {episode + 1} : score = {total_reward:.2f}, pas = {steps}")

        print(f"Score moyen : {np.mean(scores):.2f}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
