# Start Trek — LunarLander with DQN and PPO

This project trains two reinforcement-learning agents for the Gymnasium `LunarLander-v3` environment:

- **DQN** learns one Q-value for each action.
- **PPO** learns an action policy with an Actor and estimates state values with a Critic.

The repository also contains a small notebook that compares the two saved models under the same evaluation conditions.

## Setup

Python 3.12 is recommended for the project.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-build-isolation -r requirements.txt
```

On macOS, install the SWIG command first if it is missing:

```bash
brew install swig
```

On Ubuntu, the required build tools can be installed with:

```bash
sudo apt install python3-venv build-essential swig
```

In VS Code, select `.venv/bin/python` as the notebook kernel.

## Main files

- `dqn.ipynb` — Erwan's DQN implementation, training experiments, graphs, and conclusions.
- `PPO.ipynb` — Alexandr's PPO implementation, stable training, fine-tuning, graphs, and explanations.
- `comparison.ipynb` — direct evaluation of the two selected saved models.
- `model.py` — opens LunarLander and runs a saved DQN or PPO model without exploration.
- `random_agent.py` — runs a random baseline agent.
- `results/` — raw training and evaluation evidence.
- `models/` — saved DQN checkpoints from the training experiments.

## Run the notebooks

Open the project in VS Code, activate the virtual environment, and select the `.venv` kernel. The notebooks can then be run from top to bottom:

1. `dqn.ipynb` for DQN;
2. `PPO.ipynb` for PPO;
3. `comparison.ipynb` for the final model comparison.

The DQN notebook loads its saved training results by default. Set `RETRAIN = True` only if you want to repeat its long training experiments. The PPO notebook also contains training cells, so the separate comparison notebook should be used when you only want to evaluate the final models.

## Run a saved model

```bash
python model.py -h
python model.py dqn dqn_lunar_lander.pt
python model.py ppo ppo_lunar_lander_stable_finetuned_best.pt
python model.py ppo ppo_lunar_lander_stable_finetuned_best.pt 3
```

`model.py` opens a display window, selects the action with the largest output, and prints the score of every episode. It returns exit code `84` for an error and `0` after a successful run.

## DQN configuration

The final DQN checkpoint is `dqn_lunar_lander.pt`.

| Setting | Value |
| --- | ---: |
| Hidden layers | 64, 64 with ReLU |
| Batch size | 64 |
| Replay capacity | 100,000 |
| Discount factor | 0.99 |
| Selected learning rate | 0.0005 |
| Epsilon | 1.0 to 0.05 over 100,000 steps |
| Target-network update | Every 1,000 learning updates |
| Training budget | About 300,000 steps per run |

The DQN work includes three training seeds and a learning-rate comparison. Its replay buffer keeps past transitions, while the target network gives more stable Bellman targets.

## PPO configuration

The selected PPO checkpoint is `ppo_lunar_lander_stable_finetuned_best.pt`.

| Setting | Value |
| --- | ---: |
| Actor and Critic hidden layers | 64, 64 with Tanh |
| Rollout length | 1,024 steps |
| PPO epochs | 4 |
| Mini-batch size | 64 |
| Discount factor | 0.99 |
| GAE lambda | 0.95 |
| PPO clip range | 0.8 to 1.2 |
| Main learning rate | 0.0001 |
| Fine-tuning learning rate | 0.00003 |

PPO collects fresh rollouts, calculates advantages with GAE, and updates the Actor and Critic with the clipped PPO loss. The final checkpoint comes from the additional fine-tuning stage.

## Direct model comparison

`comparison.ipynb` does not train either model. It loads both checkpoints and evaluates them with:

- the same 100 held-out consecutive seeds, from `50000` to `50099`;
- deterministic `argmax` actions;
- no exploration;
- the same `LunarLander-v3` environment.

The notebook calculates the complete episode rewards, lengths, and metrics directly.

| Model | Mean | Std | Median | Minimum | Maximum | Episodes ≥ 200 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DQN | 249.45 | 54.41 | 263.03 | 14.00 | 318.42 | 87/100 |
| PPO | 230.06 | 67.77 | 250.56 | -30.57 | 292.47 | 84/100 |

For these saved checkpoints and these 100 held-out episode starts, DQN has the higher mean and more episodes above 200. PPO also solves the environment on average, but its reward distribution has more variation.

This is not proof that DQN is always better than PPO. The checkpoints were produced with different algorithms, training budgets, and experiment histories. The comparison only measures the two delivered models under one shared evaluation protocol.

## Comparison graphs

The comparison notebook produces three graphs:

1. reward for each evaluation episode;
2. overlapping reward histograms;
3. a boxplot showing the median, spread, and unusual results.

The red dashed line marks the project target of 200 points.
