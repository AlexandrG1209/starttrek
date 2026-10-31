# Start Trek

Reinforcement learning project on LunarLander-v3.
This branch contains the DQN work.

## Setup

Run these commands in Ubuntu, from the project folder.
The environment was tested with Python 3.14.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install swig==4.5.1 setuptools wheel
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install --no-build-isolation -r requirements.txt
```

SWIG builds Box2D, which also needs a C++ compiler.
The `--no-build-isolation` option lets Box2D use the installed SWIG package.
If the tools are missing on Ubuntu:

```bash
sudo apt install python3-venv build-essential
```

We use the CPU version of PyTorch for now.
Activate `.venv` with `source .venv/bin/activate` in each new terminal.

## Run the simulator

```bash
python random_agent.py
```

A window opens and the lander takes random actions.
The script prints the score and number of steps for each episode.
There is no learning in this script.

To run several episodes without a window:

```bash
python random_agent.py --episodes 10 --no-render
```

The default seed is 42. Use `--seed 10` to change it.
The seed fixes the starting state and random actions so we can repeat the test.

The first test with 10 episodes and seed 42 gave a mean score of -203.96.
This is a starting score to compare with the DQN later.

## Notebook

```bash
jupyter lab
```

Open `dqn.ipynb` and run the cells in order.
It contains the settings, random agent test and score plot.

The mean random score over 100 episodes is -193.03 with seed 42.
The DQN network has two hidden layers of 64 neurons and one output per action.
A cell checks the output shape and shows the action with the highest value.
The replay buffer keeps up to 100,000 experiences and samples batches of 64 as tensors.
There is also a small test to check that old experiences are replaced when it is full.
Action selection uses epsilon-greedy. Epsilon goes from 1 to 0.05 over 100,000 steps.
The tests check random choices and network choices. Evaluation always takes the highest Q-value.
The target network starts with the same weights as the main network and stays fixed between copies.
It is set to copy the main weights every 1,000 learning updates. The notebook checks the copy and its timing.
The notebook now has a first training run of 20 episodes, with Bellman targets, MSE loss and Adam.
It records scores, lengths, losses, epsilon and episode endings. The plots show the first training results.
Trial weights are saved in `checkpoints/dqn_short_run.pt`. This folder is ignored by Git.
Every 5 training episodes, the model plays 5 evaluation episodes without random actions or learning.
These tests use a separate environment with fixed seeds starting at 10,000.
The best mean score saves the model in `checkpoints/dqn_best.pt`.
The notebook reloads this model and checks its evaluation score.
Both saved files are local trial models. A longer training run and the final 100-episode evaluation come next.
