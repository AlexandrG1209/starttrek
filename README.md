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

## Run a saved model

```bash
python model.py dqn dqn_lunar_lander.pt
python model.py ppo ppo_lunar_lander.pt 3
python model.py -h
```

The simulator opens a window and prints the score of each episode.
Both models choose the highest output without exploration.
Errors go to stderr with exit code 84. A successful run returns 0.

## Notebooks

```bash
jupyter lab
```

Open `dqn.ipynb` or `ppo.ipynb` and run the cells in order.
The PPO implementation comes from Alexandr's branch, commit 85b15e4.
Its learning formulas are unchanged. We added headings, timing and evaluation without action sampling.

The DQN notebook loads the saved training results by default.
Set `RETRAIN = True` in its settings to train again.
This runs four experiments of 300,000 steps each, finishing the last episode.
Final evaluation still runs on the saved models when `RETRAIN = False`.

The PPO notebook runs the existing 500 updates again.
It saves that rerun as `models/ppo_reproduced.pt` so the original PPO checkpoint stays available.

## DQN settings

| Setting | Value | Reason |
| --- | --- | --- |
| Hidden layers | 64, 64 with ReLU | Small network for 8 inputs |
| Batch size | 64 | Mix past experiences at each update |
| Replay capacity | 100,000 | Reuse older steps while replacing the oldest |
| Gamma | 0.99 | Keep future rewards important |
| Learning rate | 0.001 and 0.0005 | Compare a smaller update size |
| Epsilon | 1 to 0.05 over 100,000 steps | Explore first, then follow the network more often |
| Target copy | Every 1,000 learning updates | Keep the target fixed between copies |
| Training | At least 300,000 steps per run | Longer budget after the 20-episode check |
| Validation | 10 episodes every 10,000 steps | Select a checkpoint without using final test seeds |
| CPU threads | 1 | Enough for these small networks |

The buffer stores `terminated`, so time limits still allow future rewards in the Bellman target.
The loss is MSE. Adam updates the main network at every step after the first 64 experiences.
The original settings are tested with training seeds 42, 43 and 44.
The learning-rate comparison changes only the learning rate for seed 42.

## Results

The final DQN uses learning rate 0.0005 and seed 42, chosen from validation scores.
The random baseline mean is -193.03 over 100 episodes.
Each row below is 100 consecutive test episodes without exploration, using the same starts for both models.
The PPO column uses the checkpoint from Alexandr's commit 85b15e4.

| Test seeds | DQN mean | DQN std | PPO mean | PPO std |
| --- | ---: | ---: | ---: | ---: |
| 20000-20099 | 247.55 | 51.59 | -212.12 | 28.03 |
| 30000-30099 | 253.59 | 56.24 | -213.88 | 26.41 |
| 40000-40099 | 253.29 | 47.46 | -216.17 | 25.91 |

The saved DQN passes a mean of 200 on all three test ranges.
It still has some low scores and failures. The notebook shows the full distribution and episode endings.
The original DQN learning rate was less consistent across training seeds: means 220.24, 178.68 and 196.29.
One successful checkpoint does not mean every training seed will succeed.

The current PPO checkpoint does not reach the target without exploration.
Its original notebook evaluated sampled actions, which gives a different result from choosing the highest output.
The separate 500-update rerun also stays below the target: mean -426.84 on test seeds 20000-20099.

The selected DQN run took 365.7 training seconds for 300,155 steps.
The PPO rerun took 55.0 seconds for 128,256 collected steps.
The original PPO checkpoint's training time was not recorded.
These are measurements on this machine, with different budgets; they do not prove that one algorithm is always faster.

## Files

- `dqn_lunar_lander.pt`: final DQN weights used by model.py.
- `ppo_lunar_lander.pt`: original PPO weights from commit 85b15e4.
- `models/`: best DQN weights for each experiment and the separate PPO rerun.
- `results/`: episode scores, lengths, endings, losses, epsilon values and evaluation results.
- `checkpoints/`: ignored files from the earlier short tests.

## Understanding the DQN

1. Epsilon chooses between a random action and the highest Q-value.
2. The replay buffer stores what happened.
3. A random batch gives 64 experiences for learning.
4. The target network estimates future rewards.
5. MSE measures the prediction error and Adam changes the main weights.
6. Evaluation tests the current behaviour and saves the best model.

The DQN work and common launcher are ready. PPO still needs a stronger model without exploration.
Its owner also needs to finish the PPO training analysis, training-seed tests and variation experiment before the whole submission is ready.
