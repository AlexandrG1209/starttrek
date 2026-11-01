# Start Trek

We train DQN and PPO agents on LunarLander-v3.
Erwan worked on DQN and Alexandr on PPO.

## Setup

Use Python 3.12 or newer.

On Ubuntu, install the build tools first:

```bash
sudo apt install python3-venv build-essential swig
```

On macOS, install SWIG with `brew install swig`.

Then create the environment from the project folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-build-isolation -r requirements.txt
```

## Run a model

```bash
python model.py dqn DQN/dqn_lunar_lander.pt
python model.py ppo PPO/ppo_lunar_lander_stable_finetuned_best.pt
```

A window opens and the script prints the score.
Add a number at the end to run more episodes, for example `3`.
Use `python model.py -h` for help.

## Notebooks

```bash
jupyter lab
```

Select the `.venv` Python kernel and run the cells in order.

- `DQN/dqn.ipynb`: DQN code, training and results.
- `PPO/PPO.ipynb`: PPO code, training and fine-tuning.
- `comparison.ipynb`: test both saved models without training.

The DQN notebook loads saved training results by default.
Set `RETRAIN = True` to train it again.
The PPO notebook runs training, so use the comparison notebook to check the saved models quickly.

## Results

The comparison uses the same 100 episode seeds, from 50000 to 50099.
Both models choose their best action without exploration.

The results saved in the comparison notebook are:

| Model | Mean score |
| --- | ---: |
| DQN | 249.45 |
| PPO | 230.06 |

Both means are above the target of 200. DQN scored higher in this run.
The notebooks contain the full results and plots.

## Other files

- `model.py`: load a model and play with a window.
- `random_agent.py`: test LunarLander with random actions.
- `DQN/models/`: DQN weights from the training experiments.
- `DQN/results/`: DQN training and evaluation logs.
