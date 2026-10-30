# Start Trek

Projet d'apprentissage par renforcement sur LunarLander-v3.
Cette branche contient ma partie DQN.

## Installation

Commandes a lancer dans Ubuntu, depuis le dossier du projet.
L'environnement a ete teste avec Python 3.14.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install swig==4.5.1 setuptools wheel
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install --no-build-isolation -r requirements.txt
```

SWIG sert a compiler Box2D. Il faut aussi un compilateur C++.
L'option `--no-build-isolation` permet a Box2D d'utiliser SWIG deja installe.
Si les outils manquent sur Ubuntu :

```bash
sudo apt install python3-venv build-essential
```

PyTorch est installe en version CPU pour commencer.
Il faut reactiver `.venv` avec `source .venv/bin/activate` dans chaque nouveau terminal.

## Tester le simulateur

```bash
python random_agent.py
```

Une fenetre s'ouvre et le module choisit ses actions au hasard.
Le script affiche le score et le nombre de pas de chaque episode.
Il ne fait pas encore d'apprentissage.

Pour lancer plusieurs episodes sans ouvrir de fenetre :

```bash
python random_agent.py --episodes 10 --no-render
```

La graine vaut 42 par defaut. On peut la changer avec `--seed 10`.
Elle fixe le depart du simulateur et le tirage des actions pour refaire le meme essai.

Premier essai avec 10 episodes et la graine 42 : score moyen de -203.96.
Ce resultat sert de point de depart pour comparer avec le DQN plus tard.

## Notebooks

```bash
jupyter lab
```

Ouvrir `dqn.ipynb` et executer les cellules dans l'ordre.
Il contient les parametres, le test de l'agent aleatoire et le graphique des scores.

Sur 100 episodes avec la graine 42, le score moyen est de -193.03.
Le reseau DQN est ajoute : deux couches de 64 neurones, avec une valeur en sortie par action.
Une cellule verifie la taille de la sortie et affiche l'action avec la plus grande valeur.
Le replay buffer garde jusqu'a 100 000 experiences et renvoie des lots aleatoires de 64 experiences en tenseurs.
Le notebook verifie aussi le remplacement des anciennes experiences quand la memoire est pleine.
Le choix des actions utilise epsilon-greedy. Epsilon baisse de 1 a 0.05 sur 100 000 pas.
Deux tests montrent les actions au hasard et le choix du reseau. Pour evaluer, on utilisera epsilon a 0.
Le reseau n'est pas encore entraine. La prochaine etape est le reseau cible, puis le calcul de la perte.
