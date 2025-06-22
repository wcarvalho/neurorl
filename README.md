# neurorl

Some deep reinforcement learning experiments.
The following recurrent agents are supported:

1. [Q-learning](td_agents/q_learning.py)
2. [Successor Features](td_agents/usfa.py)
3. [MuZero](td_agents/muzero.py)

## Installation

We recommend creating a new Python 3.10 virtual environment.
We recommend the excellent tool [uv](https://docs.astral.sh/uv/) for managing Python projects.
To create a virtual environment in `.venv` and install dependencies:

```bash
python3 -m venv .venv  # or uv venv --python 3.10
source .venv/bin/activate
pip3 install -e .  # or uv sync
```

## Cluster instructions

On the FAS cluster, load the modules for Python 3.10 and CUDA
*before* running the installation above:

```bash
module load python/3.10.13-fasrc01  # if not using uv
module load cuda/11.8.0-fasrc01 cudnn/8.9.2.26_cuda11-fasrc01
```

After creating the venv,
run the following,
which will automatically load CUDA whenever activating the venv:

```bash
cat << 'EOF' >> .venv/bin/activate

# load CUDA modules on FAS RC
module load cuda/11.8.0-fasrc01 cudnn/8.9.2.26_cuda11-fasrc01
EOF
```

Note that `python/3.10.13-fasrc01` must be loaded each time before activating the venv
to avoid replacing the venv's Python.

## Running experiments

To get started, run

```bash
python configs/catch_trainer.py \
  --parallel='none' \
  --run_distributed=False \
  --debug=False \
  --use_wandb=False \
  --search='baselines'
```

Experiments are defined in the `configs` directory.
To make your own experiment,
copy one of the configs (e.g. [catch_trainer.py](configs/catch_trainer.py)).
You will need to change two functions:

1. `make_environment`: this function specifies how environments are constructed.
   This codebase assumes [`dm_env`](https://github.com/google-deepmind/dm_env) environments so make sure to convert `gym` environments to `dm_env`.
2. `setup_experiment_inputs`: this function specifies how agents are loaded. In the example given, a q-learning agent is loaded.

Agents in the `td_agents` directory (e.g. [q_learning.py](td_agents/q_learning.py))
are defined with 3 components:

1. a `Config` dataclass that specifies the default hyperparameter values.
2. a **loss function** that specifies how the learner/replay buffer/actor will be created.
   You mainly change this object in order to change something about learning.
3. a **network factory function** that creates the neural networks that define the agnet.
