# Install

[FAS Install and Setup](install-fas.md)


# Running experiments

load environment
```
source activate neurorl 
```

**how do experiments work?**

Experiments are defined by configs. To make your own experiment, copy one of the configs (e.g. [catch_trainer.py](configs/catch_trainer.py)). You will need to change two functions:
1. `make_environment`: this function specifies how environments are constructed. This codebase assumes `dm_env` environments so make sure to convert `gym` environments to `dm_env`.
2. `setup_experiment_inputs`: this function specifies how agents are loaded. In the example given, a q-learning agent is loaded.

Agents are defined with 3 things (e.g. [catch_trainer.py](configs/catch_trainer.py#L124)):
1. a config ([example](td_agents/q_learning.py#L27)), which specified default values
2. a builder ([example](td_agents/q_learning.py#L30)), which specifies how the learner/replay buffer/actor will be created. you mainly change this object in order to change something about learning.
3. a network_factory ([example](td_agents/q_learning.py#L11)), which creates the neural networks that define the agnet.




# Available (Recurrent) Agents

1. [Q-learning](td_agents/q_learning.py)
2. [Successor Features](td_agents/usfa.py)
3. [MuZero](td_agents/muzero.py)
