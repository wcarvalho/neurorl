"""
Running experiments:

# DEBUGGING, single stream
python -m ipdb -c continue projects/human_sf/trainer_v2.py \
  --parallel='none' \
  --run_distributed=False \
  --debug=True \
  --use_wandb=True \
  --wandb_entity=wcarvalho92 \
  --wandb_project=human_objects_sf_debug \
  --search='flat_usfa'

# DEBUGGING, without jit
JAX_DISABLE_JIT=1 python -m ipdb -c continue projects/human_sf/trainer_v2.py \
  --parallel='none' \
  --run_distributed=False \
  --debug=True \
  --use_wandb=False \
  --wandb_entity=wcarvalho92 \
  --wandb_project=human_objects_sf_debug \
  --search='flat'


# DEBUGGING, parallel
python -m ipdb -c continue projects/human_sf/trainer_v2.py \
  --parallel='sbatch' \
  --debug_parallel=True \
  --run_distributed=False \
  --use_wandb=True \
  --wandb_entity=wcarvalho92 \
  --wandb_project=human_objects_sf_debug \
  --search='default'


# running, parallel
python projects/human_sf/trainer_v2.py \
  --parallel='sbatch' \
  --run_distributed=True \
  --use_wandb=True \
  --partition=kempner \
  --account=kempner_fellows \
  --wandb_entity=wcarvalho92 \
  --wandb_project=human_objects_sf \
  --max_concurrent=12 \
  --search='ambiguous_flat'

Change "search" to what you want to search over.

"""
import functools 
import json
from typing import Callable, Optional, Tuple

from enum import Enum

from absl import flags
from absl import app
from absl import logging
import dataclasses
import os
from ray import tune
from launchpad.nodes.python.local_multi_processing import PythonProcess
import launchpad as lp

from acme.agents.jax import actor_core as actor_core_lib
from acme import wrappers as acme_wrappers
from acme.jax import experiments
import dm_env
import haiku as hk
import jax

import minigrid

from projects.human_sf.minigrid_goto_wrapper import GotoOptionsWrapper
from envs.minigrid_wrappers import DictObservationSpaceWrapper
from library.dm_env_wrappers import GymWrapper
import library.experiment_builder as experiment_builder
import library.parallel as parallel
import library.utils as utils

from td_agents import basics

from td_agents import usfa
from projects.human_sf import q_learning
from projects.human_sf import object_q_learning
from projects.human_sf import muzero

from projects.human_sf import key_room_v3 as key_room


flags.DEFINE_string('config_file', '', 'config file')
flags.DEFINE_string('search', 'default', 'which search to use.')
flags.DEFINE_string(
    'parallel', 'none', "none: run 1 experiment. sbatch: run many experiments with SBATCH. ray: run many experiments with say. use sbatch with SLUM or ray otherwise.")
flags.DEFINE_bool(
    'debug', False, 'If in debugging mode, only 1st config is run.')
flags.DEFINE_bool(
    'make_path', True, 'Create a path under `FLAGS>folder` for the experiment')
flags.DEFINE_bool(
    'auto_name_wandb', True, 'automatically name wandb.')
FLAGS = flags.FLAGS

@dataclasses.dataclass
class QlearningConfig(q_learning.Config):
  samples_per_insert: float = 0.0

@dataclasses.dataclass
class ObjectQlearningConfig(object_q_learning.Config):
  samples_per_insert: float = 0.0


@dataclasses.dataclass
class UsfaConfig(usfa.Config):

  # arch
  final_conv_dim: int = 16
  conv_flat_dim: Optional[int] = 0
  sf_layers : Tuple[int]=(1024,)
  policy_layers : Tuple[int]=()

  # learner
  importance_sampling_exponent: float = 0.0
  samples_per_insert: float = 10.0
  sf_coeff: float = 10.0
  q_coeff: float = 1.0

  # eval actor
  eval_task_support: str = "train"  # options:

@dataclasses.dataclass
class MuZeroConfig(muzero.Config):
  """Configuration options for MuZero agent."""
  trace_length: int = 40
  min_scalar_value: Optional[float] = None
  num_bins: Optional[int] = 81  # number of bins for two-hot rep
  scalar_step_size: Optional[float] = None  # step size between bins
  value_target_source: str = 'return'

  value_layers: Tuple[int] = (512, 512)
  reward_layers: Tuple[int] = (128,)

def make_keyroom_env(seed: int,
                     room_size: int = 7,
                     evaluation: bool = False,
                     object_options: bool = False,
                     flat_task: bool = True,
                     steps_per_room: int=100,
                     swap_episodes: int = 100_000,
                     maze_idx: int = 0,
                     color_rooms: bool = True,
                     **kwargs) -> dm_env.Environment:
  """Loads environments.
  
  Args:
      evaluation (bool, optional): whether evaluation.
  
  Returns:
      dm_env.Environment: Multitask environment is returned.
  """
  del seed

  json_file = 'maze_pairs.json'
  with open(json_file, 'r') as file:
      maze_dicts = json.load(file)
  maze_dict = maze_dicts[maze_idx]

  # create gymnasium.Gym environment
  env = key_room.KeyRoom(
    room_size=room_size,
    maze_dict=maze_dict,
    max_steps_per_room=steps_per_room,
    flat_task=flat_task,
    swap_episodes=swap_episodes,
    color_rooms=color_rooms,
    training= not evaluation,
    **kwargs)

  ####################################
  # Gym wrappers
  ####################################
  gym_wrappers = [DictObservationSpaceWrapper]
  if object_options:
    gym_wrappers.append(functools.partial(
      GotoOptionsWrapper, use_options=object_options))
  
  # MUST GO LAST. GotoOptionsWrapper exploits symbolic obs
  gym_wrappers.append(functools.partial(
    minigrid.wrappers.RGBImgPartialObsWrapper, tile_size=8))

  for wrapper in gym_wrappers:
    env = wrapper(env)

  # convert to dm_env.Environment enironment
  env = GymWrapper(env)

  ####################################
  # ACME wrappers
  ####################################
  # add acme wrappers
  wrapper_list = [
    acme_wrappers.ObservationActionRewardWrapper,
    acme_wrappers.SinglePrecisionWrapper,
  ]

  return acme_wrappers.wrap_all(env, wrapper_list)

def setup_experiment_inputs(
    make_environment_fn: Callable,
    env_get_task_name: Optional[Callable[[dm_env.Environment], str]] = None,
    agent_config_kwargs: dict=None,
    env_kwargs: dict=None,
    debug: bool = False,
  ):
  """Setup."""
  config_kwargs = agent_config_kwargs or dict()
  env_kwargs = env_kwargs or dict()

  # -----------------------
  # load agent config, builder, network factory
  # -----------------------
  agent = agent_config_kwargs.get('agent', '')
  assert agent != '', 'please set agent'

  #################################
  # flat agents
  #################################

  if agent == 'flat_q':
    # has no mechanism to select from object options since dependent on what agent sees
    env_kwargs['object_options'] = False

    config = QlearningConfig(**config_kwargs)
    builder = basics.Builder(
      config=config,
      get_actor_core_fn=basics.get_actor_core,
      LossFn=q_learning.R2D2LossFn(
        discount=config.discount,
        importance_sampling_exponent=config.importance_sampling_exponent,
        burn_in_length=config.burn_in_length,
        max_replay_size=config.max_replay_size,
        max_priority_weight=config.max_priority_weight,
        bootstrap_n=config.bootstrap_n,
      ))
    network_factory = functools.partial(
      q_learning.make_minigrid_networks,
      config=config,
      task_encoder=lambda obs: hk.nets.MLP(
        (128, 128), activate_final=True)(obs['task']))


  elif agent == 'flat_usfa':
    # has no mechanism to select from object options since dependent on what agent sees
    env_kwargs['object_options'] = False

    config = UsfaConfig(**config_kwargs)
    builder = basics.Builder(
      config=config,
      ActorCls=functools.partial(
        basics.BasicActor,
        observers=[usfa.SFsObserver(period=1 if debug else 500)]
      ),
      get_actor_core_fn=functools.partial(
        basics.get_actor_core,
        extract_q_values=lambda preds: preds.q_values,
        ),
      LossFn=usfa.UsfaLossFn(
        discount=config.discount,
        importance_sampling_exponent=config.importance_sampling_exponent,
        burn_in_length=config.burn_in_length,
        max_replay_size=config.max_replay_size,
        max_priority_weight=config.max_priority_weight,
        bootstrap_n=config.bootstrap_n,
        sf_coeff=config.sf_coeff,
        loss_fn=config.sf_loss,
        q_coeff=config.q_coeff,
      ))
    network_factory = functools.partial(
            usfa.make_minigrid_networks, config=config)

  elif agent == 'flat_muzero':
    # has no mechanism to select from object options since dependent on what agent sees
    env_kwargs['object_options'] = False

    config = MuZeroConfig(**config_kwargs)

    import mctx
    # currently using same policy in learning and acting
    mcts_policy = functools.partial(
        mctx.gumbel_muzero_policy,
        max_depth=config.max_sim_depth,
        num_simulations=config.num_simulations,
        gumbel_scale=config.gumbel_scale)

    discretizer = None

    builder = basics.Builder(
        config=config,
        get_actor_core_fn=functools.partial(
            muzero.muzero_policy_act_mcts_eval,
            mcts_policy=mcts_policy,
            discretizer=discretizer,
        ),
        optimizer_cnstr=muzero.muzero_optimizer_constr,
        LossFn=muzero.MuZeroLossFn(
            discount=config.discount,
            importance_sampling_exponent=config.importance_sampling_exponent,
            burn_in_length=config.burn_in_length,
            max_replay_size=config.max_replay_size,
            max_priority_weight=config.max_priority_weight,
            bootstrap_n=config.bootstrap_n,
            value_target_source=config.value_target_source,
            discretizer=discretizer,
            mcts_policy=mcts_policy,
            simulation_steps=config.simulation_steps,
            reanalyze_ratio=config.reanalyze_ratio,
            root_policy_coef=config.root_policy_coef,
            root_value_coef=config.root_value_coef,
            model_policy_coef=config.model_policy_coef,
            model_value_coef=config.model_value_coef,
            model_reward_coef=config.model_reward_coef,
        ))
    network_factory = functools.partial(
        muzero.make_minigrid_networks,
        config=config,
        task_encoder=lambda obs: hk.nets.MLP(
              (128, 128), activate_final=True)(obs['task']))
  #################################
  # object centric agents
  #################################
  elif agent == 'object_q':
    # has no mechanism to select from object options since dependent on what agent sees
    env_kwargs['object_options'] = True

    config = ObjectQlearningConfig(**config_kwargs)
    builder = basics.Builder(
      config=config,
      get_actor_core_fn=functools.partial(
        object_q_learning.get_actor_core,
      ),
      LossFn=q_learning.R2D2LossFn(
        discount=config.discount,
        importance_sampling_exponent=config.importance_sampling_exponent,
        burn_in_length=config.burn_in_length,
        max_replay_size=config.max_replay_size,
        max_priority_weight=config.max_priority_weight,
        bootstrap_n=config.bootstrap_n,
      ))
    network_factory = functools.partial(
      object_q_learning.make_minigrid_networks,
      config=config,
      task_encoder=lambda obs: hk.nets.MLP(
        (128, 128), activate_final=True)(obs))

  elif agent == 'object_usfa':
    config = usfa.Config(**config_kwargs)
    builder = basics.Builder(
      config=config,
      get_actor_core_fn=functools.partial(
        basics.get_actor_core,
        extract_q_values=lambda preds: preds.q_values,
        ),
      LossFn=usfa.UsfaLossFn(
        discount=config.discount,

        importance_sampling_exponent=config.importance_sampling_exponent,
        burn_in_length=config.burn_in_length,
        max_replay_size=config.max_replay_size,
        max_priority_weight=config.max_priority_weight,
        bootstrap_n=config.bootstrap_n,
      ))
    # NOTE: main differences below
    network_factory = functools.partial(
            usfa.make_object_oriented_minigrid_networks, config=config)
    env_kwargs['object_options'] = True  # has no mechanism to select from object options since dependent on what agent sees
  else:
    raise NotImplementedError(agent)

  # -----------------------
  # load environment factory
  # -----------------------

  environment_factory = functools.partial(
    make_environment_fn,
    **env_kwargs)

  # -----------------------
  # setup observer factory for environment
  # -----------------------
  observers = [
    # this logs the average every reset=50 episodes (instead of every episode)
    utils.LevelAvgReturnObserver(
      reset=50 if not debug else 5,
      get_task_name=env_get_task_name,
      ),
    key_room.ObjectCountObserver(
      reset=1000 if not debug else 5,
      prefix=f'Images',
      agent_name=agent,
      get_task_name=env_get_task_name),
  ]

  return experiment_builder.OnlineExperimentConfigInputs(
    agent=agent,
    agent_config=config,
    final_env_kwargs=env_kwargs,
    builder=builder,
    network_factory=network_factory,
    environment_factory=environment_factory,
    observers=observers,
  )

def train_single(
    env_kwargs: dict = None,
    wandb_init_kwargs: dict = None,
    agent_config_kwargs: dict = None,
    log_dir: str = None,
    num_actors: int = 1,
    run_distributed: bool = False,
):

  debug = FLAGS.debug

  experiment_config_inputs = setup_experiment_inputs(
    make_environment_fn=make_keyroom_env,
    env_get_task_name= lambda env: env.unwrapped.task.goal_name(),
    agent_config_kwargs=agent_config_kwargs,
    env_kwargs=env_kwargs,
    debug=debug)

  env_setting = experiment_config_inputs.final_env_kwargs['setting']
  test_setting = TestOptions(env_setting).name
  logger_factory_kwargs = dict(
    actor_label=f"actor-{test_setting}",
    evaluator_label=f"evaluator-{test_setting}",
    learner_label=f"learner",
  )

  experiment = experiment_builder.build_online_experiment_config(
    experiment_config_inputs=experiment_config_inputs,
    log_dir=log_dir,
    wandb_init_kwargs=wandb_init_kwargs,
    logger_factory_kwargs=logger_factory_kwargs,
    debug=debug
  )
  if run_distributed:
    program = experiments.make_distributed_experiment(
        experiment=experiment,
        num_actors=num_actors)

    local_resources = {
        "actor": PythonProcess(env={"CUDA_VISIBLE_DEVICES": ""}),
        "evaluator": PythonProcess(env={"CUDA_VISIBLE_DEVICES": ""}),
        "counter": PythonProcess(env={"CUDA_VISIBLE_DEVICES": ""}),
        "replay": PythonProcess(env={"CUDA_VISIBLE_DEVICES": ""}),
        "coordinator": PythonProcess(env={"CUDA_VISIBLE_DEVICES": ""}),
    }
    controller = lp.launch(program,
              lp.LaunchType.LOCAL_MULTI_PROCESSING,
              terminal='current_terminal',
              local_resources=local_resources)
    controller.wait(return_on_first_completed=True)
    controller._kill()

  else:
    experiments.run_experiment(experiment=experiment)

def setup_wandb_init_kwargs():
  if not FLAGS.use_wandb:
    return dict()

  wandb_init_kwargs = dict(
      project=FLAGS.wandb_project,
      entity=FLAGS.wandb_entity,
      notes=FLAGS.wandb_notes,
      name=FLAGS.wandb_name,
      group=FLAGS.search or 'default',
      save_code=False,
  )
  return wandb_init_kwargs

def run_single():
  ########################
  # default settings
  ########################
  env_kwargs = dict()
  agent_config_kwargs = dict()
  num_actors = FLAGS.num_actors
  run_distributed = FLAGS.run_distributed
  wandb_init_kwargs = setup_wandb_init_kwargs()
  if FLAGS.debug:
    agent_config_kwargs.update(dict(
      samples_per_insert=1,
      min_replay_size=100,
      batch_size=3,
      trace_length=7,
    ))
    env_kwargs.update(dict(
    ))

  folder = FLAGS.folder or os.environ.get('RL_RESULTS_DIR', None)
  if not folder:
    folder = '/tmp/rl_results'

  if FLAGS.make_path:
    # i.e. ${folder}/runs/${date_time}/
    folder = parallel.gen_log_dir(
        base_dir=os.path.join(folder, 'rl_results'),
        hourminute=True,
        date=True,
    )

  ########################
  # override with config settings, e.g. from parallel run
  ########################
  if FLAGS.config_file:
    configs = utils.load_config(FLAGS.config_file)
    config = configs[FLAGS.config_idx-1]  # starts at 1 with SLURM
    logging.info(f'loaded config: {str(config)}')

    agent_config_kwargs.update(config['agent_config'])
    env_kwargs.update(config['env_config'])
    folder = config['folder']

    num_actors = config['num_actors']
    run_distributed = config['run_distributed']

    wandb_init_kwargs['group'] = config['wandb_group']
    wandb_init_kwargs['name'] = config['wandb_name']
    wandb_init_kwargs['project'] = config['wandb_project']
    wandb_init_kwargs['entity'] = config['wandb_entity']

    if not config['use_wandb']:
      wandb_init_kwargs = dict()


  if FLAGS.debug and not FLAGS.subprocess:
      configs = parallel.get_all_configurations(spaces=sweep(FLAGS.search))
      first_agent_config, first_env_config = parallel.get_agent_env_configs(
          config=configs[0])
      agent_config_kwargs.update(first_agent_config)
      env_kwargs.update(first_env_config)

  if not run_distributed:
    agent_config_kwargs['samples_per_insert'] = 1

  train_single(
    wandb_init_kwargs=wandb_init_kwargs,
    env_kwargs=env_kwargs,
    agent_config_kwargs=agent_config_kwargs,
    log_dir=folder,
    num_actors=num_actors,
    run_distributed=run_distributed
    )

def run_many():
  wandb_init_kwargs = setup_wandb_init_kwargs()

  folder = FLAGS.folder or os.environ.get('RL_RESULTS_DIR', None)
  if not folder:
    folder = '/tmp/rl_results_dir'

  assert FLAGS.debug is False, 'only run debug if not running many things in parallel'

  if FLAGS.parallel == 'ray':
    parallel.run_ray(
      wandb_init_kwargs=wandb_init_kwargs,
      use_wandb=FLAGS.use_wandb,
      debug=FLAGS.debug,
      folder=folder,
      space=sweep(FLAGS.search),
      make_program_command=functools.partial(
        parallel.make_program_command,
        trainer_filename=__file__,
        run_distributed=FLAGS.run_distributed,
        num_actors=FLAGS.num_actors),
    )
  elif FLAGS.parallel == 'sbatch':
    parallel.run_sbatch(
      trainer_filename=__file__,
      wandb_init_kwargs=wandb_init_kwargs,
      use_wandb=FLAGS.use_wandb,
      folder=folder,
      run_distributed=FLAGS.run_distributed,
      search_name=FLAGS.search,
      debug=FLAGS.debug_parallel,
      spaces=sweep(FLAGS.search),
      num_actors=FLAGS.num_actors)

def sweep(search: str = 'default'):
  if search == 'flat':
    space = [
        {
            "num_steps": tune.grid_search([40e6]),
            "agent": tune.grid_search(['flat_muzero', 'flat_q', 'flat_usfa']),
            "seed": tune.grid_search([4]),
            "group": tune.grid_search(['baselines-8']),
            "env.setting": tune.grid_search([0,1,2]),
        },
    ]
  elif search == 'ambiguous_flat':
    space = [
        # {
        #     "num_steps": tune.grid_search([30e6]),
        #     "agent": tune.grid_search(['flat_muzero', 'flat_q', 'flat_usfa']),
        #     "seed": tune.grid_search([5]),
        #     "group": tune.grid_search(['ambiguous-flat-1']),
        #     "env.setting": tune.grid_search([0]),
        #     "env.task_features_cls": tune.grid_search(['ambiguous_flat']),
        # },

        {
            "num_steps": tune.grid_search([30e6]),
            "agent": tune.grid_search(['flat_q']),
            "seed": tune.grid_search([5]),
            "group": tune.grid_search(['ambiguous-flat-1']),
            "env.setting": tune.grid_search([0]),
            "env.task_features_cls": tune.grid_search(['ambiguous_flat']),
        },
    ]
  elif search == 'flat_q':
    space = [
        {
            "num_steps": tune.grid_search([30e6]),
            "agent": tune.grid_search(['flat_q']),
            "seed": tune.grid_search([5]),
            "group": tune.grid_search(['flat_q-3']),
            "env.setting": tune.grid_search([2]),
            "env.task_features_cls": tune.grid_search(['flat', 'ambiguous_flat']),
        },
    ]
  elif search == 'flat_usfa':
    space = [
        # {
        #     "num_steps": tune.grid_search([20e6]),
        #     "agent": tune.grid_search(['flat_usfa']),
        #     "seed": tune.grid_search([5]),
        #     "group": tune.grid_search(['flat_usfa-4-reg']),
        #     "env.setting": tune.grid_search([2]),
        #     "env.task_features_cls": tune.grid_search(['flat']),
        # },
        {
            "num_steps": tune.grid_search([20e6]),
            "agent": tune.grid_search(['flat_usfa']),
            "seed": tune.grid_search([5]),
            "group": tune.grid_search(['flat_usfa-6']),
            "env.setting": tune.grid_search([2]),
            "env.task_features_cls": tune.grid_search(['flat']),
            "q_coeff": tune.grid_search([0.0]),
            "sf_coeff": tune.grid_search([1.0]),
            "sf_loss": tune.grid_search(['qlambda', 'qlearning']),
            # "env.steps_per_room": tune.grid_search([25, 50]),
            # "learning_rate": tune.grid_search([1e-1, 1e-2, 5e-3,  1e-3]),
            # "combine_policy": tune.grid_search(['product', 'sum']),
        },
    ]
  elif search == 'flat_muzero':
    space = [
        {
            "num_steps": tune.grid_search([30e6]),
            "agent": tune.grid_search(['flat_muzero']),
            "seed": tune.grid_search([5]),
            "max_sim_depth": tune.grid_search([1]),
            "group": tune.grid_search(['flat_muzero-new-1']),
            "env.setting": tune.grid_search([2]),
            "env.task_features_cls": tune.grid_search(['flat', 'ambiguous_flat']),
        },
    ]
  elif search == 'object_q':
    space = [
        {
            "num_steps": tune.grid_search([30e6]),
            "agent": tune.grid_search(['object_q']),
            "seed": tune.grid_search([5]),
            "group": tune.grid_search(['object_q-3']),
            "trace_length": tune.grid_search([10, 20]),
            "env.setting": tune.grid_search([2]),
            "env.task_features_cls": tune.grid_search(['flat', 'ambiguous_flat']),
        },
    ]

  else:
    raise NotImplementedError(search)

  return space

def main(_):
  assert FLAGS.parallel in ('ray', 'sbatch', 'none')
  if FLAGS.parallel in ('ray', 'sbatch'):
    run_many()
  else:
    run_single()

if __name__ == '__main__':
  app.run(main)
