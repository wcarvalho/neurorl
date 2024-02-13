
from typing import Callable, Dict, NamedTuple, Tuple, Optional, Any

import dataclasses
import dm_env
import functools
import chex
import jax
import jax.numpy as jnp
import haiku as hk
import matplotlib.pyplot as plt
import rlax
import numpy as np
import wandb

import distrax
from acme import specs
from acme.jax import types as jax_types, utils as jax_utils
from acme.agents.jax.r2d2 import actor as r2d2_actor
from acme.jax import networks as networks_lib
from acme.agents.jax import actor_core as actor_core_lib
from acme.wrappers import observation_action_reward

from library.utils import episode_mean
from library.utils import make_episode_mask
from library.utils import rolling_window
from library.utils import scale_gradient
from library import muzero_mlps
import library.networks as networks

from td_agents import basics
from td_agents.basics import ActorObserver, ActorState

from projects.human_sf import usfa_offtask

q_learning_lambda = usfa_offtask.q_learning_lambda
get_actor_core = usfa_offtask.get_actor_core
SFLossFn = usfa_offtask.SFLossFn
Observer = usfa_offtask.Observer
MonolithicSfHead = usfa_offtask.MonolithicSfHead
IndependentSfHead = usfa_offtask.IndependentSfHead
cumulants_from_env = usfa_offtask.cumulants_from_env
Observation = jax.Array
NestedArray = Any
State = NestedArray
InitFn = Callable[[networks_lib.PRNGKey], networks_lib.Params]
RecurrentStateFn = Callable[[networks_lib.Params], State]
ValueFn = Callable[[networks_lib.Params, Observation, State],
                         networks_lib.Value]
ModelFn = Callable[[State], NestedArray]

LARGE_NEGATIVE = -1e7


@dataclasses.dataclass
class Config(basics.Config):

  eval_task_support: str = "train"  # options:
  nsamples: int = 0  # no samples outside of train vector
  variance: float = 0.1

  final_conv_dim: int = 16
  conv_flat_dim: Optional[int] = 0
  sf_layers : Tuple[int]=(128, 128)
  policy_layers : Tuple[int]=(128, 128)
  feature_layers: Tuple[int]=(512, 512)
  transition_blocks: int = 6
  combine_policy: str = 'sum'

  head: str = 'independent'
  sf_activation: str = 'relu'
  sf_mlp_type: str = 'hk'
  out_init_value: Optional[float] = 0.0

  # learning
  importance_sampling_exponent: float = 0.0
  samples_per_insert: float = 10.0
  trace_length: int = 20
  weighted_coeff: float = 1.0
  unweighted_coeff: float = 1.0
  cat_coeff: float = 1.0

  # Online loss
  task_coeff: float = 1.0
  loss_fn : str = 'qlearning'
  lambda_: float  = .9

  # Dyna loss
  dyna_coeff: float = 0.1
  n_actions_dyna: int = 1
  n_tasks_dyna: int = 5

  # Model loss
  simulation_steps: int = 5
  feature_coeff: float = 1.0
  model_sf_coeff: float = 1.0
  model_coeff: float = 1.0
  scale_grad: float = .5
  binary_feature_loss: bool = False
  mask_zero_features: float = 0.0


###################################
# DATA CLASSES
###################################
@dataclasses.dataclass
class MbrlSfNetworks:
  """Network that can unroll state-fn and apply model or SFs over an input sequence."""
  init: InitFn
  apply: RecurrentStateFn
  unroll: RecurrentStateFn
  init_recurrent_state: RecurrentStateFn
  apply_model: ModelFn
  compute_sfs: ModelFn


class Predictions(NamedTuple):
  state: jax.Array
  q_values: jnp.ndarray  # q-value
  sf: jnp.ndarray # successor features
  policy: jnp.ndarray  # policy vector
  task: jnp.ndarray  # task vector (potentially embedded)
  action_mask: Optional[jax.Array] = None


@chex.dataclass(frozen=True)
class ModelOuputs:
  state: jax.Array
  state_features: Optional[jax.Array] = None
  state_feature_logits: Optional[jax.Array] = None
  action_mask: Optional[jax.Array] = None
  action_mask_logits: Optional[jax.Array] = None


###################################
# Actor Observer
###################################
class Observer(ActorObserver):
  def __init__(self,
               period=100,
               plot_success_only: bool = False,
               colors=None,
               prefix: str = 'SFsObserver'):
    super(Observer, self).__init__()
    self.period = period
    self.prefix = prefix
    self.successes = 0
    self.idx = -1
    self.logging = True
    self.plot_success_only = plot_success_only
    self._colors = colors or plt.rcParams['axes.prop_cycle']

  def wandb_log(self, d: dict):
    if wandb.run is not None:
      wandb.log(d)
    else:
      pass

  def observe_first(self, state: ActorState, timestep: dm_env.TimeStep) -> None:
    """Observes the initial state and initial time-step.

    Usually state will be all zeros and time-step will be output of reset."""
    self.idx += 1

    # epsiode just ended, flush metrics if you want
    if self.idx > 0:
      self.flush_metrics()

    # start collecting metrics again
    self.actor_states = [state]
    self.timesteps = [timestep]
    self.actions = []

  def observe_action(self, state: ActorState, action: jax.Array) -> None:
    """Observe state and action that are due to observation of time-step.

    Should be state after previous time-step along"""
    self.actor_states.append(state)
    self.actions.append(action)

  def observe_timestep(self, timestep: dm_env.TimeStep) -> None:
    """Observe next.

    Should be time-step after selecting action"""
    self.timesteps.append(timestep)

  def flush_metrics(self) -> Dict[str, float]:
    """Returns metrics collected for the current episode."""
    rewards = jnp.stack([t.reward for t in self.timesteps])[1:]
    total_reward = rewards.sum()
    if self.plot_success_only:
      if not total_reward > 0:
        return

    self.successes += 1

    count = self.successes if self.plot_success_only else self.idx

    if not count % self.period == 0:
      return

    tasks = [t.observation.observation['task'] for t in self.timesteps]
    tasks = np.stack(tasks)
    # task = tasks[0]

    # NOTES:
    # Actor states: T, starting at 0
    # timesteps: T, starting at 0
    # action: T-1, starting at 0
    ##################################
    # successor features
    ##################################
    # first prediction is empty (None)
    # [T, A, C]
    predictions = [s.predictions for s in self.actor_states[1:]]
    sfs = jnp.stack([p.sf for p in predictions])

    npreds = len(sfs)
    actions = jnp.stack(self.actions)[:npreds]

    index = jax.vmap(rlax.batched_index, in_axes=(2, None), out_axes=1)

    sfs = index(sfs, actions)  # [T-1, C]

    q_values = (sfs*tasks[:-1]).sum(-1)

    # ignore 0th (reset) time-step w/ 0 reward and last (terminal) time-step
    state_features = jnp.stack([t.observation.observation['state_features'] for t in self.timesteps])[1:]

    # Determine the number of plots needed based on the condition
    ndims = state_features.shape[1]
    active_dims = [j for j in range(ndims) if state_features[:, j].sum() > 0]
    if active_dims:
      n_plots = len(active_dims) + 1  # +1 for the rewards subplot

      # Calculate rows and columns for subplots
      cols = min(n_plots, 4)  # Maximum of 3 horizontally
      rows = np.ceil(n_plots / cols).astype(int)

      # Create a figure with dynamic subplots
      width = 3*cols
      height = 3*rows
      fig, axs = plt.subplots(rows, cols, figsize=(width, height), squeeze=False)
      default_cycler = iter(self._colors)

      # Plot rewards in the first subplot
      axs[0, 0].plot(rewards, label='rewards', linestyle='--', color='grey')
      axs[0, 0].plot(q_values, label='q_values', color='grey')
      axs[0, 0].set_title("Reward Predictions")
      axs[0, 0].legend()

      # Initialize subplot index for state_features and sfs
      subplot_idx = 1  # Start from the second subplot
      for j in active_dims:
          # Calculate subplot position
          row, col = divmod(subplot_idx, cols)
          ax = axs[row, col]

          try:
              color = next(default_cycler)['color']
          except StopIteration:
              raise RuntimeError(f"unexpectedly had {len(active_dims)} dimensions active")

          # Plot state_features and sfs for each active dimension
          ax.plot(state_features[:, j], label=f'$\\phi - {j}$', linestyle='--', color=color)
          ax.plot(sfs[:, j], label=f'$\\psi - {j}$', color=color)
          ax.set_title(f"Dimension {j}")
          ax.legend()

          subplot_idx += 1  # Move to the next subplot

      # Hide any unused subplots if there are any
      for k in range(subplot_idx, rows * cols):
          row, col = divmod(k, cols)
          axs[row, col].axis('off')

      plt.tight_layout()
      self.wandb_log({f"{self.prefix}/sf-group-prediction": wandb.Image(fig)})
      plt.close(fig)
    # else:
    #   if rewards.sum() > 0:
    #     raise RuntimeError("there is reward but no cumulants active. this must be a bug")

    # Close the plot
    # ##################################
    # # get images
    # ##################################
    # # [T, H, W, C]
    # frames = np.stack([t.observation.observation['image'] for t in self.timesteps])
    # # self.wandb_log({
    # #   f'{self.prefix}/episode-{task}': [wandb.Image(frame) for frame in frames]})
    # wandb.log({f'{self.prefix}/episode-{task}':
    #            wandb.Video(np.transpose(frames, (0, 3, 1, 2)), fps=.01)})


###################################
# Helper functions
###################################

def zero_entries_mask(x, key, mask_zero_probs=.25, eps=1e-5):
  """
  Create masks from x.
  
  Parameters:
  - x: A [T, N] shaped array of state features.
  - key: A random key for JAX's random number generator.
  
  Returns:
  - on_mask: A mask of 1s where x > 0.
  - off_mask: A mask of 1s with a 25% chance wherever x == 0.
  """
  # First mask: 1s where x > 0
  on_mask = (x > 0).astype(x.dtype)

  # Generate random probabilities for the second mask
  mask_zero_probs = (jnp.abs(x) < eps).astype(x.dtype)*mask_zero_probs
  off_mask = jax.random.bernoulli(key, p=mask_zero_probs).astype(x.dtype)

  return on_mask+off_mask

def sample_K_times(x, K, key, replace=True):
  indices = jax.random.choice(
    key, len(x), shape=(K,), replace=replace)  # Sampling K indices with replacement
  samples = x[indices]  # Selecting the actions
  return samples

def split_key(key, N):
  keys = jax.random.split(key, num=N+1)
  key_set = keys[1:]
  key = keys[0]
  return key, key_set

def mask_predictions(
    predictions: Predictions,
    action_mask: jax.Array):

  # mask out entries to large negative value
  mask = lambda x: jnp.where(action_mask, x, LARGE_NEGATIVE)
  q_values = mask(predictions.q_values)  # [A]

  # mask out entries to 0, vmap over cumulant dimension
  mask = lambda x: jnp.where(action_mask, x, 0.0)
  mask = jax.vmap(mask, 1, 1)
  sf = mask(predictions.sf)   # [A, Cumulants]

  return predictions._replace(
    q_values=q_values,
    sf=sf,
    action_mask=action_mask)

batch_mask_predictions = jax.vmap(mask_predictions)

###################################
# Loss functions
###################################
def one_step_sf_td(
    sf: jax.Array,
    action: jax.Array,
    next_state_features: jax.Array,
    next_sf: jax.Array,
    discounts: jax.Array,
    q_values: jax.Array,
    ):
  """This loss applies double SF-learning to a set of sfs."""
  td_error_fn = jax.vmap(
    rlax.double_q_learning,
    in_axes=(1, None, 0, None, 1, None))

  return td_error_fn(
      sf,                   # [A, C]
      action,               # []
      next_state_features,  # [C]
      discounts,            # []
      next_sf,              # [A, C]
      q_values,             # [A]
      )

@dataclasses.dataclass
class SFTDError:
  bootstrap_n: int = 5
  loss_fn : str = 'qlearning'

  def __call__(
    self,
    online_sf: jax.Array,
    online_actions: jax.Array,
    cumulants: jax.Array,
    discounts: jax.Array,
    target_sf: jax.Array,
    lambda_: jax.Array,
    selector_actions: jax.Array,
    ):

    if self.loss_fn == 'qlearning':
      error_fn = functools.partial(
        rlax.transformed_n_step_q_learning,
        n=self.bootstrap_n)
      # vmap over cumulant dimension (C), return in dim=3
      error_fn = jax.vmap(
        error_fn,
        in_axes=(2, None, 2, None, 1, None), out_axes=1)
      return error_fn(
        online_sf,  # [T, A, C]
        online_actions,  # [T]         
        target_sf,  # [T, A, C]
        selector_actions,  # [T]      
        cumulants,  # [T, C]      
        discounts)  # [T]         

    elif self.loss_fn == 'qlambda':
      error_fn = jax.vmap(q_learning_lambda,
        in_axes=(2, None, 1, None, 2, None, None), out_axes=1)

      return error_fn(
        online_sf,         # [T, A, C] (vmap 2)
        online_actions,    # [T]       (vmap None)
        cumulants,         # [T, C]    (vmap 1)
        discounts,         # [T]       (vmap None)
        target_sf,         # [T, A, C] (vmap 2)
        selector_actions,  # [T]      
        lambda_,           # [T]       (vmap None)
      )
    else:
      raise NotImplementedError


@dataclasses.dataclass
class TaskWeightedSFLossFn:

  weight_type: str = 'reg'
  combination: str = 'loss'

  sum_cumulants: bool = True


  def __call__(
    self,
    td_error: jax.Array,
    task_w: jax.Array,
    ):


    ####################
    # what will be the loss weights?
    ####################
    # [C]
    td_weights = task_w
    if self.weight_type == 'mag':
      td_weights = jnp.abs(td_weights)
    elif self.weight_type == 'reg':
      pass
    elif self.weight_type == 'indicator':
      td_weights = (jnp.abs(td_weights) < 1e-5)
    else:
      raise NotImplementedError

    td_weights = td_weights.astype(task_w.dtype)  # float

    ####################
    # how will we combine them into the loss so we're optimizing towards task
    ####################
    loss_fn = lambda x: (0.5 * jnp.square(x))
    # [C] --> []
    task_weighted_td_error = (td_error*td_weights).sum(-1)
    if self.combination == 'td':
      # in this setting, tds were weighted and apply loss there
      # [T]
      task_weighted_loss = loss_fn(task_weighted_td_error)
    elif self.combination == 'loss':
      # in this setting, weight the output of the loss
      # [T, C]
      td_loss = loss_fn(td_error)
      # [T]
      task_weighted_loss = (td_loss*td_weights).sum(-1)
    else:
      raise NotImplementedError

    ####################
    # computing regular loss
    ####################
    # [T]
    if self.sum_cumulants:
      unweighted_td_error = td_error.sum(axis=-1)
    else:
      unweighted_td_error = td_error.mean(axis=-1)
    unweighted_loss = loss_fn(unweighted_td_error)

    return task_weighted_loss, unweighted_loss

@dataclasses.dataclass
class MtrlDynaUsfaLossFn(basics.RecurrentLossFn):

  extract_cumulants: Callable = cumulants_from_env
  extract_tasks: Callable = lambda d: d.observation.observation['train_tasks']

  # Shared
  bootstrap_n: int = 5
  weighted_coeff: float = 1.0
  unweighted_coeff: float = 1.0
  action_mask: bool = False

  # Online loss
  task_coeff: float = 1.0
  loss_fn : str = 'qlearning'
  lambda_: float  = .9

  # Dyna loss
  dyna_coeff: float = 1.0
  n_actions_dyna: int = 1
  n_tasks_dyna: int = 5

  # Model loss
  simulation_steps: int = 5
  feature_coeff: float = 1.0
  model_sf_coeff: float = 1.0
  model_coeff: float = 1.0
  action_coeff: float = 1.0
  cat_coeff: float = 1e-2
  binary_feature_loss: bool = False
  mask_zero_features: float = 0.0

  def error(self,
            data,
            online_preds,
            online_state,
            target_preds,
            target_state,
            networks,
            params,
            target_params,
            key_grad, **kwargs):

    T, B = data.discount.shape[:2]

    metrics = {}
    total_batch_td_error = jnp.zeros((T, B))
    total_batch_loss = jnp.zeros(B)

    # mask of 1s for all time-steps except for terminal
    episode_mask = make_episode_mask(data, include_final=False)
    # ==========================
    # Dyna SF-learning loss
    # =========================
    if self.dyna_coeff > 0:
      multitask_dyna_loss = functools.partial(
        self.multitask_dyna_loss,
        networks=networks,
        params=params,
        target_params=target_params,
        )
      train_tasks = self.extract_tasks(data)

      multitask_dyna_loss = jax.vmap(jax.vmap(multitask_dyna_loss))
      T, B = data.discount.shape[:2]
      key_grad, loss_keys = split_key(key_grad, N=T*B)
      loss_keys = loss_keys.reshape(T, B, 2)
      discounts = data.discount
      # [T, B], [B], [B]

      action_mask = episode_mask
      if self.action_mask:
        action_mask = data.observation.observation.get('action_mask', None)

      dyna_td_error, dyna_batch_loss, dyna_metrics = multitask_dyna_loss(
            discounts,
            online_preds,
            target_preds,
            train_tasks,
            action_mask,
            loss_keys)

      dyna_batch_loss = episode_mean(dyna_batch_loss, episode_mask)

      # update
      total_batch_td_error += dyna_td_error*self.dyna_coeff
      total_batch_loss += dyna_batch_loss*self.dyna_coeff
      metrics.update({f'2.dyna.{k}': v for k, v in dyna_metrics.items()})

    #==========================
    # Online SF-learning loss
    # =========================
    if self.task_coeff > 0:
      assert T > 1, 'need at least 2 state for this'
      ontask_batch_td_error, ontask_batch_loss, ontask_metrics = self.ontask_loss(
          data=data,
          online_preds=online_preds,
          target_preds=target_preds,
          episode_mask=episode_mask)

      # [T-1, B] --> [T, B]
      ontask_batch_td_error = jnp.concatenate(
        (ontask_batch_td_error, jnp.zeros((1, B))))

      total_batch_td_error += ontask_batch_td_error*self.task_coeff
      total_batch_loss += ontask_batch_loss*self.task_coeff
      metrics.update({f'1.online.{k}': v for k, v in ontask_metrics.items()})

    # ==========================
    # Model loss
    # =========================
    if self.model_coeff > 0:
      model_loss_fn = functools.partial(
        self.model_loss,
        networks=networks,
        params=params,
        rng_key=key_grad,
      )
      # vmap over batch dimension
      model_loss_fn = jax.vmap(model_loss_fn, 
        in_axes=1, out_axes=0)

      model_batch_loss, model_metrics = model_loss_fn(
        data,
        online_preds,
        target_preds,
        data.action,
        episode_mask,
        )

      total_batch_loss += model_batch_loss*self.model_coeff
      metrics.update({f'3.model.{k}': v for k, v in model_metrics.items()})

    metrics['0.total_loss'] = total_batch_loss
    # remove last time-step since invalid for RL loss
    total_batch_td_error = total_batch_td_error[:-1]
    return total_batch_td_error, total_batch_loss, metrics

  def ontask_loss(
    self,
    data: jax.Array,
    online_preds: jax.Array,
    target_preds: jax.Array,
    episode_mask: jax.Array,
    ):
    #---------------
    # SFs
    #---------------
    # all are [T+1, B, A, C]
    # A = actions, C = cumulant dim
    online_sf = online_preds.sf  # [T+1, B, A, C]
    target_sf = target_preds.sf  # [T+1, B, A, C]

    # [T, B]
    online_sf = online_sf*episode_mask[:,:, None, None]
    target_sf = target_sf*episode_mask[:,:, None, None]

    #---------------
    # cumulants
    #---------------
    # pseudo rewards, [T/T+1, B, C]
    cumulants = self.extract_cumulants(
      data=data)
    cumulants = cumulants.astype(online_sf.dtype)

    #---------------
    # discounts + lambda
    #---------------
    discounts = (data.discount * self.discount).astype(online_sf.dtype) # [T, B]
    lambda_ = (data.discount * self.lambda_).astype(online_sf.dtype) 

    #---------------
    # loss
    #---------------
    # Get selector actions from online Q-values for double Q-learning.
    selector_actions = jnp.argmax(online_preds.q_values, axis=-1) # [T+1]

    td_fn = SFTDError(loss_fn=self.loss_fn)
    batch_axis = 1
    # [T, B, C]
    td_errors = jax.vmap(td_fn, batch_axis, batch_axis)(
      online_sf[:-1],      # [T, B, A, C]
      data.action[:-1],    # [T, B]
      cumulants,
      discounts[:-1],
      target_sf[1:],
      lambda_[:-1],
      selector_actions[1:]
    )

    # vmap over time + batch dimensions
    loss_fn = jax.vmap(jax.vmap(TaskWeightedSFLossFn()))

    task_weighted_loss, unweighted_loss = loss_fn(
      td_errors, online_preds.task[:-1])

    batch_loss = (task_weighted_loss * self.weighted_coeff + 
                  unweighted_loss * self.unweighted_coeff)

    td_errors = td_errors.mean(axis=2)

    metrics = {
      '0.total_loss': batch_loss,
      '1.unweighted_loss': unweighted_loss,
      '1.task_weighted_loss': task_weighted_loss,
      '2.td_error': jnp.abs(td_errors),
      '3.cumulants': cumulants,
      '3.sf_mean': online_sf,
      '3.sf_var': online_sf.var(axis=-1),
      }

    return td_errors, batch_loss, metrics

  def multitask_dyna_loss(
    self,
    discounts: jax.Array,
    online_preds: NestedArray,
    target_preds: NestedArray,
    tasks: jax.Array,
    action_mask: jax.Array,
    rng_key: networks_lib.PRNGKey,
    networks: MbrlSfNetworks,
    params: networks_lib.Params,
    target_params: networks_lib.Params,
    ):
    """
    Offtask Dyna with Successor features.

    Dyna as normal except:
    (a) use successor features instead of Q-values
    (b) sample imaginary tasks in addition to imaginary actions

    Logic:
      1. sample M tasks
      2. compute successor features for each task
      3. apply dyna loss to each task_sf
         for each task_sf, sample K actions
         perform dyna back-up for each action
    
    Details:
      - compute root successor features with params
      - compute state-features with params
      - compute target successor features with target params
        - this requires unrolling model with params for 

    Args:
        discounts (jax.Array): [].
        online_preds (jax.Array): [...]
        target_preds (jax.Array): [...]
        tasks (jax.Array): [N, D]
        episode_mask (jax.Array): []
        rng_key (networks_lib.PRNGKey): key.
        networks (MbrlSfNetworks): networks.
        params (networks_lib.Params): params
        target_params (networks_lib.Params): target_params.

    Returns:
        _type_: _description_
    """

    num_actions = online_preds.q_values.shape[-1]

    ##############
    # prepare function for computing SFs
    ##############
    def compute_sfs(*args, **kwargs):
      return networks.compute_sfs(params, *args, **kwargs)

    def compute_target_sfs(*args, **kwargs):
      return networks.compute_sfs(target_params, *args, **kwargs)

    ##############
    # prepare function for applying model
    ##############
    def apply_model(*args, **kwargs):
      return networks.apply_model(params, *args, **kwargs)
    apply_model = jax.vmap(
      apply_model, in_axes=(0, None, 0), out_axes=0)

    def apply_target_model(*args, **kwargs):
      return networks.apply_model(target_params, *args, **kwargs)
    apply_target_model = jax.vmap(
      apply_target_model, in_axes=(0, None, 0), out_axes=0)

    def sf_dyna_td(
      online_state: jax.Array,
      target_state: jax.Array,
      sf: jax.Array,
      q_values: jax.Array,
      task: jax.Array,
      discounts_: jax.Array,
      key: networks_lib.PRNGKey,
      ):
      """For a single task, do K simulations with dyna and compute td-error.

      Args:
          online_state (jax.Array): [D]
          target_state (jax.Array): [D]
          sf (jax.Array): [A, C]
          task (jax.Array): [C]
          discounts_ (jax.Array): []
          key (networks_lib.PRNGKey): key.

      Returns:
          _type_: _description_
      """
      optimal_q_action = jnp.argmax(q_values, axis=-1)

      # sample K random actions. append the optimal action by current Q-values
      key, sample_key = jax.random.split(key)
      nactions = min(self.n_actions_dyna, num_actions)
      sampled_actions = jax.random.choice(
        sample_key, num_actions, shape=(nactions,), replace=False)

      # [K+1]
      sampled_actions = jnp.concatenate(
        (sampled_actions, optimal_q_action[None]))
      nactions = len(sampled_actions)

      #-----------------------
      # apply model K times for params and target_params
      #-----------------------
      key, model_keys = split_key(key, N=nactions)
      # [K, ...]
      model_outputs, _ = apply_model(
          model_keys, online_state, sampled_actions)

      key, model_keys = split_key(key, N=nactions)
      # [K, ...]
      target_model_outputs, _ = apply_target_model(
          model_keys, target_state, sampled_actions)


      #-----------------------
      # successor features from model at t+1
      # compute for each of the K actions but a SINGLE task
      #-----------------------
      # [K, A, C]
      # using regular params, this will be used for Q-value action-selection
      key, sf_keys = split_key(key, N=nactions)
      next_predictions = jax.vmap(
        compute_sfs,
        in_axes=(0, 0, None), out_axes=0)(
          # [K, 2], [K, D], [C]
          sf_keys, model_outputs.state, task)

      # using target params, this will define the targets
      key, sf_keys = split_key(key, N=nactions)
      target_next_predictions = jax.vmap(
        compute_target_sfs,
        in_axes=(0, 0, None), out_axes=0)(
          # [K, 2], [K, D], [C]
          sf_keys, target_model_outputs.state, task)
      if self.action_mask:
        next_predictions = batch_mask_predictions(
          next_predictions, model_outputs.action_mask)
        target_next_predictions = batch_mask_predictions(
          target_next_predictions, model_outputs.action_mask)

      #-----------------------
      # TD error
      #-----------------------
      # VMAP over K sampled actions
      td_error_fn = jax.vmap(
        one_step_sf_td,
        in_axes=(None, 0, 0, 0, None, 0))

      next_state_features = model_outputs.state_features
      next_q_values = next_predictions.q_values
      target_next_sf = target_next_predictions.sf
      return td_error_fn(
        sf,                   # [A, C]
        sampled_actions,      # [K]
        jax.lax.stop_gradient(next_state_features),  # [K, C]
        jax.lax.stop_gradient(target_next_sf),             # [K, A, C]
        discounts_,                                # []
        jax.lax.stop_gradient(next_q_values),             # [A]
        )

    #---------------
    # sample K tasks and compute K SFs
    #---------------
    # [M, C]
    rng_key, sample_key = jax.random.split(rng_key)
    max_tasks = tasks.shape[0]
    ntasks = min(max_tasks, self.n_tasks_dyna)
    sampled_tasks = jax.random.choice(
      sample_key, tasks, shape=(ntasks,), replace=False)

    # [M, A, C]
    rng_key, sf_keys = split_key(rng_key, N=ntasks)
    predictions = jax.vmap(
      compute_sfs,
      in_axes=(0, None, 0), out_axes=0)(
        # [M, 2], [D], [M, C]
        sf_keys, online_preds.state, sampled_tasks)

    if self.action_mask:
      # vmap over K tasks for predictions, repeat action mask
      predictions = jax.vmap(
        mask_predictions, in_axes=(0, None))(
          predictions, action_mask)

    #---------------
    # get sf_dyna td-errors for each of the K task SFs
    #---------------
    loss_fn = jax.vmap(sf_dyna_td,
      in_axes=(None, None, 0, 0, 0, None, 0))

    rng_key, task_keys = split_key(rng_key, N=ntasks)
    td_errors = loss_fn(
      online_preds.state,  # [D]
      target_preds.state,  # [D]
      predictions.sf,            # [M, A, C]
      predictions.q_values,      # [M, A, C]
      sampled_tasks,             # [M, C]
      discounts,                 # []
      task_keys,                 # [M]
      )

    #---------------
    # get task-weighted SF loss for each SF TD-error
    #---------------
    # vmap over task
    loss_fn = jax.vmap(TaskWeightedSFLossFn())

    # output should be [M, K] for tasks and actions sampled
    task_weighted_loss, unweighted_loss = loss_fn(
      td_errors,
      sampled_tasks)

    batch_loss = (task_weighted_loss * self.weighted_coeff + 
                  unweighted_loss * self.unweighted_coeff)

    td_errors = td_errors.mean()
    batch_loss = batch_loss.mean()

    metrics = {
      '0.total_loss': batch_loss,
      '1.unweighted_loss': unweighted_loss,
      '1.task_weighted_loss': task_weighted_loss,
      '2.td_error': jnp.abs(td_errors),
      }

    # [T], []
    return td_errors, batch_loss, metrics

  def model_loss(
    self,
    data: NestedArray,
    online_preds: jax.Array,
    target_preds: jax.Array,
    actions: jax.Array,
    episode_mask: jax.Array,
    rng_key: networks_lib.PRNGKey,
    networks: MbrlSfNetworks,
    params: networks_lib.Params,
    ):

    # TODO: move this outside of this loss function
    discounts = data.discount*self.discount

    assert self.simulation_steps < len(online_preds.state)
    def shorten(struct):
      return jax.tree_map(lambda x:x[:-self.simulation_steps], struct)

    # [T', D]
    start_states = shorten(online_preds.state)

    # for every timestep t=0,...T,  we have predictions for t+1, ..., t+k where k = simulation_steps
    # use rolling window to create T x k prediction targets
    # this will shorten the data from T to T'=T-k+1
    roll = functools.partial(rolling_window, size=self.simulation_steps)

    #----------------------
    # MODEL ACTIONS [T', ...]
    #----------------------
    model_actions = roll(actions[:-1])

    #----------------------
    # STATE FEATURES, [T', C]
    #----------------------
    vmap_roll = jax.vmap(roll, 1, 2)
    state_features = self.extract_cumulants(data)
    state_features_target = vmap_roll(state_features)

    # mask over just time [T]
    state_features_mask_T = episode_mask[:-1]

    # mask over individual entries [T, C]
    state_features_mask = state_features_mask_T[:,None]*jnp.ones_like(state_features)
    if self.mask_zero_features > 0.0:
      # if masking zeros out, mask out some proportion of when state features are 0.
      # helps with data imbalance problem
      rng_key, sample_key = jax.random.split(rng_key)
      new_mask = zero_entries_mask(
        x=state_features,
        key=sample_key,
        # if mask_zero_features=0.0, keep 100%
        # if mask_zero_features=1.0, keep 0%.
        mask_zero_probs=1.0-self.mask_zero_features)
      state_features_mask = state_features_mask*new_mask

    # [T', k, C]
    state_features_mask_rolled = vmap_roll(state_features_mask)

    #----------------------
    # SUCCESSOR FEATURE TARGETS
    #----------------------
    # [T, A, C] --> [T, C] for chosen ones
    selector_actions = jnp.argmax(online_preds.q_values, axis=-1)
    # next_step_actions = roll(selector_actions[1:])  # [T', k, A]

    target_net_sfs = target_preds.sf
    index = jax.vmap(rlax.batched_index, in_axes=(2, None), out_axes=1)
    target_net_sfs = index(target_net_sfs, selector_actions)

    # compute return
    # [T, C]
    sf_targets = jax.vmap(
      functools.partial(
        rlax.n_step_bootstrapped_returns, n=self.bootstrap_n),
      in_axes=(1, None, 1),
      out_axes=1)(
        # [T, C], [T], [T, C]
        state_features, discounts[:-1], target_net_sfs[1:])

    # index starting at next-timestep since learning model
    num_sf_targets = sf_targets.shape[0] - 1
    zeros = jnp.zeros((1, sf_targets.shape[-1]))
    sf_targets = jnp.concatenate((sf_targets[1:], zeros))

    # [T, C] --> [T', k, C] for unrolls
    sf_targets = vmap_roll(sf_targets)
    sf_targets = jax.lax.stop_gradient(sf_targets)

    # [T] --> [T', k] for unrolls
    sf_mask = jnp.concatenate(
      (episode_mask[1:num_sf_targets], jnp.zeros(2)))
    sf_mask_rolled = roll(sf_mask)


    # ------------
    # unrolls the model from each time-step in parallel
    # ------------
    def model_unroll(key, state, actions):
      def fn(carry: jax.Array, a: jax.Array):
        k, s = carry
        k, model_key = jax.random.split(k)
        model_output, new_state = networks.apply_model(
            params, model_key, s, a)
        return (k, new_state), model_output
      carry = (key, state)
      _, model_outputs = jax.lax.scan(fn, carry, actions)

      return model_outputs

    model_unroll = jax.vmap(model_unroll)

    rng_key, model_keys = split_key(rng_key, N=len(start_states))
    # [T, K, ...]
    model_outputs = model_unroll(
        model_keys, start_states, model_actions,
    )

    # ------------
    # get action mask
    # ------------
    action_mask_target = state_features_target
    action_mask_logits = model_outputs.state_feature_logits
    action_loss_mask = state_features_mask
    action_loss_mask_rolled = state_features_mask_rolled
    if self.action_mask:
      action_mask = data.observation.observation.get('action_mask', None)
      assert action_mask is not None, 'need action_mask for prediction'

      action_mask_target = vmap_roll(action_mask[1:])
      action_mask_logits = model_outputs.action_mask_logits
      assert action_mask_logits is not None, 'need action_mask_logits'
      action_loss_mask = make_episode_mask(data, include_final=True)
      action_loss_mask_rolled = roll(action_loss_mask[1:])


    # ------------
    # compute SFs for future time-steps
    # ------------
    def compute_sfs(*args, **kwargs):
      return networks.compute_sfs(params, *args, **kwargs)
    compute_sfs = jax.vmap(compute_sfs)  # over time dimension
    compute_sfs = jax.vmap(compute_sfs)  # over simulation dimension

    # PREP TASK
    # [T, C] --> [T', k, C]
    task = online_preds.task[1:]
    task = vmap_roll(task)

    # PREP PRNGkeys
    T = task.shape[0]
    rng_key, sf_keys = split_key(
      rng_key, N=T * self.simulation_steps)
    sf_keys = sf_keys.reshape(T, self.simulation_steps, 2)

    # COMPUTE PREDICTIONS
    # [T', k, ...]
    predictions = compute_sfs(sf_keys, model_outputs.state, task)

    if self.action_mask:
      # these predictions begin at t+1 since for output of applying model
      # action_mask targets also begin at t+1
      # this means you can use the action mask targets to mask the predictions
      predictions = jax.vmap(batch_mask_predictions)(
        predictions, action_mask_target)

    # ------------
    # now that computed, select out the SF for the action taken in the env.
    # this will be our prediction
    # ------------
    predicted_sfs = predictions.sf         # [T', k, A, C]
    next_step_actions = roll(actions[1:])  # [T', k, A]
    index = jax.vmap(jax.vmap(rlax.batched_index))  # vmap T', k
    index = jax.vmap(index, in_axes=(3, None), out_axes=2)

    # [T', k, C]
    predicted_sfs = index(predicted_sfs, next_step_actions)

    def compute_losses(
      predicted_features_logits,
      features_target_,
      features_mask_,
      task_weights_,
      #
      predicted_sf_,
      sf_target_,
      sf_mask_,
      #
      action_mask_logits_,
      action_mask_,
      action_loss_mask_,
      ):
      def binary_cross_entropy(logits, target):
        return -distrax.Bernoulli(logits).log_prob(target)
      binary_cross_entropy = jax.vmap(binary_cross_entropy)

      if self.binary_feature_loss:
        features_loss = binary_cross_entropy(
          predicted_features_logits, features_target_)
        # features_loss = features_loss.sum(-1)  # [k]
      else:
        features_loss = rlax.l2_loss(
          predicted_features_logits, features_target_)

      # first mask out over cumulant dimension
      features_loss = jax.vmap(episode_mean)(features_loss, features_mask_)

      # then mask out over time-step simulation dimension
      # mask is whenever any of the individual cumulant masks were on.
      features_mask_T = (features_mask_.sum(-1) > 0).astype(features_mask_.dtype)
      features_loss = episode_mean(features_loss, features_mask_T)  # []

       # [k, C]
      sf_td = sf_target_ - predicted_sf_
      sf_loss_fn = jax.vmap(TaskWeightedSFLossFn())

      # [k]
      task_weighted_loss, unweighted_loss = sf_loss_fn(
        sf_td, task_weights_)

      sf_loss = (task_weighted_loss * self.weighted_coeff + 
                    unweighted_loss * self.unweighted_coeff)
      sf_loss = sf_loss.mean(-1)  # []
      # sf_l2 = rlax.l2_loss(predicted_sf_, sf_target_)  # [k, C]
      # sf_l2 = sf_l2.sum(-1)  # [k]
      # sf_loss = episode_mean(sf_l2, sf_mask_)  # []

      if self.action_mask:
        mask_log_prob = binary_cross_entropy(
          action_mask_logits_, action_mask_)
        mask_log_prob = mask_log_prob.mean(-1)
        action_mask_loss = episode_mean(mask_log_prob, action_loss_mask_)
      else:
        action_mask_loss = jnp.zeros_like(sf_loss)

      return features_loss, sf_loss, action_mask_loss

    feature_loss, sf_loss, action_mask_loss = jax.vmap(compute_losses)(
      model_outputs.state_feature_logits,
      state_features_target,
      state_features_mask_rolled,
      task,
      predicted_sfs,
      sf_targets,
      sf_mask_rolled,
      action_mask_logits,
      action_mask_target,
      action_loss_mask_rolled,
      )

    def apply_mask(loss, mask):
      return episode_mean(loss, mask[:len(loss)])

    feature_loss = apply_mask(feature_loss, state_features_mask_T)
    sf_loss = apply_mask(sf_loss, sf_mask)
    action_mask_loss = apply_mask(action_mask_loss, action_loss_mask)
    action_mask_loss = action_mask_loss*float(self.action_mask)

    metrics = {
      "0.feature_loss": feature_loss,
      "0.sf_loss": sf_loss,
      "1.action_mask_loss": action_mask_loss,
    }

    total_loss = (
      feature_loss * self.feature_coeff * self.cat_coeff + 
      sf_loss * self.model_sf_coeff + 
      action_mask_loss * self.action_coeff * self.cat_coeff
      )

    return total_loss, metrics

###################################
# Architectures
###################################
class SfGpiHead(hk.Module):

  """Universal Successor Feature Approximator GPI head"""
  def __init__(self,
    num_actions: int,
    sf_net : hk.Module,
    ):
    """Summary

    Args:
        num_actions (int): Description
        hidden_size (int, optional): hidden size of SF MLP network
        variance (float, optional): variances of sampling
        nsamples (int, optional): number of policies
        eval_task_support (bool, optional): include eval task in support
    
    Raises:
        NotImplementedError: Description
    """

    super(SfGpiHead, self).__init__()
    self.num_actions = num_actions
    self.sf_net = sf_net

  def __call__(self,
    state: jax.Array,  # memory output (e.g. LSTM)
    task: jax.Array,  # task vector
    ) -> Predictions:
    """Summary

    Args:
        state (jax.Array): D, typically rnn_output
        policies (jax.Array): N x D
        task (jax.Array): D

    Returns:
        Predictions: Description

    """
    sfs, q_values = self.sf_net(state, task, task)
    num_actions = q_values.shape[-1]
    # [N, D] --> [N, A, D]
    policies_repeated = jnp.repeat(task[:, None],
                                   repeats=num_actions, axis=1)

    return Predictions(
      state=state,
      sf=sfs,       # [N, A, D_w]
      policy=policies_repeated,         # [N, A, D_w]
      q_values=q_values,  # [N, A]
      task=task)         # [D_w]


  def sfgpi(self,
    state: jax.Array,  # memory output (e.g. LSTM)
    task: jax.Array,  # task vector
    policies: Optional[jax.Array] = None,  # task vector
    ) -> Predictions:
    """Summary

    Args:
        state (jax.Array): D, typically rnn_output
        policies (jax.Array): N x D
        task (jax.Array): D

    Returns:
        Predictions: Description

    """
    if policies is None:
      policies = jnp.expand_dims(task, axis=-2)
      import ipdb; ipdb.set_trace()

    sfs, q_values = jax.vmap(
      self.sf_net, in_axes=(None, 0, None), out_axes=0)(
        state,
        policies,
        task)

    # GPI
    # -----------------------
    # [N, A] --> [A]
    q_values = jnp.max(q_values, axis=-2)
    num_actions = q_values.shape[-1]

    # [N, D] --> [N, A, D]
    policies_repeated = jnp.repeat(policies[:, None], repeats=num_actions, axis=1)

    return Predictions(
      state=state,
      sf=sfs,       # [N, A, D_w]
      policy=policies_repeated,         # [N, A, D_w]
      q_values=q_values,  # [N, A]
      task=task)         # [D_w]


class UsfaArch(hk.RNNCore):
  """Universal Successor Feature Approximator."""

  def __init__(self,
               torso: networks.OarTorso,
               memory: hk.RNNCore,
               transition_fn: hk.RNNCore,
               sf_head: SfGpiHead,
               name: str = 'usfa_arch'):
    super().__init__(name=name)
    self._torso = torso
    self._memory = memory
    self._sf_head = sf_head
    self._transition_fn = transition_fn

  def initial_state(self, batch_size: Optional[int],
                    **unused_kwargs) -> State:
    return self._memory.initial_state(batch_size)

  def __call__(
      self,
      inputs: observation_action_reward.OAR,  # [D]
      state: State,  # [D]
      evaluation: bool = False,
  ) -> Tuple[Predictions, State]:
    torso_outputs = self._torso(inputs)  # [D+A+1]
    memory_input = jnp.concatenate(
      (torso_outputs.image, torso_outputs.action), axis=-1)

    core_outputs, new_state = self._memory(memory_input, state)

    # inputs = jax.tree_map(lambda x:x.astype(jnp.float32), inputs)
    if evaluation:
      predictions = self._sf_head.sfgpi(
        state=core_outputs,
        task=inputs.observation['task'],
        policies=inputs.observation['train_tasks']
        )
    else:
      predictions = self._sf_head(
        state=core_outputs,
        task=inputs.observation['task'],
      )
    return predictions, new_state

  def unroll(
      self,
      inputs: observation_action_reward.OAR,  # [T, B, ...]
      state: State  # [T, ...]
  ) -> Tuple[State, State]:

    torso_outputs = hk.BatchApply(self._torso)(inputs)  # [T, B, D+A+1]
    memory_input = jnp.concatenate(
      (torso_outputs.image, torso_outputs.action), axis=-1)

    core_outputs, new_states = hk.static_unroll(
      self._memory, memory_input, state)

    # treat T,B like this don't exist with vmap
    predictions = jax.vmap(jax.vmap(self._sf_head))(
        core_outputs,                # [T, B, D]
        inputs.observation['task'],  # [T, B]
      )
    return predictions, new_states

  def apply_model(
      self,
      state: State,  # [B, D]
      action: jax.Array,  # [B]
  ) -> Tuple[ModelOuputs, State]:
    # [B, D], [B, D]
    model_predictions, new_state = self._transition_fn(action, state)
    return model_predictions, new_state

  def compute_sfs(
      self,
      state: State,  # [B, D]
      task: jax.Array,  # [B, D]
  ) -> Predictions:
    # [B, D], [B, D]
    sf_predictions = self._sf_head(state=state, task=task)
    return sf_predictions


def make_mbrl_usfa_network(
        environment_spec: specs.EnvironmentSpec,
        make_core_module: Callable[[], hk.RNNCore]) -> MbrlSfNetworks:
  """Builds a MbrlSfNetworks from a hk.Module factory."""

  dummy_observation = jax_utils.zeros_like(environment_spec.observations)
  dummy_action = jnp.array(0)

  def make_unrollable_network_functions():
    network = make_core_module()

    def init() -> Tuple[networks_lib.NetworkOutput, State]:
      out, _ = network(dummy_observation, network.initial_state(None))
      return network.apply_model(out.state, dummy_action)

    apply = network.__call__
    return init, (apply,
                  network.unroll,
                  network.initial_state,
                  network.apply_model,
                  network.compute_sfs)

  # Transform and unpack pure functions
  f = hk.multi_transform(make_unrollable_network_functions)
  apply, unroll, initial_state_fn, apply_model, compute_sfs = f.apply

  def init_recurrent_state(key: jax_types.PRNGKey,
                           batch_size: Optional[int] =  None) -> State:
    no_params = None
    return initial_state_fn(no_params, key, batch_size)

  return MbrlSfNetworks(
      init=f.init,
      apply=apply,
      unroll=unroll,
      apply_model=apply_model,
      compute_sfs=compute_sfs,
      init_recurrent_state=init_recurrent_state)


def make_minigrid_networks(
        env_spec: specs.EnvironmentSpec,
        config: Config) -> networks_lib.UnrollableNetwork:
  """Builds default USFA networks for Minigrid games."""

  num_actions = env_spec.actions.num_values
  state_features_dim = env_spec.observations.observation['state_features'].shape[0]

  def make_core_module() -> UsfaArch:

    ###########################
    # Setup transition function: ResNet
    ###########################
    def transition_fn(action: int, state: State):
      action_onehot = jax.nn.one_hot(
          action, num_classes=num_actions)
      assert action_onehot.ndim in (1, 2), "should be [A] or [B, A]"

      def _transition_fn(action_onehot, state):
        """ResNet transition model that scales gradient.

        Same tricks that MuZero uses."""
        # action: [A]
        # state: [D]
        new_state = muzero_mlps.Transition(
            channels=config.state_dim,
            num_blocks=config.transition_blocks)(
            action_onehot, state)
        new_state = scale_gradient(new_state, config.scale_grad)

        state_feature_logits = muzero_mlps.PredictionMlp(
          config.feature_layers,
          state_features_dim,
          name='state_features')(new_state)
        if config.binary_feature_loss:
          state_features = distrax.Bernoulli(
            logits=state_feature_logits).sample(seed=hk.next_rng_key())
          state_features = state_features.astype(new_state.dtype)
        else:
          state_features = state_feature_logits

        outputs = ModelOuputs(
          state=new_state,
          state_feature_logits=state_feature_logits,
          state_features=state_features)

        return outputs, new_state

      if action_onehot.ndim == 2:
        _transition_fn = jax.vmap(_transition_fn)
      return _transition_fn(action_onehot, state)
    transition_fn = hk.to_module(transition_fn)('transition_fn')


    ###################################
    # SF Head
    ###################################
    if config.head == 'independent':
      SfNetCls = IndependentSfHead
    elif config.head == 'monolithic':
      SfNetCls = MonolithicSfHead
    else:
      raise NotImplementedError

    sf_head = SfGpiHead(
      num_actions=num_actions,
      sf_net=SfNetCls(
        layers=config.sf_layers,
        state_features_dim=state_features_dim,
        num_actions=num_actions,
        policy_layers=config.policy_layers,
        combine_policy=config.combine_policy,
        activation=config.sf_activation,
        mlp_type=config.sf_mlp_type,
        out_init_value=config.out_init_value,
        ))

    return UsfaArch(
      torso=networks.OarTorso(
        num_actions=num_actions,
        vision_torso=networks.BabyAIVisionTorso(
          conv_dim=config.final_conv_dim,
          out_dim=config.conv_flat_dim),
        output_fn=networks.TorsoOutput,
      ),
      memory=hk.LSTM(config.state_dim),
      transition_fn=transition_fn,
      sf_head=sf_head,
      )

  return make_mbrl_usfa_network(
    env_spec, make_core_module)
