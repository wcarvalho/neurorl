
from typing import Callable, NamedTuple, Tuple, Optional

import dataclasses
import functools
import jax
import jax.numpy as jnp
import haiku as hk
import rlax
import numpy as np

from acme import specs
from acme.agents.jax.r2d2 import actor as r2d2_actor
from acme.jax import networks as networks_lib
from acme.agents.jax import actor_core as actor_core_lib
from acme.wrappers import observation_action_reward

from td_agents import basics
from td_agents import r2d2

import networks

@dataclasses.dataclass
class Config(r2d2.Config):
  eval_task_support: str = "train"
  nsamples: int = 0  # no samples outside of train vector

def expand_tile_dim(x, size, axis=-1):
  """E.g. shape=[1,128] --> [1,10,128] if dim=1, size=10
  """
  ndims = len(x.shape)
  _axis = axis
  if axis < 0: # go AFTER -axis dims, e.g. x=[1,128], axis=-2 --> [1,10,128]
    axis += 1
    _axis = axis % ndims # to account for negative

  x = jnp.expand_dims(x, _axis)
  tiling = [1]*_axis + [size] + [1]*(ndims-_axis)
  return jnp.tile(x, tiling)

def make_episode_mask(data, include_final=False, **kwargs):
  """Look at where have valid task data. Everything until 1 before final valid data counts towards task. Data.discount always ends two before final data. 
  e.g. if valid data is [x1, x2, x3, 0, 0], data.discount is [1,0,0,0,0]. So can use that to obtain masks.
  
  Args:
      data (TYPE): Description
      include_final (bool, optional): if True, include all data. if False, include until 1 time-step before final data
  
  Returns:
      TYPE: Description
  """
  T, B = data.discount.shape
  # for data [x1, x2, x3, 0, 0]
  if include_final:
    # return [1,1,1,0,0]
    return jnp.concatenate((jnp.ones((2, B)), data.discount[:-2]), axis=0)
  else:
    # return [1,1,0,0,0]
    return jnp.concatenate((jnp.ones((1, B)), data.discount[:-1]), axis=0)

def episode_mean(x, mask):
  if len(mask.shape) < len(x.shape):
    nx = len(x.shape)
    nd = len(mask.shape)
    extra = nx - nd
    dims = list(range(nd, nd+extra))
    batch_loss = jnp.multiply(x, jnp.expand_dims(mask, dims))
  else:
    batch_loss = jnp.multiply(x, mask)
  return (batch_loss.sum(0))/(mask.sum(0)+1e-5)

def cumulants_from_env(data, online_preds, online_state, target_preds, target_state):
  return data.observation.observation.state_features # [T, B, C]

def cumulants_from_preds(
  data,
  online_preds,
  online_state,
  target_preds,
  target_state,
  stop_grad=True,
  use_target=False):

  if use_target:
    cumulants = target_preds.state_feature
  else:
    cumulants = online_preds.state_feature
  if stop_grad:
    return jax.lax.stop_gradient(cumulants) # [T, B, C]
  else:
    return cumulants # [T, B, C]

def sample_gauss(mean, var, key, nsamples, axis):
  # gaussian (mean=mean, var=.1I)
  # mean = [B, D]
  if nsamples >= 1:
    mean = jnp.expand_dims(mean, -2) # [B, 1, ]
    samples = jnp.tile(mean, [1, nsamples, 1])
    dims = samples.shape # [B, N, D]
    samples =  samples + jnp.sqrt(var) * jax.random.normal(key, dims)
    samples = samples.astype(mean.dtype)
  else:
    samples = jnp.expand_dims(mean, axis=1) # [B, N, D]
  return samples

@dataclasses.dataclass
class UsfaLossFn(basics.RecurrentLossFn):

  extract_cumulants: Callable = cumulants_from_env

  def error(self, data, online_preds, online_state, target_preds, target_state, **kwargs):
    # ======================================================
    # Prepare Data
    # ======================================================
    # all are [T+1, B, N, A, C]
    # N = num policies, A = actions, C = cumulant dim
    online_sf = online_preds.sf
    online_z = online_preds.z
    target_sf = target_preds.sf

    # pseudo rewards, [T/T+1, B, C]
    cumulants = self.extract_cumulants(
      data=data, online_preds=online_preds, online_state=online_state,
      target_preds=target_preds, target_state=target_state)
    cumulants = cumulants.astype(online_sf.dtype)

    # Get selector actions from online Q-values for double Q-learning.
    online_q =  (online_sf*online_z).sum(axis=-1) # [T+1, B, N, A]
    selector_actions = jnp.argmax(online_q, axis=-1) # [T+1, B, N]
    online_actions = data.action # [T, B]

    # Preprocess discounts & rewards.
    discounts = (data.discount * self.discount).astype(online_q.dtype) # [T, B]

    cumulants_T = cumulants.shape[0]
    data_T = online_sf.shape[0]

    if cumulants_T == data_T:
      # shorten cumulants
      cum_idx = data_T - 1
    elif cumulants_T == data_T - 1:
      # no need to shorten cumulants
      cum_idx = cumulants_T
    elif cumulants_T > data_T:
      raise RuntimeError("This should never happen?")
    else:
      raise NotImplementedError


    # ======================================================
    # Loss for SF
    # ======================================================
    def sf_loss(online_sf, online_actions, target_sf, selector_actions, cumulants, discounts):
      """Vmap over cumulant dimension.
      
      Args:
          online_sf (TYPE): [T, A, C]
          online_actions (TYPE): [T]
          target_sf (TYPE): [T, A, C]
          selector_actions (TYPE): [T]
          cumulants (TYPE): [T, C]
          discounts (TYPE): [T]

      Returns:
          TYPE: Description
      """
      # copies selector_actions, online_actions, vmaps over cumulant dim
      td_error_fn = jax.vmap(
        functools.partial(
            rlax.transformed_n_step_q_learning,
            n=self.bootstrap_n),
        in_axes=(2, None, 2, None, 1, None), out_axes=1)

      return td_error_fn(
        online_sf[:-1],       # [T, A, C] (vmap 2) 
        online_actions[:-1],  # [T]       (vmap None) 
        target_sf[1:],        # [T, A, C] (vmap 2) 
        selector_actions[1:], # [T]       (vmap None) 
        cumulants[:cum_idx],       # [T, C]    (vmap 1) 
        discounts[:-1])       # [T]       (vmap None)


    # ======================================================
    # Prepare loss (via vmaps)
    # ======================================================
    # vmap over batch dimension (B)
    sf_loss = jax.vmap(sf_loss, in_axes=1, out_axes=1)
    # vmap over policy dimension (N)
    sf_loss = jax.vmap(sf_loss, in_axes=(2, None, 2, 2, None, None), out_axes=2)
    # output = [0=T, 1=B, 2=N, 3=C]
    batch_td_error = sf_loss(
      online_sf,        # [T, B, N, A, C] (vmap 2,1)
      online_actions,   # [T, B]          (vmap None,1)
      target_sf,        # [T, B, N, A, C] (vmap 2,1)
      selector_actions, # [T, B, N]       (vmap 2,1)
      cumulants,        # [T, B, C]       (vmap None,1)
      discounts)        # [T, B]          (vmap None,1)


    if self.mask_loss:
      # [T, B]
      episode_mask = make_episode_mask(data, include_final=False)
      # average over {T, N, C} --> # [B]
      batch_loss = episode_mean(
        x=(0.5 * jnp.square(batch_td_error)).mean(axis=(2,3)),
        mask=episode_mask[:-1])
    else:
      batch_loss = (0.5 * jnp.square(batch_td_error)).mean(axis=(0,2,3))

    batch_td_error = batch_td_error.mean(axis=(2, 3)) # [T, B]

    metrics = {
      f'z.loss_SfMain_{self.loss}': batch_loss.mean(),
      'z.sf_mean': online_sf.mean(),
      'z.sf_var': online_sf.var(),
      'z.sf_max': online_sf.max(),
      'z.sf_min': online_sf.min()}

    return batch_td_error, batch_loss, metrics # [T, B], [B]

def get_actor_core(
    networks: basics.NetworkFn,
    config: Config,
    evaluation: bool = False,
) -> r2d2_actor.R2D2Policy:
  """Returns ActorCore for R2D2."""

  num_epsilons = config.num_epsilons
  evaluation_epsilon = config.evaluation_epsilon
  if (not num_epsilons and evaluation_epsilon is None) or (num_epsilons and
                                                           evaluation_epsilon):
    raise ValueError(
        'Exactly one of `num_epsilons` or `evaluation_epsilon` must be '
        f'specified. Received num_epsilon={num_epsilons} and '
        f'evaluation_epsilon={evaluation_epsilon}.')

  def select_action(params: networks_lib.Params,
                    observation: networks_lib.Observation,
                    state: r2d2_actor.R2D2ActorState[actor_core_lib.RecurrentState]):
    rng, policy_rng = jax.random.split(state.rng)

    predictions, recurrent_state = networks.apply(
      params, policy_rng, observation, state.recurrent_state, evaluation)
    action = rlax.epsilon_greedy(state.epsilon).sample(policy_rng, predictions.q_values)

    return action, r2d2_actor.R2D2ActorState(
        rng=rng,
        epsilon=state.epsilon,
        recurrent_state=recurrent_state,
        prev_recurrent_state=state.recurrent_state)

  def init(
      rng: networks_lib.PRNGKey
  ) -> r2d2_actor.R2D2ActorState[actor_core_lib.RecurrentState]:
    rng, epsilon_rng, state_rng = jax.random.split(rng, 3)
    if not evaluation:
      epsilon = jax.random.choice(epsilon_rng,
                                  np.logspace(config.epsilon_min, config.epsilon_max, config.num_epsilons, base=config.epsilon_base))
    else:
      epsilon = evaluation_epsilon
    initial_core_state = networks.init_recurrent_state(state_rng, None)

    return r2d2_actor.R2D2ActorState(
        rng=rng,
        epsilon=epsilon,
        recurrent_state=initial_core_state,
        prev_recurrent_state=initial_core_state)

  def get_extras(
      state: r2d2_actor.R2D2ActorState[actor_core_lib.RecurrentState]
      ) -> r2d2_actor.R2D2Extras:
    return {'core_state': state.prev_recurrent_state}

  return actor_core_lib.ActorCore(init=init, select_action=select_action,
                                  get_extras=get_extras)

class USFAPreds(NamedTuple):
  q_value: jnp.ndarray  # q-value
  sf: jnp.ndarray # successor features
  policy: jnp.ndarray  # policy vector
  task: jnp.ndarray  # task vector (potentially embedded)

class USFAInputs(NamedTuple):
  task: jnp.ndarray  # task vector
  usfa_input: jnp.ndarray  # memory output (e.g. LSTM)
  train_tasks: Optional[jnp.ndarray] = None  # train task vectors

class UsfaHead(hk.Module):
  """Page 17."""
  def __init__(self,
    num_actions: int,
    state_features_dim: int,
    hidden_sizes : Tuple[int]=(128, 128),
    nsamples: int=1,
    variance: Optional[float]=0.5,
    eval_task_support: str = 'train', 
    **kwargs,
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
    super(UsfaHead, self).__init__()
    self.num_actions = num_actions
    self.state_features_dim = state_features_dim
    self.var = variance
    self.nsamples = nsamples
    self.eval_task_support = eval_task_support

    self.mlp = hk.nets.MLP(tuple(hidden_sizes)+(num_actions * state_features_dim,))

  def compute_sf_q(self, inputs: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
    """Forward pass
    
    Args:
        inputs (jnp.ndarray): Z
        w (jnp.ndarray): A x C
    
    Returns:
        jnp.ndarray: 2-D tensor of action values of shape [batch_size, num_actions]
    """
    # [A * C]
    sf = self.mlp(inputs)
    # [A, C]
    sf = jnp.reshape(sf, (self.num_actions, self.state_features_dim))

    # dot-product
    q_values = jnp.sum(sf*w, axis=-1) # [B, A]
    return sf, q_values

  def __call__(self,
    usfa_input: jnp.ndarray,  # memory output (e.g. LSTM)
    task: jnp.ndarray,  # task vector
    ) -> USFAPreds:

    # -----------------------
    # policies + embeddings
    # -----------------------
    if self.nsamples > 0:
      # sample N times: [D_w] --> [N+1, D_w]
      z_samples = sample_gauss(
        mean=task, var=self.var, key=hk.next_rng_key(), nsamples=self.nsamples, axis=-2)
      # combine samples with original task vector
      z_base = jnp.expand_dims(task, axis=1) # [1, D_w]
      policy_embeddings = jnp.concatenate((z_base, z_samples), axis=-2)  # [N+1, D_w]
    else:
      policy_embeddings = jnp.expand_dims(task, axis=-2) # [1, D_w]

    return self.sfgpi(
      usfa_input=usfa_input,
      z=policy_embeddings,
      w=task)

  def evaluate(self,
    task: jnp.ndarray,  # task vector
    usfa_input: jnp.ndarray,  # memory output (e.g. LSTM)
    train_tasks: jnp.ndarray,  # all train tasks
    ) -> USFAPreds:

    import ipdb; ipdb.set_trace()
    # -----------------------
    # embed task
    # -----------------------
    w = task # [D]

    # train basis (z)
    w_train = train_tasks # [N, D]
    N = w_train.shape[0]

    # -----------------------
    # policies + embeddings
    # -----------------------
    if len(w_train.shape)==2:
      # z = [N, D]
      # w = [D]
      z = expand_tile_dim(w_train, axis=0, size=B)
    else:
      # [N, D]
      z = w_train


    if self.eval_task_support == 'train':
      pass # z = w_train
      # [N, D]
    elif self.eval_task_support == 'eval':
      # [1, D]
      z = jnp.expand_dims(w, axis=1)

    elif self.eval_task_support == 'train_eval':
      w_expand = jnp.expand_dims(w, axis=1)
      # [B, N+1, D]
      z = jnp.concatenate((z, w_expand), axis=1)
    else:
      raise RuntimeError(self.eval_task_support)

    preds = self.sfgpi(
      usfa_input=usfa_input, z=z, w=w)

    return preds

  def sfgpi(self,
    usfa_input: jnp.ndarray,
    z: jnp.ndarray,
    w: jnp.ndarray) -> USFAPreds:
    """Summary
    
    Args:
        inputs (USFAInputs): Description
        z (jnp.ndarray): N x D
        w (jnp.ndarray): D
    Returns:
        USFAPreds: Description
    """

    n_policies = z.shape[0]

    # [N, D_s]
    sf_input = jnp.concatenate(
      (expand_tile_dim(usfa_input, size=n_policies, axis=-2),z),
      axis=-1)
    # -----------------------
    # prepare task vectors
    # -----------------------
    # [D_z] --> [N, D_z]
    w_expand = expand_tile_dim(w, axis=-2, size=n_policies)

    # [N, D_w] --> [N, A, D_w]
    z = expand_tile_dim(z, axis=-2, size=self.num_actions)
    w_expand = expand_tile_dim(w_expand, axis=-2, size=self.num_actions)

    # -----------------------
    # compute successor features
    # -----------------------
    # inputs = [N, D_s], [N, A, D_w], ouputs = [N, A, D_w], [N, A]
    # repeat once for each policy
    sf, q_values = jax.vmap(self.compute_sf_q)(sf_input, w_expand)

    # -----------------------
    # GPI
    # -----------------------
    # [N, A] --> [A]
    q_values = jnp.max(q_values, axis=-2)

    return USFAPreds(
      sf=sf,       # [N, A, D_w]
      policy=z,         # [N, A, D_w]
      q_value=q_values,  # [N, A]
      task=w)         # [D_w]


class UsfaArch(hk.RNNCore):
  """Universal Successor Feature Approximator."""

  def __init__(self,
               torso: networks.OarTorso,
               memory: hk.RNNCore,
               head: UsfaHead,
               name: str = 'usfa_arch'):
    super().__init__(name=name)
    self._torso = torso
    self._memory = memory
    self._head = head

  def __call__(
      self,
      inputs: observation_action_reward.OAR,  # [D]
      state: hk.LSTMState,  # [D]
      evaluation: bool = False,
  ) -> Tuple[USFAPreds, hk.LSTMState]:
    torso_outputs = self._torso(inputs)  # [D+A+1]
    memory_input = jnp.concatenate(
      (torso_outputs.image, torso_outputs.action), axis=-1)

    core_outputs, new_state = self._memory(memory_input, state)

    if evaluation:
      predictions = self._head.evaluate(
        task=inputs.observation['task'],
        usfa_input=core_outputs,
        train_tasks=inputs.observation['train_tasks']
        )
    else:
      predictions = self._head(
        task=inputs.observation['task'],
        usfa_input=core_outputs,
      )
    return predictions, new_state

  def initial_state(self, batch_size: Optional[int],
                    **unused_kwargs) -> hk.LSTMState:
    return self._memory.initial_state(batch_size)

  def unroll(
      self,
      inputs: observation_action_reward.OAR,  # [T, B, ...]
      state: hk.LSTMState  # [T, ...]
  ) -> Tuple[USFAPreds, hk.LSTMState]:
    """Efficient unroll that applies torso, core, and duelling mlp in one pass."""
    torso_outputs = hk.BatchApply(self._torso)(inputs)  # [T, B, D+A+1]

    memory_input = jnp.concatenate(
      (torso_outputs.image, torso_outputs.action), axis=-1)

    core_outputs, new_states = hk.static_unroll(
      self._memory, memory_input, state)

    predictions = hk.BatchApply(self._head)(
        task=inputs.observation['task'],
        usfa_input=core_outputs,
      )
    return predictions, new_states

def make_minigrid_networks(
        env_spec: specs.EnvironmentSpec,
        config: Config) -> networks_lib.UnrollableNetwork:
  """Builds default USFA networks for Minigrid games."""

  num_actions = env_spec.actions.num_values
  state_features_dim = env_spec.observations.observation['state_features'].shape[0]

  def make_core_module() -> UsfaArch:
    vision_torso = networks.AtariVisionTorso(
      out_dim=config.state_dim)

    observation_fn = networks.OarTorso(
      num_actions=num_actions,
      vision_torso=vision_torso,
      output_fn=networks.TorsoOutput,
    )

    usfa_head = UsfaHead(
      num_actions=num_actions,
      state_features_dim=state_features_dim,
      nsamples=config.nsamples,
      eval_task_support=config.eval_task_support)

    return UsfaArch(
      torso=observation_fn,
      memory=hk.LSTM(config.state_dim),
      head=usfa_head)

  return networks_lib.make_unrollable_network(
    env_spec, make_core_module)


class ObjectOrientedUsfaArch(hk.RNNCore):
  """Universal Successor Feature Approximator."""

  def __init__(self,
               torso: networks.OarTorso,
               memory: hk.RNNCore,
               head: UsfaHead,
               name: str = 'usfa_arch'):
    super().__init__(name=name)
    self._torso = torso
    self._memory = memory
    self._head = head
  def make_object_oriented_usfa_inputs(
      self,
      inputs: observation_action_reward.OAR,
      core_outputs: jnp.ndarray,
      ):
    action_embdder = lambda x: hk.Linear(128, w_init=hk.initializers.RandomNormal)(x)
    object_embdder = lambda x: hk.Linear(128, w_init=hk.initializers.RandomNormal)(x)

    # each are [B, N, D] where N differs for action and object embeddings
    action_embeddings = hk.BatchApply(action_embdder)(inputs.observation['actions'])
    object_embeddings = hk.BatchApply(object_embdder)(inputs.observation['objects'])
    option_inputs = jnp.concatenate((action_embeddings, object_embeddings), axis=-2)

    # vmap concat over middle dimension to replicate concat across all "actions"
    # [B, D1] + [B, A, D2] --> [B, A, D1+D2]
    concat = lambda a, b: jnp.concatenate((a,b), axis=-1)
    concat = jax.vmap(in_axes=(None, 1), out_axes=1)(concat)

    return concat(core_outputs, option_inputs)

  def __call__(
      self,
      inputs: observation_action_reward.OAR,  # [B, ...]
      state: hk.LSTMState,  # [B, ...]
      evaluation: bool = False,
  ) -> Tuple[USFAPreds, hk.LSTMState]:
    torso_outputs = self._torso(inputs)  # [B, D+A+1]
    memory_input = jnp.concatenate(
      (torso_outputs.image, torso_outputs.action), axis=-1)
    core_outputs, new_state = self._memory(memory_input, state)

    import ipdb; ipdb.set_trace()
    head_inputs = USFAInputs(
      task=inputs.observation['task'],
      usfa_input=self.make_object_oriented_usfa_inputs(inputs, core_outputs),
      train_tasks=inputs.observation['train_tasks'],
    )
    if evaluation:
      predictions = self._head.evaluate(head_inputs)
      import ipdb; ipdb.set_trace()
    else:
      predictions = self._head(head_inputs)
      import ipdb; ipdb.set_trace()
    return predictions, new_state

  def initial_state(self, batch_size: Optional[int],
                    **unused_kwargs) -> hk.LSTMState:
    return self._memory.initial_state(batch_size)

  def unroll(
      self,
      inputs: observation_action_reward.OAR,  # [T, B, ...]
      state: hk.LSTMState  # [T, ...]
  ) -> Tuple[USFAPreds, hk.LSTMState]:
    """Efficient unroll that applies torso, core, and duelling mlp in one pass."""
    torso_outputs = hk.BatchApply(self._torso)(inputs)  # [T, B, D+A+1]

    memory_input = jnp.concatenate(
      (torso_outputs.image, torso_outputs.action), axis=-1)

    core_outputs, new_states = hk.static_unroll(
      self._memory, memory_input, state)

    head_inputs = USFAInputs(
      task=torso_outputs.task,
      usfa_input=core_outputs,
    )
    predictions = hk.BatchApply(self._head)(head_inputs)  # [T, B, A]
    return predictions, new_states


def make_object_oriented_minigrid_networks(
        env_spec: specs.EnvironmentSpec,
        config: Config) -> networks_lib.UnrollableNetwork:
  """Builds default USFA networks for Minigrid games."""

  num_actions = env_spec.actions.num_values
  state_features_dim = env_spec.observations.observation['state_features'].shape[0]

  def make_core_module() -> ObjectOrientedUsfaArch:
    vision_torso = networks.AtariVisionTorso(
      out_dim=config.state_dim)

    observation_fn = networks.OarTorso(
      num_actions=num_actions,
      vision_torso=vision_torso,
      output_fn=networks.TorsoOutput,
    )

    usfa_head = UsfaHead(
      num_actions=num_actions,
      state_features_dim=state_features_dim,
      nsamples=config.nsamples,
      eval_task_support=config.eval_task_support)

    return ObjectOrientedUsfaArch(
      torso=observation_fn,
      memory=hk.LSTM(config.state_dim),
      head=usfa_head)

  return networks_lib.make_unrollable_network(
    env_spec, make_core_module)