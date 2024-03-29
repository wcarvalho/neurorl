# Copyright 2018 DeepMind Technologies Limited. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Default logger."""

import logging
from typing import Any, Callable, Mapping, Optional

from acme.utils.loggers import aggregators
from acme.utils.loggers import asynchronous as async_logger
from acme.utils.loggers import base
from acme.utils.loggers import csv
from acme.utils.loggers import filters
from acme.utils.loggers import terminal

import jax
import numpy as np
import time

from pathlib import Path
import library.utils as data_utils

try:
  import wandb
  WANDB_AVAILABLE=True
except ImportError:
  WANDB_AVAILABLE=False


def copy_numpy(values):
  return jax.tree_map(np.array, values)


def make_logger(
    log_dir: str,
    label: str,
    save_data: bool = False,
    time_delta: float = 1.0,
    asynchronous: bool = False,
    use_tensorboard: bool = False,
    use_wandb: bool = True,
    print_fn: Optional[Callable[[str], None]] = None,
    serialize_fn: Optional[Callable[[Mapping[str, Any]], str]] = copy_numpy,
    steps_key: str = 'steps',
    log_with_key: Optional[str] = None,
) -> base.Logger:
  """Makes a default Acme logger.

    Loggers/Filters used:
      - TerminalLogger: log to terminal
      - CSVLogger (off by default): save data as csv
      - WandbLogger: save data to wandb
      - Dispatcher: aggregates loggers (all before act independently)
      - NoneFilter: removes NaN data
      - AsyncLogger
      - HasKeyFilter: only write data for specified key
      - TimeFilter: how often to write data
  
  Args:
    label: Name to give to the logger.
    save_data: Whether to persist data.
    time_delta: Time (in seconds) between logging events.
    asynchronous: Whether the write function should block or not.
    print_fn: How to print to terminal (defaults to print).
    serialize_fn: An optional function to apply to the write inputs before
      passing them to the various loggers.
    steps_key: Ignored.
    log_with_key: only log things with this key.

  Returns:
    A logger object that responds to logger.write(some_dict).
  """
  # del steps_key
  if not print_fn:
    print_fn = logging.info
  terminal_logger = terminal.TerminalLogger(label=label, print_fn=print_fn)

  loggers = [terminal_logger]

  if save_data:
    loggers.append(csv.CSVLogger(log_dir, label=label))

  if use_tensorboard:
    raise NotImplementedError

  if use_wandb and WANDB_AVAILABLE:
    loggers.append(WandbLogger(
      label=label,
      steps_key=steps_key,
      ))

  # Dispatch to all writers and filter Nones and by time.
  logger = aggregators.Dispatcher(loggers, serialize_fn)
  logger = filters.NoneFilter(logger)
  if asynchronous:
    logger = async_logger.AsyncLogger(logger)

  if log_with_key is not None:
    logger = HasKeyFilter(logger, key=log_with_key)
  if time_delta > 0:
    logger = filters.TimeFilter(logger, time_delta)

  return logger


def _format_key(key: str) -> str:
  """Internal function for formatting keys in Tensorboard format."""
  new = key.title().replace("_", "").replace("/", "-")
  return new

def default_logger_name_fn(logger_label, value_key):
  if 'grad' in value_key.lower():
    # e.g. [MeanGrad/FarmSharedOutput/~/FeatureAttention/Conv2D1] --> [Loss/MeanGrad-FarmSharedOutput-~-FeatureAttention-Conv2D1]
    name = f'z.grads_{logger_label}/{_format_key(value_key)}'
  else:
    name = f'{logger_label}/{_format_key(value_key)}'
  return name

class WandbLogger(base.Logger):
  """Logs to a tf.summary created in a given logdir.
  If multiple TFSummaryLogger are created with the same logdir, results will be
  categorized by labels.
  """

  def __init__(
      self,
      label: str = 'Logs',
      steps_key: Optional[str] = None,
      name_fn = None,
      **kwargs,
  ):
    """Initializes the logger.
    Args:
      logdir: directory to which we should log files.
      label: label string to use when logging. Default to 'Logs'.
      steps_key: key to use for steps. Must be in the values passed to write.
    """
    self._time = time.time()
    self.label = label
    self._iter = 0
    self._steps_key = steps_key
    if name_fn is None:
      name_fn = default_logger_name_fn
    self._name_fn = name_fn

  def write(self, values: base.LoggingData):
    if self._steps_key is not None and self._steps_key not in values:
      logging.warning('steps key "%s" not found. Skip logging.', self._steps_key)
      logging.warning('Available keys:', str(values.keys()))
      return

    step = values[self._steps_key] if self._steps_key is not None else self._iter

    to_log={}
    for key in values.keys() - [self._steps_key]:
      value = values[key]
      if isinstance(value, dict):
        new_dict = data_utils.flatten_dict(
          value, parent_key=key, sep="/")
        for k2, v2 in new_dict.items():
          # bit of a hack
          name = self.label + "/" +  k2
          to_log[name] = v2
      else:
        name = self._name_fn(self.label, key)
        to_log[name] = value

    to_log[f'{self.label}/step']  = step

    wandb.log(to_log)

    self._iter += 1

  def close(self):
    try:
      wandb.finish()
    except Exception as e:
      pass

class HasKeyFilter(base.Logger):
  """Logger which writes to another logger at a given time interval."""

  def __init__(self, to: base.Logger, key: str):
    """Initializes the logger.
    Args:
      to: A `Logger` object to which the current object will forward its results
        when `write` is called.
      key: which key to to write
    """
    self._to = to
    self._key = key
    assert key is not None

  def write(self, values: base.LoggingData):
    hasdata = values.pop(self._key, None)
    if hasdata:
      self._to.write(values)

  def close(self):
    self._to.close()