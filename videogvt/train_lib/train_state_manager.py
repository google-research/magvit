# Copyright 2023 The videogvt Authors.
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

"""Manager of train states."""

from typing import Any, Callable, Dict, Mapping, Optional, Union

import flax
import jax.numpy as jnp
import optax
from scenic.train_lib import train_utils as scenic_train_utils


@flax.struct.dataclass
class VQGANTrainState:
  """Dataclass to keep track of state of training.

  The state of training is structured as a struct.dataclass, which enables
  instances of this class to be passed into jax transformations like tree_map
  and pmap.
  """
  global_step: Optional[int] = 0
  g_tx: Optional[optax.GradientTransformation] = flax.struct.field(
      default=None, pytree_node=False)
  d_tx: Optional[optax.GradientTransformation] = flax.struct.field(
      default=None, pytree_node=False)
  g_opt_state: Optional[optax.OptState] = None
  d_opt_state: Optional[optax.OptState] = None
  g_model_state: Optional[Any] = None
  d_model_state: Optional[Any] = None
  rng: Optional[jnp.ndarray] = None
  metadata: Optional[Dict[str, Any]] = None
  g_params: Optional[Any] = None
  d_params: Optional[Any] = None
  ema_params: Optional[Any] = None

  def __getitem__(self, item):
    """Make TrainState a subscriptable object."""
    return getattr(self, item)

  def get(self, keyname: str, default: Optional[Any] = None) -> Any:
    """Return the value for key if it exists otherwise the default."""
    try:
      return self[keyname]
    except KeyError:
      return default


@flax.struct.dataclass
class VQGANTrainStateDeprecated:
  """Dataclass to keep track of state of training.

  The state of training is structured as a flax.struct.dataclass, which enables
  instances of this class to be passed into jax transformations like tree_map
  and pmap.
  """
  global_step: Optional[int] = 0
  g_optimizer: Optional[Any] = None
  d_optimizer: Optional[Any] = None
  generator_state: Optional[Any] = None
  discriminator_state: Optional[Any] = None
  ema_params: Optional[Any] = None
  rng: Optional[jnp.ndarray] = None
  metadata: Optional[Dict[str, Any]] = None

  def __getitem__(self, item):
    """Makes TrainState a subscriptable object."""
    return getattr(self, item)

  def get(self, keyname: str, default: Optional[Any] = None) -> Any:
    """Return the value for key if it exists otherwise the default."""
    try:
      return self[keyname]
    except KeyError:
      return default




ScenicTrainState = scenic_train_utils.TrainState

TrainState = Union[VQGANTrainState, ScenicTrainState]
AllTrainState = Union[
    VQGANTrainState,
    ScenicTrainState,
    VQGANTrainStateDeprecated,
]


def check_train_state_type(restored_train_state: Mapping[str, Any]):
  """Returns the type of the TrainState."""
  train_state = restored_train_state
  if 'g_optimizer' in train_state and 'd_optimizer' in train_state:
    return VQGANTrainStateDeprecated
  elif 'g_opt_state' in train_state and 'd_opt_state' in train_state:
    return VQGANTrainState
  elif 'opt_state' in train_state and 'model_state' in train_state:
    return ScenicTrainState
  else:
    return 'unknown'
