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

"""Pretrained image models for use during training."""

from typing import Optional, Sequence, Tuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import tensorflow as tf
from videogvt.train_lib import resnet_v1

_DEFAULT_RESNET_PATH = None

RESNET_IMG_SIZE = 224
VALID_MODELS = ["resnet50"]


@flax.struct.dataclass
class ModelState:
  params: flax.core.FrozenDict
  batch_stats: flax.core.FrozenDict


class ObjectFromDict(object):

  def __init__(self, d):
    for a, b in d.items():
      if isinstance(b, (list, tuple)):
        setattr(self, a,
                [ObjectFromDict(x) if isinstance(x, dict) else x for x in b])
      else:
        setattr(self, a, ObjectFromDict(b) if isinstance(b, dict) else b)


def create_train_state(config: ml_collections.ConfigDict, rng: np.ndarray,
                       input_shape: Sequence[int],
                       num_classes: int) -> Tuple[nn.Module, ModelState]:
  """Create and initialize the model.

  Args:
    config: Configuration for model.
    rng: JAX PRNG Key.
    input_shape: Shape of the inputs fed into the model.
    num_classes: Number of classes in the output layer.

  Returns:
    model: Flax nn.Nodule model architecture.
    state: The initialized ModelState with the optimizer.
  """
  if config.model_name == "resnet50":
    model_cls = resnet_v1.ResNet50
  else:
    raise ValueError(f"Model {config.model_name} not supported.")
  model = model_cls(num_classes=num_classes)
  variables = model.init(rng, jnp.ones(input_shape), train=False)
  params = variables["params"]
  batch_stats = variables["batch_stats"]
  return model, ModelState(params=params, batch_stats=batch_stats)


def get_pretrained_model(
    model_name: str = "resnet50",
    checkpoint_path: Optional[str] = _DEFAULT_RESNET_PATH
) -> Tuple[nn.Module, ModelState]:
  """Returns a pretrained model loaded from weights in checkpoint_dir.

  Args:
    model_name: Name of model architecture to load. Currently only supports
      "resnet50".
    checkpoint_path: Path of .npy containing pretrained state.

  Returns:
    model: Flax nn.Nodule model architecture.
    state: The initialized ModelState with the optimizer.
  """
  if model_name not in VALID_MODELS:
    raise ValueError(f"Model {model_name} not supported.")
  # Initialize model.
  config = ml_collections.ConfigDict()
  config.model_name = "resnet50"
  config.sgd_momentum = 0.9  # Unused for inference.
  config.seed = 42  # Unused for inference.
  model_rng = jax.random.PRNGKey(config.seed)
  model, state = create_train_state(
      config,
      model_rng,
      input_shape=(1, RESNET_IMG_SIZE, RESNET_IMG_SIZE, 3),
      num_classes=1000)

  if checkpoint_path is not None:
    # Set up checkpointing of the model and the input pipeline.
    with tf.io.gfile.GFile(checkpoint_path, "rb") as f:
      checkpoint_data = np.load(f, allow_pickle=True).item()
      state = ModelState(
          params=checkpoint_data["params"],
          batch_stats=checkpoint_data["batch_stats"])
  return model, state


def get_pretrained_embs(state: ModelState, model: nn.Module,
                        images: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Extract embeddings from a pretrained model.

  Args:
    state: ModelState containing model parameters.
    model: Pretrained Flax model.
    images: Array of shape (H, W, 3).

  Returns:
    pool: Pooled outputs from intermediate layer of shape (H', W', C).
    outputs: Outputs from last layer with shape (num_classes,).
  """

  if len(images.shape) != 4 or images.shape[3] != 3:
    raise ValueError("images should be of shape (H, W, 3).")
  if images.shape[1] != RESNET_IMG_SIZE and images.shape[2] != RESNET_IMG_SIZE:
    images = jax.image.resize(
        images,
        (images.shape[0], RESNET_IMG_SIZE, RESNET_IMG_SIZE, images.shape[3]),
        "bilinear")
  variables = {"params": state.params, "batch_stats": state.batch_stats}
  pool, outputs = model.apply(variables, images, mutable=False, train=False)
  return pool, outputs
