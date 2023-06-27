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

"""VQGAN Trainer."""
import concurrent.futures
import copy
import functools
from typing import Any, Dict, Sequence, Tuple, Union

from absl import logging
from clu import metric_writers
from clu import metrics
from clu import parameter_overview
from clu import periodic_actions
import flax
from flax import jax_utils
import flax.linen as nn
from flax.training import checkpoints
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
from scenic.common_lib import debug_utils
from scenic.common_lib import video_utils
from scenic.dataset_lib import dataset_utils
from scenic.train_lib import lr_schedules
from scenic.train_lib import train_utils as scenic_train_utils
from tensorflow.io import gfile
from videogvt.train_lib import eval_utils
from videogvt.train_lib import losses
from videogvt.train_lib import train_utils
from videogvt.train_lib import vqgan_train_lib


Batch = train_utils.Batch
PyTree = train_utils.PyTree
TrainState = vqgan_train_lib.TrainState
TrainMetrics = vqgan_train_lib.TrainMetrics
EvalFeatureDict = eval_utils.EvalFeatureDict


def get_optimizer(config: ml_collections.ConfigDict):
  """Constructs the VQGAN optimizer from the given HParams."""

  assert (
      'lr_configs' in config and 'learning_rate_schedule' in config.lr_configs
  ), 'Only support learning_rate_schedule'

  lr_config = config['lr_configs']
  g_lr_config = copy.deepcopy(lr_config).unlock()
  g_lr_config.base_learning_rate = config.optimizer.g_lr
  g_lr_config.lock()

  d_lr_config = copy.deepcopy(lr_config).unlock()
  d_lr_config.base_learning_rate = config.optimizer.d_lr
  d_lr_config.lock()

  lr_meta_fn = lr_schedules.lr_fn_dict[g_lr_config['learning_rate_schedule']]
  g_lr_fn = lr_meta_fn(g_lr_config)
  d_lr_fn = lr_meta_fn(d_lr_config)

  tx_dict = vqgan_train_lib.get_optimizer(g_lr_fn, d_lr_fn, config=config)
  return tx_dict, g_lr_fn, d_lr_fn


def initialize_model(*, model_dict: Dict[str, nn.Module],
                     input_spec: Sequence[Union[Tuple[Tuple[int, ...],
                                                      jnp.dtype],
                                                Tuple[int, ...], None]],
                     config: ml_collections.ConfigDict, rng: jnp.ndarray,
                     is_train: bool):
  """Initializes parameters and model state.

  Args:
    model_dict: Dictionary of the models.
    input_spec: An iterable of (shape, dtype) pairs specifying the shape and
      dtype of the inputs. If unspecified the dtype is float32.
    config: Configurations of the initialization.
    rng: Jax rng keys.
    is_train: whether is in training mode.

  Returns:
    Initial params, Init model_state, and number of trainable_params.
  """
  batch_size = config.batch_size
  if not is_train:
    batch_size = config.get('eval_batch_size', batch_size)
  dev_batch_size = batch_size // jax.device_count()
  dummy_input = []
  for spec in input_spec:
    if spec is not None:
      in_st = debug_utils.input_spec_to_jax_shape_dtype_struct(
          spec, batch_size=dev_batch_size)
      dummy_input.append(jnp.zeros(in_st.shape, in_st.dtype))
    else:
      dummy_input.append(None)
  # We want all parameters to be created in host RAM, not on any device, they'll
  # be sent there later as needed, otherwise we already encountered two
  # situations where we allocate them twice.
  @functools.partial(jax.jit, backend='cpu')
  def _initialize_model(rng):
    """Initialization function to be jitted."""
    d_rng, g_rng = jax.random.split(rng, 2)
    generator_variables = model_dict['generator'].init(g_rng, *dummy_input)
    g_model_state, g_params = generator_variables.pop('params')

    d_dummy_input = jax.tree_util.tree_map(
        lambda x, y: jnp.concatenate([x, y], axis=0), dummy_input, dummy_input)
    discriminator_variables = model_dict['discriminator'].init(
        d_rng, *d_dummy_input)
    d_model_state, d_params = discriminator_variables.pop('params')

    state_dict = dict(
        generator=g_model_state,
        discriminator=d_model_state)
    init_params_dict = dict(
        generator=g_params, discriminator=d_params)

    return state_dict, init_params_dict

  state_dict, init_params_dict = _initialize_model(rng)
  return state_dict, init_params_dict


def create_train_state(input_spec: Sequence[Union[Tuple[Tuple[int, ...],
                                                        jnp.dtype],
                                                  Tuple[int, ...], None]],
                       config: ml_collections.ConfigDict, rng: np.ndarray,
                       is_train: bool):
  """Initializes train state and optimizer with a pytree input_spec.

  If the root type of `input_spec` is `Sequence`, each element is fed to the
  model as position arguments. For the video dataset input the expected shape is
  always [local_bs, t, h, w, 3].

  This function interface is different from the standard scenic function
  `initialize_model`, which does three things initialize model, optimizer,
  and train_state.

  Args:
    input_spec: A PyTree whose leaves are (shape, dtype) pairs specifying the
      shape and dtype of the inputs. If unspecified the dtype is the default
      dtype. input_spec =
        [(dataset.meta_data['input_shape'],
          dataset.meta_data.get('input_dtype', dtype))]
    config: Configurations of the initialization.
    rng: Jax rng key.
    is_train: whether is in training mode.

  Returns:
    Model dict,  initial train_state, and learning_rate function.
  """
  model_dict = train_utils.get_vq_model(config)

  rng, init_rng = jax.random.split(rng)
  model_state_dict, init_params_dict = initialize_model(
      model_dict=model_dict,
      input_spec=input_spec,
      config=config,
      rng=init_rng,
      is_train=is_train)

  tx_dict, g_lr_fn, d_lr_fn = get_optimizer(config)
  del d_lr_fn

  g_params = flax.core.unfreeze(init_params_dict['generator'])
  d_params = flax.core.unfreeze(init_params_dict['discriminator'])

  # Create optimizer.
  # We jit this, such that the arrays that are created on the same
  # device as the input is, in this case the CPU. Else they'd be on device[0].
  g_opt_state = jax.jit(
      tx_dict['generator'].init, backend='cpu')(g_params)
  d_opt_state = jax.jit(
      tx_dict['discriminator'].init, backend='cpu')(d_params)

  logging.info('logging generator parameters')
  parameter_overview.log_parameter_overview(g_params)
  logging.info('logging discriminator parameters')
  parameter_overview.log_parameter_overview(d_params)
  if jax.tree_util.tree_leaves(model_state_dict):
    logging.info('logging model states')
    parameter_overview.log_parameter_overview(model_state_dict)
  ema_params = init_params_dict['generator']

  metadata = dict(
      lecam_ema_real=jnp.asarray(0.), lecam_ema_fake=jnp.asarray(0.))

  train_state = TrainState(
      global_step=0,
      g_opt_state=g_opt_state,
      d_opt_state=d_opt_state,
      g_tx=tx_dict['generator'],
      d_tx=tx_dict['discriminator'],
      g_params=g_params,
      d_params=d_params,
      g_model_state=flax.core.freeze(model_state_dict['generator']),
      d_model_state=flax.core.freeze(model_state_dict['discriminator']),
      ema_params=ema_params,
      metadata=metadata,
      rng=rng)
  return model_dict, train_state, g_lr_fn


def train_g_d(
    train_state: TrainState,
    batch: Batch,
    *,
    model_dict: Dict[str, nn.Module],
    config: ml_collections.ConfigDict) -> Tuple[TrainState, metrics.Collection]:
  """Perform a single training step of g and d."""

  dtype = train_utils.get_dtype(config)
  step = train_state.global_step
  new_rng, _ = jax.random.split(train_state.rng)

  generator = model_dict['generator']
  discriminator = model_dict['discriminator']
  mutable = ['batch_stats', 'spectral_norm_stats']

  if config.pretrained_image_model:
    additional_data = train_utils.get_additional_data()

  def loss_fn(params_d, params_g):
    g_variables = {'params': params_g, **train_state.g_model_state}
    d_variables = {'params': params_d, **train_state.d_model_state}
    real_video = batch['inputs']
    (generated_video, result_dict), new_g_variables = generator.apply(
        g_variables, real_video, is_train=True, mutable=mutable)

    if config.vqgan.gradient_penalty == 'none':
      all_videos = jnp.concatenate([real_video, generated_video])
      # TODO(Lijun-Yu): maybe dual discriminators
      logit, new_d_variables = discriminator.apply(
          d_variables, all_videos, mutable=mutable)
      logit = jnp.asarray(logit, dtype)
      real_logit, fake_logit = jnp.split(logit, 2)
      grad_penalty = 0.0
    elif config.vqgan.gradient_penalty == 'r1':
      real_logit, grad_penalty = losses.r1_gradient_penalty(
          lambda train: discriminator,
          d_variables,
          real_video,
          penalty_cost=config.vqgan.grad_penalty_cost)
      fake_logit, new_d_variables = discriminator.apply(
          d_variables, generated_video, mutable=mutable)
    else:
      raise ValueError(
          f'{config.vqgan.gradient_penalty} is not a recognized gradient'
          ' penalty type.'
      )

    real_logit = jnp.asarray(real_logit, dtype)
    fake_logit = jnp.asarray(fake_logit, dtype)
    real_pred = jnp.mean(real_logit)
    fake_pred = jnp.mean(fake_logit)

    if config.get('lecam_weight', 0.) > 0:
      metadata = train_state.metadata
      lecam_loss = losses.lecam_reg(real_pred, fake_pred,
                                    metadata['lecam_ema_real'],
                                    metadata['lecam_ema_fake'])
      lecam_loss = lecam_loss * config.lecam_weight
    else:
      lecam_loss = 0.0

    d_adversarial_loss = losses.discriminator_loss(
        real_logit=real_logit,
        fake_logit=fake_logit,
        loss_type=config.vqgan.loss_type)
    g_adversarial_loss = losses.generator_loss(
        fake_logit=fake_logit, loss_type=config.vqgan.loss_type
    ) * config.vqgan.g_adversarial_loss_weight

    reconstruction_loss = losses.l2_loss(real_video, generated_video)
    perceptual_loss = 0.0

    if config.perceptual_loss_weight != 0:
      assert config.pretrained_image_model
      # TODO(Lijun-Yu, roadjiang): maybe add a pretrained video model.
      perceptual_loss = losses.calculate_perceptual_loss_on_pretrained(
          additional_data['image_model'],
          additional_data['image_model_state'],
          train_utils.flatten_t_dim(real_video),
          train_utils.flatten_t_dim(generated_video),
          perceptual_loss_on_logit=config.perceptual_loss_on_logit
      ) * config.perceptual_loss_weight

    quantizer_loss = result_dict['quantizer_loss']
    if config.vqvae.get('entropy_loss_enlarge_steps', 0) > 0:
      quantizer_loss += (
          result_dict['entropy_loss']
          * config.vqvae.entropy_loss_enlarge_ratio
          * jnp.maximum(0, 1 - step / config.vqvae.entropy_loss_enlarge_steps)
      )
    logit_laplace_loss = 0.
    d_loss = d_adversarial_loss + grad_penalty + lecam_loss
    g_loss = (
        reconstruction_loss
        + g_adversarial_loss
        + perceptual_loss
        + quantizer_loss
        + logit_laplace_loss
    )
    new_g_state = flax.core.freeze(new_g_variables)
    new_d_state = flax.core.freeze(new_d_variables)
    return (d_loss,
            g_loss), (new_g_state, new_d_state, d_adversarial_loss,
                      grad_penalty, reconstruction_loss, g_adversarial_loss,
                      perceptual_loss, quantizer_loss, logit_laplace_loss,
                      lecam_loss, (real_pred, fake_pred))

  params_d = train_state.d_params
  params_g = train_state.g_params
  (d_loss,
   g_loss), func_vjp, (new_g_state, new_d_state, d_adversarial_loss,
                       grad_penalty, reconstruction_loss, g_adversarial_loss,
                       perceptual_loss, quantizer_loss, logit_laplace_loss,
                       lecam_loss, preds) = jax.vjp(
                           loss_fn, params_d, params_g, has_aux=True)

  d_grad, _ = func_vjp((1., 0.))
  _, g_grad = func_vjp((0., 1.))

  # Compute average gradient across multiple workers.
  d_grad = jax.lax.pmean(d_grad, axis_name='device')
  g_grad = jax.lax.pmean(g_grad, axis_name='device')

  d_updates, new_d_opt_state = train_state.d_tx.update(d_grad,
                                                       train_state.d_opt_state,
                                                       train_state.d_params)
  new_d_params = optax.apply_updates(train_state.d_params, d_updates)

  g_updates, new_g_opt_state = train_state.g_tx.update(g_grad,
                                                       train_state.g_opt_state,
                                                       train_state.g_params)
  new_g_params = optax.apply_updates(train_state.g_params, g_updates)

  ema_decay = config.polyak_decay

  new_ema_params = vqgan_train_lib.compute_ema_params(
      train_state.ema_params, new_g_params, config
  )

  if config.get('lecam_weight', 0.) > 0:
    metadata = train_state.metadata
    real_pred, fake_pred = preds
    new_metadata = {
        'lecam_ema_real':
            metadata['lecam_ema_real'] * ema_decay +
            (1 - ema_decay) * real_pred,
        'lecam_ema_fake':
            metadata['lecam_ema_fake'] * ema_decay +
            (1 - ema_decay) * fake_pred,
    }
  else:
    new_metadata = train_state.metadata

  new_state = train_state.replace(  # pytype: disable=attribute-error
      global_step=step + 1,
      g_opt_state=new_g_opt_state,
      d_opt_state=new_d_opt_state,
      g_params=new_g_params,
      d_params=new_d_params,
      g_model_state=new_g_state,
      d_model_state=new_d_state,
      ema_params=new_ema_params,
      metadata=new_metadata,
      rng=new_rng)
  metrics_update = TrainMetrics.gather_from_model_output(
      axis_name='device',
      g_loss=g_loss,
      d_loss=d_loss,
      d_adversarial_loss=d_adversarial_loss,
      grad_penalty=grad_penalty,
      reconstruction_loss=reconstruction_loss,
      g_adversarial_loss=g_adversarial_loss,
      perceptual_loss=perceptual_loss,
      quantizer_loss=quantizer_loss,
      logit_laplace_loss=logit_laplace_loss,
      lecam_loss=lecam_loss,
  )
  return new_state, metrics_update


def train_step(
    train_state: TrainState,
    batch: Batch,
    *,
    model_dict: Dict[str, nn.Module],
    config: ml_collections.ConfigDict) -> Tuple[Any, metrics.Collection]:
  """Runs a single step of training.

  Args:
    train_state: The state of training TrainState
    batch: A single batch of data. Dictionary where
      batch['inputs'].shape= device_bs, t, h, w, 3
    model_dict: A dictionary of generator and discriminator Flax models
    config: Configurations of the experiment.

  Returns:
    Updated state of training, computed metrics.
  """
  if config.vqgan.model_type == '2D' and config.get('dataset_type',
                                                    'video') == 'video':
    sampled_frames = video_utils.sample_frames_uniformly(
        batch['inputs'], n_sampled_frames=config.num_train_sampled_frames)
    batch = dict(inputs=sampled_frames)

  new_train_state, metrics_update = train_g_d(
      train_state,
      batch,
      model_dict=model_dict,
      config=config)
  return new_train_state, metrics_update


def sample_step(train_state: TrainState, batch: Batch, *,
                model_dict: Dict[str, nn.Module],
                config: ml_collections.ConfigDict):
  """Runs a single step to generate samples given input videos.

  Args:
    train_state: The state of training TrainState
    batch: A single batch of data. Dictionary where
      batch['inputs'].shape= device_bs, t, h, w, 3
      batch['batch_mask'].shape= device_bs
      batch['batch_mask'].dtype= float32
      where batch['batch_mask'] > 0 for valid examples
    model_dict: A dictionary of generator and discriminator Flax models
    config: Configurations of the experiment.

  Returns:
    Sampled videos formatted in spatial grids.
  """
  del config
  variables = {
      'params': train_state.g_params,
      **train_state.g_model_state
  }
  ema_variables = {
      'params': train_state.ema_params,
      **train_state.g_model_state
  }

  generator = model_dict['generator']
  generate_fn = functools.partial(generator.apply, variables)
  ema_generate_fn = functools.partial(generator.apply, ema_variables)
  outputs = {
      'original_video': batch['inputs'],
  }
  if 'batch_mask' in batch:
    outputs.update(dict(batch_mask=batch['batch_mask']))

  if 'label' in batch:
    outputs['label'] = batch['label']
  generated_video, _ = generate_fn(batch['inputs'])
  outputs['generated_video'] = jnp.clip(generated_video, 0, 1)
  generated_video_ema, result_dict_ema = ema_generate_fn(batch['inputs'])
  outputs['generated_video_ema'] = jnp.clip(generated_video_ema, 0, 1)
  outputs['generated_tokens_ema'] = result_dict_ema['encoding_indices']
  results = {}
  return outputs, results


def eval_step(
    train_state: TrainState,
    batch: Batch,
    *,
    model_dict: Dict[str, nn.Module],
    config: ml_collections.ConfigDict,
    metric_params: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
  """Runs a single step of evaluation.

  Args:
    train_state: The state of evaluating TrainState
    batch: A single batch of data. Dictionary where batch['inputs'].shape=
      device_bs, t, h, w, 3
    model_dict: A dictionary of generator and discriminator Flax models
    config: Configurations of the experiment.
    metric_params: Params for metric models.

  Returns:
    Metric features and generated outputs.
  """
  outputs, _ = sample_step(
      train_state, batch, model_dict=model_dict, config=config)

  dataset_type = config.get('dataset_type', 'video')
  if dataset_type == 'image':
    # Extend to 16 frames for IS and FVD.
    batch_mask = outputs.pop('batch_mask', None)
    tokens = outputs.pop('generated_tokens_ema')
    outputs = jax.tree_util.tree_map(lambda x: jnp.tile(x, (1, 16, 1, 1, 1)),
                                     outputs)
    outputs['generated_tokens_ema'] = tokens
    if batch_mask is not None:
      outputs['batch_mask'] = batch_mask

  features = eval_utils.eval_step_get_features(
      outputs,
      metric_params=metric_params,
      model_suffix_list=['', '_ema'],
      config=config)

  if dataset_type == 'image':
    # Reduce back to 1 frame.
    batch_mask = outputs.pop('batch_mask', None)
    tokens = outputs.pop('generated_tokens_ema')
    outputs = jax.tree_util.tree_map(lambda x: x[:, :1], outputs)
    outputs['generated_tokens_ema'] = tokens
    if batch_mask is not None:
      outputs['batch_mask'] = batch_mask

  if config.eval.get('enable_lpips', False):
    outputs = jax.lax.all_gather(outputs, axis_name='device', tiled=True)
  return features, outputs


def log_samples(batch_samples: Batch, writer: metric_writers.MetricWriter,
                prefix: str, step: int, log_sample_size: int,
                log_video: bool = True):
  """Logs a batch of samples as static image and video."""
  image_samples = {}
  for suffix in ['', '_ema']:
    key = f'generated_video{suffix}'
    image_samples[key] = train_utils.draw_frames_side_by_side(
        batch_samples['original_video'],
        batch_samples[key],
        show_num=log_sample_size)
  image_samples = {
      f'{prefix}/{k}_static': v for k, v in image_samples.items()
  }
  writer.write_images(step, image_samples)
  if log_video:
    video_samples = {}
    for suffix in ['', '_ema']:
      key = f'generated_video{suffix}'
      video_samples[key] = train_utils.draw_videos_side_by_side(
          batch_samples['original_video'],
          batch_samples[key],
          show_num=log_sample_size)
    video_samples = {f'{prefix}/{k}': v for k, v in video_samples.items()}
    video_samples = jax.tree_util.tree_map(
        lambda x: (x * 255.).astype(jnp.uint8), video_samples)
    writer.write_videos(step, video_samples)


def train(*, rng: jnp.ndarray, config: ml_collections.ConfigDict,
          dataset: dataset_utils.Dataset, workdir: str,
          writer: metric_writers.MetricWriter):
  """Main training loop lives in this function."""
  lead_host = jax.process_index() == 0
  dtype = train_utils.get_dtype(config)
  writer.write_hparams(eval_utils.flatten_config(config))

  # Build the flax_model and the optimizers.
  _, init_rng = jax.random.split(rng)
  input_spec = [(
      dataset.meta_data['input_shape'],  # bs, t, h, w, 3
      dataset.meta_data.get('input_dtype', dtype))]
  model_dict, train_state, learning_rate_fn = create_train_state(
      input_spec, config, init_rng, True)

  # Restores the model if needed.
  if config.logging.enable_checkpoint:
    train_state = checkpoints.restore_checkpoint(
        ckpt_dir=workdir, target=train_state)

  start_step = int(train_state.global_step)
  # Log model size
  if start_step == 0:
    if config.get('vqgan.finetune_path'):
      logging.info(
          'loading the model for finetuning from %s', config.vqgan.finetune_path
      )
      train_state = train_utils.restore_finetune_checkpoint(
          config.vqgan.finetune_path,
          train_state,
          step=None,
          not_restore_g_params_keys=None,
      )

    g_param_size = parameter_overview.count_parameters(train_state.g_params)
    d_param_size = parameter_overview.count_parameters(train_state.d_params)
    writer.write_scalars(
        0, {
            'param_size/generator': g_param_size,
            'param_size/discriminator': d_param_size
        })

  if (start_step == 0  # Which means "no" checkpoint is restored!
      and config.get('init_from') is not None):
    assert not (
        config.get('init_from') and config.get('vqgan.finetune_decoder')
    ), '3d-inflation and finetuning are conflicting setting.'
    init_checkpoint_path = config.init_from.get('checkpoint_path')
    train_state = vqgan_train_lib.init_from_pretrained_checkpoint(
        init_checkpoint_path, train_state, config)

  # Get learning rate scheduler.
  total_steps, steps_per_epoch = scenic_train_utils.get_num_training_steps(
      config, dataset.meta_data)

  train_utils.check_training_step(steps_per_epoch, config)

  # Training step pmap.
  train_state = jax_utils.replicate(train_state)
  train_step_pmapped = jax.pmap(
      functools.partial(
          train_step,
          model_dict=model_dict,
          config=config),
      axis_name='device',
      # We can donate the buffer of train_state.
      donate_argnums=(0),
  )

  sample_step_pmapped = jax.pmap(
      functools.partial(sample_step, model_dict=model_dict, config=config),
      axis_name='device',
      # We can donate the buffer of train_batch.
      donate_argnums=(1),
  )

  checkpoint_steps = config.logging.get('checkpoint_steps', steps_per_epoch)
  log_metric_steps = config.logging.get('log_metric_steps', checkpoint_steps)
  log_sample_size = config.logging.get('log_sample_size', 2)

  logging.info('Starting training loop at step %d of total_steps=%d.',
               start_step, total_steps)

  report_progress = periodic_actions.ReportProgress(
      num_train_steps=total_steps, writer=writer)
  hooks = [report_progress]
  if config.get('xprof', True) and lead_host:
    hooks.append(periodic_actions.Profile(num_profile_steps=5, logdir=workdir))

  train_metrics = None
  with metric_writers.ensure_flushes(writer):
    for step in range(start_step + 1, total_steps + 1):
      with jax.profiler.StepTraceAnnotation('train', step_num=step):
        train_batch = next(dataset.train_iter)
        if config.dataset_configs.get('one_hot_labels', False):
          train_batch.pop('label', None)
        train_state, metrics_update = train_step_pmapped(
            train_state, train_batch)
        metric_update = jax_utils.unreplicate(metrics_update)
        train_metrics = (
            metric_update
            if train_metrics is None else train_metrics.merge(metric_update))

      # Quick indication that training is happening.
      logging.log_first_n(logging.INFO, 'Finished training step %d.', 5, step)
      if step < 200 and config.get('vqgan.finetune_decoder'):
        # checks the finetuning parameter subset.
        encoder_params_sum = train_utils.sum_model_params(
            train_state.g_params['encoder'])
        decoder_params_sum = train_utils.sum_model_params(
            train_state.g_params['decoder'])
        quantizer_params_sum = train_utils.sum_model_params(
            train_state.g_params['quantizer'])
        logging.log_first_n(
            logging.INFO,
            'Params_sum: decoder %f, quantizer %f, encoder %f.', 200,
            decoder_params_sum, quantizer_params_sum, encoder_params_sum)

      for h in hooks:
        try:
          h(step)
        except:  # pylint: disable=bare-except
          logging.exception('Hook failed')
          continue

      ############### LOG TRAIN METRICS ###############
      if (step % log_metric_steps == 1) or (step == total_steps):
        train_metrics = train_metrics.compute()
        train_metrics = {f'train/{k}': v for k, v in train_metrics.items()}
        writer.write_scalars(step, train_metrics)
        writer.write_scalars(step, {'train/lr': learning_rate_fn(step-1)})
        # Reset metric accumulation for next evaluation cycle.
        if step < total_steps:
          train_metrics = None

      ##################### CHECKPOINTING ############################
      if ((step % checkpoint_steps == 1) or (step == total_steps)):
        with report_progress.timed('checkpoint'):
          # Sync model state across replicas.
          train_state = vqgan_train_lib.sync_model_state_across_replicas(
              train_state)
          if config.logging.enable_checkpoint:
            train_utils.save_checkpoint(workdir, train_state,
                                        config.logging.checkpoint_kept)

        ############### LOG TRAIN SAMPLES ###############
        with report_progress.timed('sample'):
          batch_to_sample = jax.tree_util.tree_map(
              lambda x: x[:, :log_sample_size], train_batch)
          batch_samples, _ = sample_step_pmapped(train_state, batch_to_sample)
          batch_samples = jax.device_get(jax_utils.unreplicate(batch_samples))
          log_samples(batch_samples, writer, 'train', step, log_sample_size,
                      config.get('dataset_type', 'video') == 'video')
          del batch_to_sample, batch_samples

  # Tell evaluation job to stop.
  task_manager = eval_utils.TaskManager(workdir)
  task_manager.mark_training_done()
  logging.info('Finishing training at step %d', total_steps)
  # Return the train and eval summary after last step for regression testing.
  return train_state, train_metrics


def _compute_lpips_one_step(batch_outputs, metric_params):
  """Compute the lpips for one batch of generated videos."""
  # lpips calls tf_hub which cannot be jit/pmaped by tf2jax
  cur_batch = batch_outputs
  original_videos = cur_batch['original_video']
  generated_videos = cur_batch['generated_video_ema']
  assert generated_videos.ndim == 5  # total_bs, t, h, w, 3
  return eval_utils.run_lpips_models(
      original_videos,
      generated_videos,
      metric_functions=metric_params['lpips'])


def evaluate(*, rng: jnp.ndarray, config: ml_collections.ConfigDict,
             dataset: dataset_utils.Dataset, workdir: str,
             writer: metric_writers.MetricWriter):
  """Main evaluation loop lives in this function."""
  lead_host = jax.process_index() == 0
  dtype = train_utils.get_dtype(config)
  config, ckpt_manager, ckpt_list = eval_utils.get_eval_jobs(workdir, config)

  # Build the flax_model and the optimizers.
  _, init_rng = jax.random.split(rng)
  input_spec = [(
      dataset.meta_data['input_shape'],  # bs, t, h, w, 3
      dataset.meta_data.get('input_dtype', dtype))]
  model_dict, train_state, _ = create_train_state(input_spec, config, init_rng,
                                                  False)
  total_train_steps, _ = scenic_train_utils.get_num_training_steps(
      config, dataset.meta_data)

  # Load metric params
  metric_params = eval_utils.load_metric_params(config)

  # Eval step pmap.
  eval_step_pmapped = jax.pmap(
      functools.partial(
          eval_step,
          model_dict=model_dict,
          config=config,
          metric_params=metric_params,
      ),
      axis_name='device',
      # We can donate the buffer of batch.
      donate_argnums=(1),
  )

  log_sample_size = config.logging.get('log_sample_size', 2)
  label_names, result_dir, write_executor = None, None, None
  if config.eval.get('results_dir') is not None:
    label_names = eval_utils.load_label_names(config)
    result_dir = eval_utils.get_result_dir(config)
    write_executor = concurrent.futures.ThreadPoolExecutor(100)

  all_metrics, batch_samples = {}, None
  for ckpt_path in ckpt_list:
    # Restores the model
    if not gfile.exists(ckpt_path):
      logging.warn(
          'Unable to evaluate ckpt %s because it does not exist. '
          'If this is a parallel evaluation job, try to increase '
          'config.logging.checkpoint_kept or use more accelerators.', ckpt_path)
      continue

    train_state = train_utils.restore_checkpoint(
        ckpt_path,
        train_state,
        is_legacy_checkpoint=config.eval_from.get('legacy_checkpoint', False))
    global_step = int(train_state.global_step)
    start_step = global_step  # Step offset for logging
    train_state_replicated = jax_utils.replicate(train_state)
    logging.info('Starting evaluation at global step %d', global_step)

    is_final_eval = False
    if global_step == total_train_steps or config.get('eval_only', False):
      is_final_eval = True
    data_iters = eval_utils.get_eval_data_iters(dataset, config, is_final_eval)
    eval_metrics = {}
    for data_split, (data_iter, num_steps, total_num_examples,
                     num_repeats) in data_iters.items():
      if not config.get('eval_only', False):
        report_progress = eval_utils.ReportEvaluationProgress(
            num_train_steps=None, writer=writer)
      else:
        report_progress = periodic_actions.ReportProgress(
            num_train_steps=start_step + num_steps, writer=writer)
      hooks = [report_progress]
      if config.get('xprof', False) and lead_host:
        hooks.append(
            periodic_actions.Profile(num_profile_steps=5, logdir=workdir))

      logging.info(
          'Evaluating for %d steps on %s split', num_steps, data_split)
      metric_suffix = f"{data_split}{config.eval.get('suffix', '')}"

      eval_features, batch_outputs = [], {}
      with metric_writers.ensure_flushes(writer):
        for step in range(num_steps):
          eval_batch = next(data_iter)
          if config.dataset_configs.get('one_hot_labels', False):
            eval_batch.pop('label', None)
          batch_features, batch_outputs = eval_step_pmapped(
              train_state_replicated, eval_batch)
          batch_features = jax.device_get(jax_utils.unreplicate(batch_features))
          eval_features.append(batch_features)
          # Quick indication that evaluation is happening.
          logging.log_first_n(logging.INFO, 'Finished evaluation step %d.', 5,
                              step)
          for h in hooks:
            try:
              h(start_step + step)
            except:  # pylint: disable=bare-except
              logging.exception('Hook failed')
              continue
          if write_executor is not None or config.eval.get(
              'enable_lpips', False):
            batch_outputs = jax.device_get(batch_outputs)
            if config.eval.get('enable_lpips', False):
              # batch_outputs has been all_gathered
              batch_samples = jax_utils.unreplicate(batch_outputs)
              lpips_scores = _compute_lpips_one_step(batch_samples,
                                                     metric_params)
              eval_features[-1].update(lpips_scores)
              # only keep the local batch for writing
              batch_samples = eval_utils.get_local_batch(batch_samples)
            else:
              batch_samples = jax.tree_util.tree_map(
                  lambda x: x.reshape(-1, *x.shape[2:]), batch_outputs)
            if write_executor is not None:
              # batch_samples is local flattened batch as np.array
              # batch_features is global flattened batch as np.array
              batch_features = eval_utils.get_local_batch(batch_features)
              batch_samples.update(batch_features)
              eval_utils.write_batch_samples(
                  batch_samples, step, write_executor, result_dir,
                  f'{metric_suffix}_step{global_step}', label_names, config)

        start_step += num_steps
        eval_features = eval_utils.gather_outputs_with_mask(
            eval_features, num_samples=total_num_examples)
        eval_metrics = eval_utils.compute_eval_metrics(
            eval_features,
            config,
            total_num_examples,
            num_repeats,
            model_suffix_list=['', '_ema'])
        eval_metrics = {
            f'eval_{metric_suffix}/{k}': v for k, v in eval_metrics.items()
        }
        writer.write_scalars(global_step, eval_metrics)
        all_metrics.update(eval_metrics)
        # Sample from last batch.
        batch_samples = jax.device_get(
            jax.tree_util.tree_map(lambda x: x[0, :log_sample_size],
                                   batch_outputs))
        log_samples(batch_samples, writer, f'eval_{metric_suffix}', global_step,
                    log_sample_size,
                    config.get('dataset_type', 'video') == 'video')

    del train_state_replicated
    ckpt_manager.add_eval_result(ckpt_path, eval_metrics, 0)
    logging.info('Finishing evaluation at global step %d', global_step)

  if write_executor is not None:
    write_executor.shutdown()
  return train_state, all_metrics, batch_samples
