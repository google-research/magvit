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

"""MaskGVT Trainer."""
import functools
import itertools
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

from absl import logging
from clu import metric_writers
from clu import metrics
from clu import parameter_overview
from clu import periodic_actions
from flax import jax_utils
from flax.core import unfreeze
import flax.linen as nn
from flax.training import checkpoints
import jax
from jax.example_libraries.optimizers import clip_grads
from jax.example_libraries.optimizers import l2_norm
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
from scenic.common_lib import debug_utils
from scenic.dataset_lib import dataset_utils
from scenic.model_lib.base_models import model_utils as scenic_model_utils
from scenic.train_lib import lr_schedules
from scenic.train_lib import train_utils as scenic_train_utils
from tensorflow.io import gfile
from videogvt.models import transformer_lm
from videogvt.train_lib import eval_utils
from videogvt.train_lib import mask_utils
from videogvt.train_lib import maskgvt_train_lib
from videogvt.train_lib import task_registry
from videogvt.train_lib import train_utils


InputSpec = Sequence[Union[Tuple[Tuple[int, ...], jnp.dtype], Tuple[int, ...],
                           None]]
Batch = train_utils.Batch
LossFn = Callable[[jnp.ndarray, Batch, Optional[jnp.ndarray]], float]
Task = task_registry.Task
TrainState = maskgvt_train_lib.TrainState
TrainMetrics = maskgvt_train_lib.TrainMetrics
TokenizerFn = maskgvt_train_lib.TokenizerFn


def _get_dummy_inputs(input_spec: InputSpec,
                      task: Task,
                      tokenizer_fn: TokenizerFn,
                      config: ml_collections.ConfigDict,
                      batch_size: Optional[int] = None):
  """Gets the dummy inputs for the transformer model.

  Args:
    input_spec: An iterable of (shape, dtype) pairs specifying the shape and
      dtype of the inputs. If unspecified the dtype is float32.
    task: Task.
    tokenizer_fn: The encoder of vq-model to tokenize the video.
    config: Configurations of the initialization.
    batch_size: Optional batch size

  Returns:
    Input of jnp.array
  """
  if batch_size is None:
    batch_size = (config.batch_size //
                  jax.device_count()) if config.get('batch_size') else None
  dummy_input = []
  for spec in input_spec:
    if spec is not None:
      in_st = debug_utils.input_spec_to_jax_shape_dtype_struct(
          spec, batch_size=batch_size)
      dummy_input.append(jnp.zeros(in_st.shape, in_st.dtype))
    else:
      dummy_input.append(None)

  # creates the input format for the transformer_model
  x = tokenizer_fn(*dummy_input)

  expected_shape = tuple(config.transformer.latent_shape)
  encoder_shape = x.shape[1:]

  if expected_shape != encoder_shape:
    raise ValueError(
        f'{expected_shape} not match the encoder output {encoder_shape}.')

  batch = {
      'inputs': x,
      'cond_inputs': x,
      'cond_mask': x,
      'label': jnp.zeros((batch_size, 1), jnp.int32),
  }
  task = task._replace(mask_fn=mask_utils.all_mask)
  batch_tokens = maskgvt_train_lib.generate_tokens_for_lm(
      batch, task, config)
  return batch_tokens['inputs']


def initialize_cache(inputs: np.ndarray,
                     *,
                     task: Task,
                     model: nn.Module,
                     input_spec: InputSpec,
                     tokenizer_fn: TokenizerFn,
                     config: ml_collections.ConfigDict):
  """Initializes the cache variables for self-attention layers."""
  dummy_inputs = _get_dummy_inputs(input_spec, task, tokenizer_fn, config,
                                   batch_size=inputs.shape[0])
  initial_variables = model.init(jax.random.PRNGKey(0), dummy_inputs)
  return initial_variables['cache']


def initialize_model(*,
                     model: nn.Module,
                     dummy_inputs: jnp.ndarray,
                     rng: jnp.ndarray):
  """Initializes parameters and model state.

  Args:
    model: The model.
    dummy_inputs: Dummy model inputs.
    rng: Jax rng keys.

  Returns:
    Init model_state, Initial params.
  """
  # We want all parameters to be created in host RAM, not on any device, they'll
  # be sent there later as needed, otherwise we already encountered two
  # situations where we allocate them twice.
  @functools.partial(jax.jit, backend='cpu')
  def _initialize_model(rng):
    """Initialization function to be jitted."""
    model_variables = model.init(rng, dummy_inputs)
    model_state = dict(model_variables)
    init_params = model_state.pop('params')
    return model_state, init_params
  model_state, init_params = _initialize_model(rng)
  return model_state, init_params


def create_train_state(input_spec: InputSpec,
                       task: Task,
                       tokenizer_fn: TokenizerFn,
                       config: ml_collections.ConfigDict,
                       rng: np.ndarray):
  """Creates train state."""
  rng, init_rng = jax.random.split(rng, 2)
  dummy_inputs = _get_dummy_inputs(input_spec, task, tokenizer_fn, config)
  total_seq_len = dummy_inputs.shape[1]  # pytype: disable=attribute-error  # jax-ndarray
  logging.info('Total sequence length: %d', total_seq_len)
  vq_codebook_size = maskgvt_train_lib.get_vq_codebook_size(config)
  num_classes = maskgvt_train_lib.get_num_classes(config)
  num_special_tokens = maskgvt_train_lib.get_num_special_tokens(config)
  vocab_size = vq_codebook_size + num_classes + num_special_tokens

  model_config = transformer_lm.TransformerConfig(
      vocab_size=vocab_size,
      output_vocab_size=vocab_size,
      emb_dim=config.transformer.hidden_size,
      num_heads=config.transformer.num_heads,
      num_layers=config.transformer.num_layers,
      qkv_dim=config.transformer.hidden_size,
      mlp_dim=config.transformer.mlp_dim,
      dropout_rate=config.transformer.dropout_rate,
      attention_dropout_rate=config.transformer.attention_dropout_rate,
      max_len=total_seq_len,
      max_predict_length=total_seq_len)

  if not config.transformer.learnable_position:
    model_config = model_config.replace(posemb_init=None)

  model = transformer_lm.TransformerLM(model_config)

  model_state, init_params = initialize_model(
      model=model, dummy_inputs=dummy_inputs, rng=init_rng)
  logging.info('logging transformer parameters')
  parameter_overview.log_parameter_overview(init_params)
  if jax.tree_leaves(model_state):
    logging.info('logging model states')
    parameter_overview.log_parameter_overview(model_state)

  learning_rate_fn = lr_schedules.get_learning_rate_fn(config)
  tx = maskgvt_train_lib.get_standard_adamw_optimizer(learning_rate_fn, config)

  # Create optimizer.
  # Needs to unfreeze params https://github.com/deepmind/optax/issues/160
  init_params = unfreeze(init_params)
  opt_state = jax.jit(tx.init, backend='cpu')(init_params)

  train_state = TrainState(
      global_step=0,
      opt_state=opt_state,
      tx=tx,
      params=init_params,
      model_state=model_state,
      rng=rng,
      metadata=dict())

  return model, model_config, train_state, learning_rate_fn


def train_step(
    train_state: TrainState, batch: Batch, task: Task, *,
    flax_model: nn.Module,
    config: ml_collections.ConfigDict) -> Tuple[Any, metrics.Collection]:
  """Runs a single step of training.

  Args:
    train_state: The state of training TrainState
    batch: A single batch of data. Dictionary where
      batch['inputs'].shape= device_bs, t, h, w, 3
    task: Current training task.
    flax_model: Flax model
    config: Configurations of the experiment.

  Returns:
    Updated state of training, computed metrics.
  """
  new_rng, dropout_rng = jax.random.split(
      train_state.rng, 2)

  batch_tokens = maskgvt_train_lib.generate_tokens_for_lm(batch, task, config)

  # Bind the dropout rng to the host/device we are on.
  dropout_rng = scenic_train_utils.bind_rng_to_host_device(
      dropout_rng, axis_name='device', bind_to='device')
  mutable = ['batch_stats', 'spectral_norm_stats']

  def training_loss_fn(params):
    variables = {'params': params, **train_state.model_state}
    logits, new_model_state = flax_model.apply(
        variables,
        batch_tokens['inputs'],
        mutable=mutable,
        rngs={'dropout': dropout_rng}, deterministic=False)
    # logits shape [bs, 1 + (l_cond) + l_t * l_h * l_w,
    #               vq_codebook_size + num_classes + num_special_tokens]
    # comment out this because we need to update the paramters for
    # num_special_tokens and num_classes
    # logits = logits[:, :, :vq_codebook_size]
    one_hot_targets = jax.nn.one_hot(batch_tokens['targets'], logits.shape[-1])

    sof_ce_loss = (
        scenic_model_utils.weighted_unnormalized_softmax_cross_entropy(
            logits,
            one_hot_targets,
            weights=batch.get('batch_mask'),
            label_smoothing=config.get('label_smoothing'),
        )
    )

    weights = batch_tokens['weights']
    # seq loss rather than token average loss
    masked_sof_ce_loss = jnp.sum(
        sof_ce_loss * weights, axis=-1) / (
            jnp.sum(weights, axis=-1) + 1e-8)
    token_loss = jnp.mean(masked_sof_ce_loss)

    l2_loss = 0
    if config.get('l2_decay_factor') is None:
      total_loss = token_loss
    else:
      l2_loss = scenic_model_utils.l2_regularization(params)
      total_loss = token_loss + 0.5 * config.l2_decay_factor * l2_loss

    return total_loss, (new_model_state, token_loss, l2_loss)

  compute_gradient_fn = jax.value_and_grad(training_loss_fn, has_aux=True)
  (total_loss, (new_model_state, token_loss,
                l2_loss)), grad = compute_gradient_fn(train_state.params)

  # Re-use same axis_name as in the call to `pmap(...train_step...)` below.
  grad = jax.lax.pmean(grad, axis_name='device')
  if config.get('max_grad_norm') is not None:
    grad = clip_grads(grad, config.max_grad_norm)
  grad_norm = l2_norm(grad)
  updates, new_opt_state = train_state.tx.update(grad, train_state.opt_state,
                                                 train_state.params)
  new_params = optax.apply_updates(train_state.params, updates)

  new_train_state = train_state.replace(  # pytype: disable=attribute-error
      global_step=train_state.global_step + 1,
      opt_state=new_opt_state,
      params=new_params,
      model_state=new_model_state,
      rng=new_rng)
  metrics_update = TrainMetrics.gather_from_model_output(
      axis_name='device',
      total_loss=total_loss,
      token_loss=token_loss,
      l2_loss=l2_loss,
      grad_norm=grad_norm,
      mask_ratio=0.)
  return new_train_state, metrics_update


def sample_step(train_state: TrainState, batch: Batch, task: Task,
                *, flax_model: nn.Module,
                config: ml_collections.ConfigDict):
  """Runs a single step to predict tokens during training.

  Args:
    train_state: The state of training TrainState
    batch: A single batch of data. Dictionary where
      batch['input_indices'].shape= device_bs, l_t*l_h*l_w+1, vocabulary_size
    task: Current training task.
    flax_model: Flax model
    config: Configurations of the experiment.

  Returns:
    Token predition visualized as images formatted in spatial grids.
  """
  vq_codebook_size = maskgvt_train_lib.get_vq_codebook_size(config)
  batch_tokens = maskgvt_train_lib.generate_tokens_for_lm(batch, task, config)
  variables = {'params': train_state.params, **train_state.model_state}
  logits = flax_model.apply(
      variables, batch_tokens['inputs'], deterministic=True)
  # only illustrate the last latent_seq_len tokens.
  logits = logits[..., :vq_codebook_size]
  # we do not do sampling.sampling as there is no sampling during training
  # do not use batch_tokens['inputs'] here as it has mask token as bos.
  pred_visualization = maskgvt_train_lib.illustrate_predictions(
      batch_tokens['targets'], logits, batch_tokens['targets'], config)
  outputs = maskgvt_train_lib.draw_video_boundary(pred_visualization)
  results = {}
  return outputs, results


def generate_step(train_state: TrainState,
                  batch: Batch,
                  cache: Dict[str, jnp.ndarray],
                  task: Task,
                  rng: jnp.ndarray,
                  *,
                  flax_model: nn.Module,
                  tokenizer_dict: Dict[str, TokenizerFn],
                  config: ml_collections.ConfigDict):
  """Runs generation given a batch of inputs.

  Args:
    train_state: The state of training TrainState
    batch: A single batch of data. Dictionary where batch['inputs'].shape=
      device_bs, t, h, w, 3 batch['batch_mask'].shape= device_bs
      batch['batch_mask'].dtype= float32 where batch['batch_mask'] > 0 for valid
      examples
    cache: Intermediate variables for self-attention layers.
    task: Current generation task.
    rng: The PRNGKey for generation sampling
    flax_model: Flax model
    tokenizer_dict: A dictionary of tokenizer and detokenizer
    config: Configurations of the experiment.

  Returns:
    Sampled videos formatted in spatial grids.
  """
  variables = {'params': train_state.params, **train_state.model_state}
  latent_shape = (batch['inputs'].shape[0], *config.transformer.latent_shape)

  # Tokenize
  input_video = batch.pop('inputs')
  outputs = {
      'original_video': input_video,
      'batch_mask': batch['batch_mask']
  }
  if task.cond_fn is not None:
    cond_dict = task.cond_fn(input_video)
    outputs['condition_video'] = cond_dict['video']
    outputs['condition_mask'] = cond_dict['video_mask']
    batch['cond_inputs'] = tokenizer_dict['tokenizer'](cond_dict.pop('video'))
    batch['cond_mask'] = cond_dict.pop('latent_mask')
    outputs['condition_latents'] = batch['cond_inputs']
  batch['inputs'] = jnp.zeros(latent_shape, dtype=jnp.int32)

  if 'label' in batch:
    outputs['label'] = batch['label']

  lm_tokens = maskgvt_train_lib.generate_tokens_for_lm(
      batch, task, config)
  inputs, cond_lengths = lm_tokens['inputs'], lm_tokens['cond_lengths']
  generated_seq = maskgvt_train_lib.lm_decode(
      inputs, cond_lengths, cache, rng, variables, flax_model, config)
  outputs['generated_latents'] = generated_seq.reshape(latent_shape)
  # Detokenize
  generated_video = tokenizer_dict['detokenizer'](outputs['generated_latents'])
  if task.cond_fn is not None and config.sampling.get(
      'replace_condition_pixels', False):
    generated_video = jnp.where(
        cond_dict.pop('video_mask'), input_video, generated_video)
  outputs['generated_video'] = jnp.clip(generated_video, 0, 1)
  results = {}
  return outputs, results


def eval_step(
    train_state: TrainState,
    batch: Batch,
    cache: Dict[str, jnp.ndarray],
    task: Task,
    rng: jnp.ndarray,
    *,
    flax_model: nn.Module,
    tokenizer_dict: Dict[str, TokenizerFn],
    config: ml_collections.ConfigDict,
    metric_params: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
  """Runs a single step of evaluation.

  Args:
    train_state: The state of evaluating TrainState
    batch: A single batch of data. Dictionary where batch['inputs'].shape=
      device_bs, t, h, w, 3
    cache: cached states for authoressive decoding
    task: Current evaluation task.
    rng: A PRNGKey for current step.
    flax_model: Flax model
    tokenizer_dict: A dictionary of tokenizer and detokenizer
    config: Configurations of the experiment.
    metric_params: Params for metric models.

  Returns:
    Metric features and generated outputs.
  """
  outputs, _ = generate_step(
      train_state,
      batch,
      cache,
      task,
      rng,
      flax_model=flax_model,
      tokenizer_dict=tokenizer_dict,
      config=config)

  features = eval_utils.eval_step_get_features(
      outputs,
      metric_params=metric_params,
      model_suffix_list=[''],
      config=config)

  if config.eval.get('return_samples'):
    outputs = jax.lax.all_gather(outputs, axis_name='device', tiled=True)
  return features, outputs


def log_samples(batch_samples: Batch,
                writer: metric_writers.MetricWriter,
                data_split: str,
                step: int,
                log_sample_size: int,
                log_video: bool = True):
  """Logs a batch of samples as static image and video."""
  video_batches = [
      batch_samples['original_video'],
      batch_samples['generated_video'],
  ]
  if 'condition_video' in batch_samples:
    condition_video = jnp.where(
        batch_samples['condition_mask'], batch_samples['condition_video'],
        batch_samples['condition_video'] * 0.3)
    video_batches.append(condition_video)
  image_sample = train_utils.draw_frames_side_by_side(
      *video_batches, show_num=log_sample_size)
  image_samples = {f'{data_split}/generated_video_static': image_sample}
  writer.write_images(step, image_samples)
  if log_video:
    video_sample = train_utils.draw_videos_side_by_side(
        *video_batches, show_num=log_sample_size)
    video_samples = {f'{data_split}/generated_video': video_sample}
    video_samples = jax.tree_util.tree_map(
        lambda x: (x * 255.).astype(jnp.uint8), video_samples)
    writer.write_videos(step, video_samples)


def train(*, rng: jnp.ndarray, config: ml_collections.ConfigDict,
          dataset: dataset_utils.Dataset, workdir: str,
          writer: metric_writers.MetricWriter):
  """Main training loop lives in this function."""
  lead_host = jax.process_index() == 0
  dtype = train_utils.get_dtype(config)

  tokenizer_fn = maskgvt_train_lib.load_vq_tokenizer(
      config, tokenizer_only=True)['tokenizer']
  writer.write_hparams(eval_utils.flatten_config(config))

  # Build the flax_model and the optimizers.
  rng, init_rng, task_rng = jax.random.split(rng, 3)
  input_spec = [(
      dataset.meta_data['input_shape'],  # bs, t, h, w, 3
      dataset.meta_data.get('input_dtype', dtype))]

  # Get training tasks
  task_reg = task_registry.get_task_registry(config, True)
  task_ids = config.get('tasks', ('full_generation',))
  logging.info('Training tasks %s', task_reg)

  model, model_config, train_state, learning_rate_fn = create_train_state(
      input_spec, task_reg[task_ids[0]], tokenizer_fn, config, init_rng)
  del model_config

  # Restores the transformer model if needed.
  if config.logging.enable_checkpoint:
    train_state = checkpoints.restore_checkpoint(
        ckpt_dir=workdir, target=train_state)
  start_step = int(train_state.global_step)

  # Log model size
  if start_step == 0:
    param_size = parameter_overview.count_parameters(
        train_state.params)
    writer.write_scalars(0, {'param_size': param_size})

  if (start_step == 0  # Which means "no" checkpoint is restored!
      and config.get('init_from') is not None):
    init_checkpoint_path = config.init_from.get('checkpoint_path')
    train_state = maskgvt_train_lib.init_from_pretrained_checkpoint(
        init_checkpoint_path, train_state,
        config.init_from.get('expand_embedding', False), config)

  # Get learning rate scheduler.
  total_steps, steps_per_epoch = scenic_train_utils.get_num_training_steps(
      config, dataset.meta_data)
  train_utils.check_training_step(steps_per_epoch, config)

  # pmap
  tokenizer_pmapped = jax.pmap(
      functools.partial(
          maskgvt_train_lib.tokenize_with_cond,
          tokenizer_fn=tokenizer_fn),
      axis_name='device',
      static_broadcasted_argnums=(1,),
      donate_argnums=(0),)

  train_state = jax_utils.replicate(train_state)
  train_step_pmapped = jax.pmap(
      functools.partial(train_step, flax_model=model, config=config),
      axis_name='device',
      static_broadcasted_argnums=(2,),
      # We can donate the buffer of train_state.
      donate_argnums=(0),
  )

  sample_step_pmapped = jax.pmap(
      functools.partial(sample_step, flax_model=model, config=config),
      static_broadcasted_argnums=(2,),
      axis_name='device',
  )

  checkpoint_steps = config.logging.get('checkpoint_steps', steps_per_epoch)
  log_metric_steps = config.logging.get('log_metric_steps', checkpoint_steps)
  log_sample_size = config.logging.get('log_sample_size', 8)

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
        cur_task_rng = jax.random.fold_in(task_rng, step)
        # TODO(Lijun-Yu): use probabilistic sampling to favor stronger tasks.
        task_idx = jax.random.choice(cur_task_rng, len(task_ids))
        task = task_reg[task_ids[task_idx]]
        train_batch.update(
            tokenizer_pmapped(train_batch.pop('inputs'), task.cond_fn))
        train_state, metrics_update = train_step_pmapped(
            train_state, train_batch, task)
        metric_update = jax_utils.unreplicate(metrics_update)
        train_metrics = (
            metric_update
            if train_metrics is None else train_metrics.merge(metric_update))
      # Quick indication that training is happening.
      logging.log_first_n(logging.INFO, 'Finished training step %d.', 5, step)
      for h in hooks:
        h(step)

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
          train_state = maskgvt_train_lib.sync_model_state_across_replicas(
              train_state)
          if config.logging.enable_checkpoint:
            train_utils.save_checkpoint(workdir, train_state,
                                        config.logging.checkpoint_kept)

        ############### LOG TRAIN SAMPLES ###############
        with report_progress.timed('sample'):
          sample_batch = jax.tree_util.tree_map(
              lambda x: x[:, :log_sample_size], train_batch)
          batch_outputs, _ = sample_step_pmapped(train_state, sample_batch,
                                                 task)
          batch_outputs = jax.device_get(jax_utils.unreplicate(batch_outputs))
          samples = {}
          key = 'predicted_tokens'
          samples[key] = batch_outputs
          samples = {f'train_sample/{k}': v for k, v in samples.items()}
          writer.write_images(step, samples)
          del sample_batch, batch_outputs, samples

  # Tell evaluation job to stop.
  task_manager = eval_utils.TaskManager(workdir)
  task_manager.mark_training_done()
  # Wait until computations are done before exiting.
  jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()
  logging.info('Finishing training at step %d', total_steps)
  # Return the train and eval summary after last step for regression testing.
  return train_state, train_metrics


def evaluate(*, rng: jnp.ndarray, config: ml_collections.ConfigDict,
             dataset: dataset_utils.Dataset, workdir: str,
             writer: metric_writers.MetricWriter):
  """Main evaluation loop lives in this function."""
  lead_host = jax.process_index() == 0
  dtype = train_utils.get_dtype(config)
  config, ckpt_manager, ckpt_list = eval_utils.get_eval_jobs(workdir, config)

  tokenizer_dict = maskgvt_train_lib.load_vq_tokenizer(config)
  writer.write_hparams(eval_utils.flatten_config(config))

  # Build the flax_model and the optimizers.
  rng, init_rng = jax.random.split(rng, 2)
  input_spec = [(
      dataset.meta_data['input_shape'],  # bs, t, h, w, 3
      dataset.meta_data.get('input_dtype', dtype))]

  # Get evaluation task
  task_reg = task_registry.get_task_registry(config, False)
  task_ids = config.get('tasks', ('full_generation',))
  if 'task' in config.eval:
    task_ids = (config.eval.task,)
  assert len(task_ids) == 1
  # Loop through all tasks during parallel evaluation
  task_ids_iter = itertools.cycle(task_ids)

  model, model_config, train_state, _ = create_train_state(
      input_spec, task_reg[task_ids[0]], tokenizer_dict['tokenizer'],
      config, init_rng)
  del model
  # to avoid loading cache from checkpoints
  pred_model_config = model_config.replace(decode=True)
  pred_model = transformer_lm.TransformerLM(pred_model_config)

  total_train_steps, _ = scenic_train_utils.get_num_training_steps(
      config, dataset.meta_data)

  # Load metric params
  metric_params = eval_utils.load_metric_params(config)

  # Pmap.
  eval_step_pmapped = jax.pmap(
      functools.partial(
          eval_step,
          flax_model=pred_model,
          tokenizer_dict=tokenizer_dict,
          config=config,
          metric_params=metric_params,
      ),
      static_broadcasted_argnums=(3,),
      axis_name='device',
      # We can donate the buffer of batch.
      donate_argnums=(1),
  )

  init_cache_pmapped = jax.pmap(
      functools.partial(
          initialize_cache,
          task=task_reg[task_ids[0]],
          model=pred_model,
          input_spec=input_spec,
          tokenizer_fn=tokenizer_dict['tokenizer'],
          config=config),
      axis_name='device'
  )

  log_sample_size = config.logging.get('log_sample_size', 2)

  eval_metrics, batch_samples, batch_sample_list = None, None, []
  for ckpt_path in ckpt_list:
    # Restores the model
    if not gfile.exists(ckpt_path):
      logging.warn(
          'Unable to evaluate ckpt %s because it does not exist. '
          'If this is a parallel evaluation job, try to increase '
          'config.logging.checkpoint_kept or use more accelerators.', ckpt_path)
      continue
    task = task_reg[next(task_ids_iter)]
    train_state = train_utils.restore_checkpoint(
        ckpt_path,
        train_state,
        is_legacy_checkpoint=config.eval_from.get('legacy_checkpoint', False))
    global_step = int(train_state.global_step)
    ckpt_rng = jax.random.fold_in(rng, global_step)
    start_step = global_step  # Step offset for logging
    train_state_replicated = jax_utils.replicate(train_state)
    logging.info('Starting evaluation at global step %d', global_step)

    is_final_eval = False
    if global_step == total_train_steps or config.get('eval_only', False):
      is_final_eval = True
    data_iters = eval_utils.get_eval_data_iters(dataset, config, is_final_eval)
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
      metric_suffix = data_split + config.eval.get('suffix', '')

      eval_features = []
      with metric_writers.ensure_flushes(writer):
        for step in range(num_steps):
          eval_batch = next(data_iter)
          ckpt_rng, step_rng = jax.random.split(ckpt_rng, 2)
          cache = init_cache_pmapped(eval_batch['inputs'])
          step_rng = jax.random.split(step_rng, jax.local_device_count())
          batch_features, batch_outputs = eval_step_pmapped(
              train_state_replicated, eval_batch, cache, task, step_rng)
          batch_features = jax.device_get(jax_utils.unreplicate(batch_features))
          eval_features.append(batch_features)
          # Quick indication that training is happening.
          logging.log_first_n(logging.INFO, 'Finished evaluation step %d.', 5,
                              step)
          for h in hooks:
            h(start_step + step)
          if config.eval.get('return_samples', False):
            batch_samples = jax.device_get(jax_utils.unreplicate(batch_outputs))
            batch_sample_list.append(batch_samples)
        start_step += num_steps
        eval_features = eval_utils.gather_outputs_with_mask(
            eval_features, num_samples=total_num_examples)
        eval_metrics = eval_utils.compute_eval_metrics(eval_features, config,
                                                       total_num_examples,
                                                       num_repeats)
        eval_metrics = {
            f'eval_{metric_suffix}/{k}': v for k, v in eval_metrics.items()
        }
        writer.write_scalars(global_step, eval_metrics)
        # Sample from last batch.
        batch_samples = jax.device_get(jax_utils.unreplicate(batch_outputs))
        log_samples(batch_samples, writer, f'eval_{metric_suffix}', global_step,
                    log_sample_size,
                    config.get('dataset_type', 'video') == 'video')

    del train_state_replicated
    ckpt_manager.add_eval_result(ckpt_path, eval_metrics, 0)
    logging.info('Finishing evaluation at global step %d', global_step)

  # Wait until computations are done before exiting.
  jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()
  # Return the train and eval summary after last step for regression testing.
  if config.eval.get('return_samples', False):
    samples = jax.tree_util.tree_map(lambda *x: np.concatenate(x),
                                     *batch_sample_list)
  else:
    samples = batch_samples
  return train_state, eval_metrics, samples
