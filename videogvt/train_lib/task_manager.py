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

"""Auxiliary class and functions for eval jobs."""

import csv
import os
import time
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

from absl import logging
import jax
import tensorflow as tf
from tensorflow.io import gfile


class TaskManager:
  """Class for checking the model folder repeately for evaluation."""

  def __init__(self, model_dir: str) -> None:
    self._model_dir = model_dir

  @property
  def model_dir(self) -> str:
    return self._model_dir

  def mark_training_done(self) -> None:
    if jax.process_index() == 0:
      with tf.io.gfile.GFile(
          os.path.join(self.model_dir, 'TRAIN_DONE'), 'w'
      ) as f:
        f.write('')

  def is_training_done(self) -> None:
    return tf.io.gfile.exists(os.path.join(self.model_dir, 'TRAIN_DONE'))

  def add_eval_result(self,
                      checkpoint_path: str,
                      result_dict: Dict[str, Any],
                      default_value: int = -1) -> None:
    pass

  def _get_checkpoints_with_results(self):
    return set()

  def newest_checkpoint(self,
                        timeout: int = 3600 * 8,
                        include_tmp: bool = False) -> Iterable[str]:
    """Yield the newest unevaluated checkpoint.

    Args:
      timeout: Optional timeout for waiting for new checkpoints. Set this to do
        continious evaluation.
      include_tmp: Whether to evaluate tmp checkpoints.

    Yields:
      Path to the latest unevaluated checkpoint.
    """

    def _find_latest_checkpoint(
        checkpoints: Iterable[str]) -> Tuple[int, Optional[str]]:
      sorted_ckpts = sorted((int(x.split('_')[-1]), x) for x in checkpoints)
      if sorted_ckpts:
        return sorted_ckpts[-1]
      else:
        return -1, None

    logging.info('Looking for checkpoints in %s', self._model_dir)
    evaluated_checkpoints = self._get_checkpoints_with_results()
    last_eval_step, _ = _find_latest_checkpoint(evaluated_checkpoints)

    last_eval = time.time()
    while not (time.time() - last_eval > timeout or self.is_training_done()):
      # Check if directory exists. The train job may only create the directory
      # some time after the test job starts.
      if tf.io.gfile.exists(self.model_dir):
        checkpoints = tf.io.gfile.glob(
            os.path.join(self._model_dir, 'checkpoint*')
        )
        if include_tmp:
          checkpoints = set(checkpoints)
        else:
          checkpoints = set([x for x in checkpoints if 'tmp' not in x])
        last_step, last_ckpt = _find_latest_checkpoint(checkpoints)

        if last_step > last_eval_step:
          # update the time *before* yielding to prevent long evals from
          # triggering the loop breaking condition.
          last_eval = time.time()
          last_eval_step = last_step
          yield last_ckpt

      time.sleep(5)

  def unevaluated_checkpoints(self,
                              timeout: int = 3600 * 8,
                              num_batched_steps: int = 1,
                              eval_every_steps: Optional[int] = None,
                              include_tmp: bool = False) -> Iterable[str]:
    """Generator for checkpoints without evaluation results.

    Args:
      timeout: Optional timeout for waiting for new checkpoints. Set this to do
        continious evaluation.
      num_batched_steps: Steps that are batched into a single tf.function.
        Required for computing correct evaluation checkpoints.
      eval_every_steps: Only evaluate checkpoints from steps divisible by this
        integer.
      include_tmp: Whether to evaluate tmp checkpoints.

    Yields:
      Path to checkpoints that have not yet been evaluated.
    """
    logging.info('Looking for checkpoints in %s', self._model_dir)
    evaluated_checkpoints = self._get_checkpoints_with_results()
    last_eval = time.time()
    while True:
      # Check if directory exists. The train job may only create the directory
      # some time after the test job starts.
      if not tf.io.gfile.exists(self.model_dir):
        logging.info('Directory %s does not exist!', self.model_dir)
      else:
        logging.info(
            'what is in %s:  are  %s',
            self.model_dir,
            tf.io.gfile.listdir(self.model_dir),
        )
        unevaluated_checkpoints = []
        checkpoints = tf.io.gfile.glob(
            os.path.join(self._model_dir, 'checkpoint*')
        )
        if include_tmp:
          checkpoints = set(checkpoints)
        else:
          checkpoints = set([x for x in checkpoints if 'tmp' not in x])
        logging.info('checkpoints: %s', checkpoints)
        unevaluated_checkpoints = checkpoints - evaluated_checkpoints
        step_and_ckpt = sorted(
            (int(x.split('_')[-1]), x) for x in unevaluated_checkpoints
        )

        unevaluated_checkpoints = []
        for step, ckpt in step_and_ckpt:
          if eval_every_steps:
            if step > num_batched_steps and (step % eval_every_steps <
                                             num_batched_steps):
              unevaluated_checkpoints.append(ckpt)
          else:
            unevaluated_checkpoints.append(ckpt)

        logging.info(
            (
                'Found checkpoints: %s\nEvaluated checkpoints: %s\n'
                'Unevaluated checkpoints: %s'
            ),
            checkpoints,
            evaluated_checkpoints,
            unevaluated_checkpoints,
        )
        for checkpoint_path in unevaluated_checkpoints:
          yield checkpoint_path

        if unevaluated_checkpoints:
          evaluated_checkpoints |= set(unevaluated_checkpoints)
          last_eval = time.time()
          continue
      if time.time() - last_eval > timeout or self.is_training_done():
        break
      time.sleep(5)


class TaskManagerWithCsvResults(TaskManager):
  """Task Manager that writes results to a CSV file."""

  def __init__(self, model_dir: str, score_file: Optional[str] = None) -> None:
    super().__init__(model_dir)
    if score_file is None:
      score_file = os.path.join(self._model_dir, 'scores.csv')
    else:
      score_file = os.path.join(self._model_dir, score_file)
    self._score_file = score_file

  def _get_checkpoints_with_results(self):
    """Return the checkpoints as set."""
    if not tf.io.gfile.exists(self._score_file):
      return set()
    with tf.io.gfile.GFile(self._score_file) as f:
      reader = csv.DictReader(f)
      return {r['checkpoint_path'] for r in reader}
    return set()

  def add_eval_result(self, checkpoint_path: str, result_dict: Dict[str, Any],
                      default_value: int = -1) -> None:
    """Add eval result to the CSV file."""
    if jax.process_index() == 0:
      step = int(os.path.basename(checkpoint_path).split('_')[-1])
      csv_header = ['checkpoint_path', 'step'] + sorted(result_dict)
      write_header = not tf.io.gfile.exists(self._score_file)
      if write_header:
        with tf.io.gfile.GFile(self._score_file, 'w') as f:
          writer = csv.DictWriter(
              f, fieldnames=csv_header, extrasaction='ignore'
          )
          writer.writeheader()
      row = dict(checkpoint_path=checkpoint_path, step=str(step))
      for k, v in result_dict.items():
        if isinstance(v, float):
          v = '{:.3f}'.format(v)
        row[k] = v
      with tf.io.gfile.GFile(self._score_file, 'a') as f:
        writer = csv.DictWriter(f, fieldnames=csv_header, extrasaction='ignore')
        writer.writerow(row)


class CustomTaskManager(TaskManagerWithCsvResults):
  """Task Manager that writes results to a CSV file."""

  def __init__(
      self,
      model_dir: str,
      work_dir: Optional[str] = None,
      file_pattern: str = 'checkpoint*',
      sort_key_fn: Callable[[str], int] = lambda x: int(x.split('_')[-1]),
  ) -> None:
    super().__init__(model_dir)
    self.file_pattern = file_pattern
    self.sort_key_fn = sort_key_fn
    if work_dir is not None:
      score_file = os.path.join(work_dir, 'scores.csv')
      self._score_file = score_file

  def add_eval_result(
      self,
      checkpoint_path: str,
      result_dict: Dict[str, Any],
      default_value: int = -1,
  ) -> None:
    """Add eval result to the CSV file."""
    if jax.process_index() == 0:
      try:
        step = self.sort_key_fn(checkpoint_path)
      except ValueError:
        step = -1
      csv_header = ['checkpoint_path', 'step'] + sorted(result_dict)
      write_header = not gfile.exists(self._score_file)
      if write_header:
        with gfile.GFile(self._score_file, 'w') as f:
          writer = csv.DictWriter(
              f, fieldnames=csv_header, extrasaction='ignore'
          )
          writer.writeheader()
      row = dict(checkpoint_path=checkpoint_path, step=str(step))
      for k, v in result_dict.items():
        if isinstance(v, float):
          v = '{:.3f}'.format(v)
        row[k] = v
      if not self._score_file.startswith('/xfile'):
        with gfile.GFile(self._score_file, 'a') as f:
          writer = csv.DictWriter(
              f, fieldnames=csv_header, extrasaction='ignore'
          )
          writer.writerow(row)
      else:  # xfile doesn't support append mode
        with gfile.GFile(self._score_file, 'w+') as f:
          f.write(f.read())
          writer = csv.DictWriter(
              f, fieldnames=csv_header, extrasaction='ignore'
          )
          writer.writerow(row)

  def unevaluated_checkpoints(
      self,
      timeout: int = 3600 * 8,
      num_batched_steps: int = 1,
      eval_every_steps: Optional[int] = None,
      include_tmp: bool = False,
      return_all: bool = False,
  ) -> Iterable[str]:
    """Generator for checkpoints without evaluation results.

    Args:
      timeout: Optional timeout for waiting for new checkpoints. Set this to do
        continious evaluation.
      num_batched_steps: Steps that are batched into a single tf.function.
        Required for computing correct evaluation checkpoints.
      eval_every_steps: Only evaluate checkpoints from steps divisible by this
        integer.
      include_tmp: Whether to evaluate tmp checkpoints.
      return_all: Whether to return all checkpoints including evaluated ones.

    Yields:
      Path to checkpoints that have not yet been evaluated.
    """
    logging.info('Looking for checkpoints in %s', self.model_dir)
    if not return_all:
      evaluated_checkpoints = self._get_checkpoints_with_results()
    else:
      evaluated_checkpoints = set()
    last_eval = time.time()
    while True:
      # Check if directory exists. The train job may only create the directory
      # some time after the test job starts.
      if not gfile.exists(self.model_dir):
        logging.info('Directory %s does not exist!', self.model_dir)
      else:
        logging.info(
            'what is in %s:  are  %s',
            self.model_dir,
            gfile.listdir(self.model_dir),
        )
        unevaluated_checkpoints = []
        checkpoints = gfile.glob(
            os.path.join(self._model_dir, self.file_pattern)
        )
        if include_tmp:
          checkpoints = set(checkpoints)
        else:
          checkpoints = set(
              [x for x in checkpoints if 'tmp' not in os.path.basename(x)]
          )
        logging.info('checkpoints: %s', checkpoints)
        unevaluated_checkpoints = checkpoints - evaluated_checkpoints
        step_and_ckpt = sorted(
            (self.sort_key_fn(x), x) for x in unevaluated_checkpoints
        )

        unevaluated_checkpoints = []
        for step, ckpt in step_and_ckpt:
          if eval_every_steps:
            if step > num_batched_steps and (
                step % eval_every_steps < num_batched_steps
            ):
              unevaluated_checkpoints.append(ckpt)
          else:
            unevaluated_checkpoints.append(ckpt)

        logging.info(
            (
                'Found checkpoints: %s\nEvaluated checkpoints: %s\n'
                'Unevaluated checkpoints: %s'
            ),
            checkpoints,
            evaluated_checkpoints,
            unevaluated_checkpoints,
        )
        for checkpoint_path in unevaluated_checkpoints:
          yield checkpoint_path

        if unevaluated_checkpoints:
          evaluated_checkpoints |= set(unevaluated_checkpoints)
          last_eval = time.time()
          continue
      if time.time() - last_eval > timeout or self.is_training_done():
        break
      time.sleep(5)
